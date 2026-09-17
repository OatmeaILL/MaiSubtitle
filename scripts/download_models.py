"""模型下载助手（发布版：覆盖"装上就能跑"所需的全部权重）。

- HF 模型默认走 **hf-mirror.com**（可用 HF_ENDPOINT 覆盖；镜像失败自动回退 huggingface.co）
- ModelScope 模型走 modelscope.cn（国内直连快）
- GitHub 资产走代理链（direct → ghfast.top → gh-proxy.com → ghproxy.cn）
- 断点续传（hf_hub 自带 incomplete resume），已下过的自动跳过（state 会校验本地是否真在，见 target_present）
- 用法：
    uv run python scripts/download_models.py                     # 全部（含可选项）
    uv run python scripts/download_models.py --only whisper_turbo silero_vad
  必需项（默认档）：whisper_turbo（识别权重）+ silero_vad；翻译模型见 model_manager 的说明
  （Qwen 权重 → CT2 转换，或直接把现成 CT2 目录放进 models/）。
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.environ.get("MAISUB_MODELS_DIR", PROJECT_ROOT / "models"))
STATE_FILE = MODELS_DIR / "download_state.json"
LOGS_DIR = PROJECT_ROOT / "logs"

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
# hf-mirror 无法代理 Xet CAS 存储，禁用后走普通 HTTP resolve 通道
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# GitHub 直连不通时的代理链
GH_PROXIES = ["", "https://ghfast.top/", "https://gh-proxy.com/", "https://ghproxy.cn/"]


# ---------------------------------------------------------------- HF 模型清单
# dest 相对 MODELS_DIR；patterns 为空表示整个仓库快照
HF_MODELS = {
    # Whisper large-v3-turbo（faster-whisper 版）：全项目唯一 whisper 权重，**识别必需**。
    "whisper_turbo": {
        "repo": "Systran/faster-whisper-large-v3-turbo",
        "dest": "faster-whisper-large-v3-turbo",
    },
    # Qwen3-ASR-0.6B 的 ONNX INT4 导出：**备选识别后端**（设置里选 Qwen3-ASR 才需要）。
    # 只存在于 HuggingFace（ModelScope 无镜像）。
    "qwen3_asr_0_6b_onnx_int4": {
        "repo": "vrfai/Qwen3-ASR-0.6B-int4",
        "dest": "qwen3-asr-0.6b-onnx-int4",
    },
}


# ---------------------------------------------------------------- ModelScope 模型清单
# hf-mirror 限流/不可用时走 modelscope.cn（国内直连快）
MS_MODELS = {
    # FSMN-VAD（可选 VAD 引擎，vad_engine="fsmn"）：CPU/ONNX int8，约 0.5MB
    "fsmn_vad": {
        "repo": "damo/speech_fsmn_vad_zh-cn-16k-common-onnx",
        "dest": "fsmn-vad-onnx",
    },
    # Qwen2.5-1.5B 官方权重（翻译模型的 **CT2 转换源**；转换见 scripts/convert_qwen_ct2.py）
    "qwen_1_5b_hf": {
        "repo": "Qwen/Qwen2.5-1.5B-Instruct",
        "dest": "Qwen2.5-1.5B-Instruct-hf",
    },
    # Qwen3-1.7B 官方权重（质量档翻译的 CT2 转换源；可选）
    "qwen3_1_7b_hf": {
        "repo": "Qwen/Qwen3-1.7B",
        "dest": "Qwen3-1.7B-hf",
    },
    # 接缝标点的 CPU 小模型（可选；没装时 punct_engine=auto 自动走 Qwen，功能不受影响）
    "punc_cpu": {
        "repo": "iic/punc_ct-transformer_zh-cn-common-vocab272727-onnx",
        "dest": "punc-ct-transformer-zh-en-onnx",
    },
}

# GitHub 资产（url 相对 github.com）
GITHUB_ASSETS = {
    "silero_vad": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
        "dest": "silero_vad.onnx",
    },
}


def target_present(name: str) -> bool:
    """目标是否真实存在（防 state 说谎：本地文件被删/移走后 state 仍记 "ok"）。

    state 只记"下载成功过"，不做清理；若模型目录被手动删除，必须重新下载。
    """
    if name in HF_MODELS:
        p = MODELS_DIR / HF_MODELS[name]["dest"]
    elif name in MS_MODELS:
        p = MODELS_DIR / MS_MODELS[name]["dest"]
    elif name in GITHUB_ASSETS:
        p = MODELS_DIR / GITHUB_ASSETS[name]["dest"]
    else:
        return False
    if p.is_file():
        return p.stat().st_size > 0
    return p.is_dir() and any(p.iterdir())


def download_ms(name: str, spec: dict) -> bool:
    from modelscope import snapshot_download as ms_download
    local = MODELS_DIR / spec["dest"]
    for attempt in range(3):
        try:
            ms_download(spec["repo"], local_dir=str(local))
            return True
        except Exception as e:
            print(f"  [retry {attempt + 1}/3] {name}: {str(e)[:150]}")
            time.sleep(5 * (attempt + 1))
    print(f"  [fail] {name}")
    return False


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def download_hf(name: str, spec: dict) -> bool:
    """HF 快照下载：默认走镜像（env HF_ENDPOINT，缺省 hf-mirror.com），失败回退官方源。"""
    from huggingface_hub import snapshot_download
    local = MODELS_DIR / spec["dest"]
    local.mkdir(parents=True, exist_ok=True)
    endpoints = [None, "https://huggingface.co"]      # None = 用 env 里的镜像
    for ep in endpoints:
        for attempt in range(2):
            try:
                kw = {"allow_patterns": spec.get("patterns") or None, "max_workers": 4}
                if ep:
                    kw["endpoint"] = ep
                snapshot_download(repo_id=spec["repo"], local_dir=local, **kw)
                if ep:
                    print(f"  [ok] {name}（回退官方源成功）")
                return True
            except Exception as e:
                if spec.get("optional"):
                    print(f"  [skip] {name}: {str(e)[:120]}")
                    return True
                print(f"  [retry {ep or '镜像'} {attempt + 1}/2] {name}: {str(e)[:120]}")
                time.sleep(3 * (attempt + 1))
    print(f"  [fail] {name}")
    return False


def gh_download(url: str, dest: Path, tries: int = 3) -> bool:
    """GitHub 资产下载：直连失败自动切代理链。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(tries):
        for proxy in GH_PROXIES:
            full = proxy + url
            try:
                r = subprocess.run(
                    ["curl", "-L", "--fail", "--retry", "2", "-m", "1800",
                     "-o", str(dest), full],
                    capture_output=True, text=True,
                )
                if r.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
                    # 代理可能返回假 200 错误页：校验大小与 zip 完整性
                    size = dest.stat().st_size
                    if size < 1024 * 1024 and url.endswith(".zip"):
                        print(f"    [warn] {proxy} 返回疑似错误页 ({size}B)，跳过")
                        continue
                    return True
            except Exception:
                pass
        time.sleep(3 * (attempt + 1))
    return False


def download_github_asset(name: str, spec: dict) -> bool:
    dest = MODELS_DIR / spec["dest"]
    if dest.exists() and dest.stat().st_size > 0:
        return True
    ok = gh_download(spec["url"], dest)
    if ok:
        print(f"  [ok] {name} -> {dest.name} ({dest.stat().st_size // 1024} KB)")
    else:
        print(f"  [fail] {name}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="只下载指定项，如 --only whisper_turbo silero_vad")
    args = ap.parse_args()

    all_names = list(HF_MODELS) + list(MS_MODELS) + list(GITHUB_ASSETS)
    targets = args.only if args.only else all_names
    unknown = [t for t in targets if t not in all_names]
    if unknown:
        print(f"未知下载项: {unknown}\n可用: {all_names}")
        sys.exit(2)

    state = load_state()
    results = {}
    for name in targets:
        if state.get(name) == "ok" and target_present(name):
            print(f"[done] {name}（已下载过，跳过）")
            results[name] = True
            continue
        if state.get(name) == "ok":
            print(f"[stale] {name}（state 记为已下载，但本地不存在 → 重新下载）")
        print(f"[down] {name} ...")
        if name in HF_MODELS:
            ok = download_hf(name, HF_MODELS[name])
        elif name in MS_MODELS:
            ok = download_ms(name, MS_MODELS[name])
        elif name in GITHUB_ASSETS:
            ok = download_github_asset(name, GITHUB_ASSETS[name])
        if ok:
            state[name] = "ok"
            save_state(state)
        results[name] = ok

    failed = [k for k, v in results.items() if not v]
    print("\n==== 下载结果 ====")
    for k, v in results.items():
        print(f"  {'OK ' if v else 'FAIL'} {k}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
