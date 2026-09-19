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
import urllib.request
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
# need = 下完**必须存在**的文件：hf_hub 在"远端不可达但 local_dir 已有文件"时会
#        直接返回、不抛异常（见 download_hf 注释），不逐文件校验就会把"没下到"报成 OK。
HF_MODELS = {
    # Whisper large-v3-turbo（faster-whisper 版）：全项目唯一 whisper 权重，**识别必需**。
    # ⚠ repo id 不能想当然写 Systran/...：2026-09-17 实测 hf-mirror / aifasthub 对它一律
    #   **401**（resolve 接口 404 RepoNotFound），官方 hf.co 在开发机连不上。开发时实际用的
    #   是这个 **dropbox-dash** 仓库（就是 models/ 里那份权重的 README 自己写的来源），
    #   hf-mirror 上 200、文件集一致（model.bin 1543MB + config/tokenizer/vocabulary）。
    "whisper_turbo": {
        "repo": "dropbox-dash/faster-whisper-large-v3-turbo",
        "dest": "faster-whisper-large-v3-turbo",
        "need": ["model.bin", "config.json", "tokenizer.json"],
        # 保险丝：hf-mirror 万一也挂，回退 ModelScope 上的同款权重（实测 200，文件集一致）。
        "ms_fallback": "pengzhendong/faster-whisper-large-v3-turbo",
    },
    # Qwen3-ASR-0.6B 的 ONNX INT4 导出：**备选识别后端**（设置里选 Qwen3-ASR 才需要）。
    # ⚠ 源必须是 ONNX 导出方 `andrewleech/qwen3-asr-0.6b-onnx`：它的 config.json 里有
    #   `mel` 与 `special_tokens` 两段，模型类直接读（asr_qwen_onnx.py）。
    #   曾经错写成 `vrfai/Qwen3-ASR-0.6B-int4` —— 那是**transformers 格式**的同名仓库，
    #   文件集里根本没有 encoder.int4.onnx，却会把 config.json 覆盖成 transformers 版
    #   → 运行期 `KeyError: 'mel'`，且多下 925MB 用不上的 safetensors（2026-09-18 实锤）。
    #   patterns 只取运行需要的文件：该仓库另有 3.4GB 的 tar.gz 打包件，不下。
    "qwen3_asr_0_6b_onnx_int4": {
        "repo": "andrewleech/qwen3-asr-0.6b-onnx",
        "dest": "qwen3-asr-0.6b-onnx-int4",
        "patterns": ["config.json", "preprocessor_config.json", "added_tokens.json",
                     "tokenizer.json", "encoder.int4.onnx", "encoder.int4.onnx.data",
                     "decoder_init.int4.onnx", "decoder_step.int4.onnx",
                     "embed_tokens.bin", "decoder_weights.int4.data"],
        "need": ["config.json", "tokenizer.json", "encoder.int4.onnx",
                 "decoder_init.int4.onnx", "decoder_step.int4.onnx",
                 "embed_tokens.bin", "decoder_weights.int4.data"],
    },
}


# ---------------------------------------------------------------- ModelScope 模型清单
# hf-mirror 限流/不可用时走 modelscope.cn（国内直连快）
MS_MODELS = {
    # FSMN-VAD（可选 VAD 引擎，vad_engine="fsmn"）：CPU/ONNX int8，约 0.5MB
    "fsmn_vad": {
        "repo": "damo/speech_fsmn_vad_zh-cn-16k-common-onnx",
        "dest": "fsmn-vad-onnx",
        "need": ["model_quant.onnx"],
    },
    # Qwen2.5-1.5B 官方权重（翻译模型的 **CT2 转换源**；转换见 scripts/convert_qwen_ct2.py）
    "qwen_1_5b_hf": {
        "repo": "Qwen/Qwen2.5-1.5B-Instruct",
        "dest": "Qwen2.5-1.5B-Instruct-hf",
        "need": ["model.safetensors", "tokenizer_config.json"],
    },
    # Qwen3-1.7B 官方权重（质量档翻译的 CT2 转换源；可选）
    "qwen3_1_7b_hf": {
        "repo": "Qwen/Qwen3-1.7B",
        "dest": "Qwen3-1.7B-hf",
        "need": ["model.safetensors", "tokenizer_config.json"],
    },
    # 接缝标点的 CPU 小模型（可选；没装时 punct_engine=auto 自动走 Qwen，功能不受影响）
    "punc_cpu": {
        "repo": "iic/punc_ct-transformer_zh-cn-common-vocab272727-onnx",
        "dest": "punc-ct-transformer-zh-en-onnx",
        "need": ["model_quant.onnx", "config.yaml"],
    },
    # FireRedVAD 的 Stream-VAD 权重（**默认 VAD 的导出源**，2.3MB，Apache-2.0）：
    # 下载后由 scripts/export_fireredvad_onnx.py 导出成 models/fireredvad-onnx（≈2MB）
    "firered": {
        "repo": "FireRedTeam/FireRedVAD",
        "dest": "fireredvad",
        "need": ["Stream-VAD/model.pth.tar", "Stream-VAD/cmvn.ark"],
    },
    # 腾讯混元 Hy-MT2-1.8B（**默认翻译引擎 hymt2 的权重**，Apache-2.0，约 4.1GB）：
    # 架构 HunYuanDenseV1ForCausalLM 不被 CTranslate2 支持 → 只能走 transformers/PyTorch，
    # 所以它需要 torch（安装器会一起装）。need 必须是真权重（3888MB 的 model.safetensors），
    # 不能只看 config.json —— 那只证明"目录建出来了"。
    "hymt2": {
        "repo": "Tencent-Hunyuan/Hy-MT2-1.8B",
        "dest": "Hy-MT2-1.8B",
        "need": ["model.safetensors", "tokenizer.json"],
    },
}

# GitHub 资产（url 相对 github.com）
GITHUB_ASSETS = {
    "silero_vad": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
        "dest": "silero_vad.onnx",
        # 校验门槛：代理链在直连失败时会回一个 **假 200 的 HTML 错误页**
        # （2026-09-17 实测：6,990 字节、以 `<!-- 62982819547` 开头）。
        # 只判"文件非空"会把垃圾当模型装进 models/，直到运行期加载 VAD 才炸。
        # 真文件 643,854 字节（ONNX/protobuf 二进制）。
        "min_size": 500_000,
    },
}


# ---------------------------------------------------------------- 通道测速
# 同一份权重 HF 镜像与 ModelScope 常常都有，但速度能差好几倍
# （2026-09-18 实测：hf-mirror 单连接 ~3MB/s，ModelScope ~18MB/s → 1.5GB 的 whisper
# 相差约 7 分钟）。所以有双通道的项先各读几 MB 量一下，明显更快的那边先用。
# 只在"快 2 倍以上"才换源：默认仍以**开发时验证过的 HF 源**为准。
CHANNEL_PROBE = {
    "hf": "https://hf-mirror.com/dropbox-dash/faster-whisper-large-v3-turbo/resolve/main/model.bin",
    "ms": ("https://modelscope.cn/api/v1/models/Tencent-Hunyuan/Hy-MT2-1.8B/repo"
           "?Revision=master&FilePath=model.safetensors"),
}
PROBE_MB = 4                 # 每个通道最多读这么多
PROBE_SECONDS = 5.0          # 每个通道最多花这么多秒
SWITCH_FACTOR = 2.0          # ModelScope 至少快 2 倍才优先用它
_SPEED = {}                  # 测速结果缓存


def host_speed(url: str) -> float:
    """读最多 PROBE_MB / PROBE_SECONDS 估算 MB/s；失败或读不到数据返回 0。"""
    got = 0
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MaiSubtitle-download"})
        with urllib.request.urlopen(req, timeout=8) as r:
            while got < PROBE_MB * 1024 * 1024:
                chunk = r.read(262144)
                if not chunk:
                    break
                got += len(chunk)
                if time.perf_counter() - t0 > PROBE_SECONDS:
                    break
    except Exception:
        pass
    dt = time.perf_counter() - t0
    if dt <= 0 or got == 0:
        return 0.0
    return got / 1048576 / dt


def prefer_model_scope(spec: dict) -> bool:
    """有 ms_fallback 的项：ModelScope 明显更快就先试它（各自只测一次，进程内缓存）。"""
    if not spec.get("ms_fallback"):
        return False
    for key, url in CHANNEL_PROBE.items():
        if key not in _SPEED:
            _SPEED[key] = host_speed(url)
    hf_speed = _SPEED.get("hf", 0.0)
    ms_speed = _SPEED.get("ms", 0.0)
    print(f"  [测速] HF 镜像 {hf_speed:.1f} MB/s ｜ ModelScope {ms_speed:.1f} MB/s")
    return ms_speed > hf_speed * SWITCH_FACTOR


# ---------------------------------------------------------------- 下载判定

def spec_of(name: str) -> dict:
    return HF_MODELS.get(name) or MS_MODELS.get(name) or GITHUB_ASSETS.get(name) or {}


def looks_like_error_page(p: Path) -> bool:
    """像不像代理/网关塞回来的 HTML 错误页（而不是模型文件）。

    实测：github release 直连 502 时，代理会正常返回 200 + 一段 HTML
    （6,990 字节，以 `<!--` 开头）→ curl 退出码是 0，光看"文件非空"根本发现不了。
    真模型是二进制（ONNX/protobuf），既不会以 '<' 开头，也不会整段能 UTF-8 解码。
    """
    with p.open("rb") as _f:
        head = _f.read(512)
    if not head:
        return True
    if head.lstrip()[:1] == b"<":
        return True
    try:
        head.decode("utf-8")
    except UnicodeDecodeError:
        return False                 # 二进制 → 不是错误页
    return True                      # 纯文本 → 当错误页处理


def missing_files(name: str) -> list:
    """目标里还缺什么；空列表 = 齐了。

    · 有 need 的**逐文件**查（HF 模型：只下了一半也算缺）；
    · 文件型资产查"体积够不够 + 内容像不像模型"（防假 200 错误页）；
    · 其余只要求"存在且非空"。
    """
    spec = spec_of(name)
    if not spec:
        return ["(未知项)"]
    dest = MODELS_DIR / spec["dest"]
    if not dest.exists():
        return ["(目录不存在)"]
    need = spec.get("need")
    if need:
        return [f for f in need
                if not (dest / f).is_file() or (dest / f).stat().st_size == 0]
    if dest.is_file():
        size = dest.stat().st_size
        if size == 0:
            return ["(空文件)"]
        if size < spec.get("min_size", 0):
            return [f"(体积异常 {size} 字节)"]
        if looks_like_error_page(dest):
            return ["(内容不是模型，疑似代理错误页)"]
        return []
    return [] if any(dest.iterdir()) else ["(目录为空)"]


def target_present(name: str) -> bool:
    """目标是否真的可用（防 state 说谎：本地被删/移走，或当初只下了一半）。

    state 只记"下载成功过"，从不清理；所以除了"目录在不在"，HF 模型还要看关键文件。
    """
    return not missing_files(name)


def download_ms(name: str, spec: dict) -> bool:
    """ModelScope 快照下载。同样要逐文件校验 —— 中断/限流会留下半份目录，
    而"目录非空"看起来和"下好了"一模一样。"""
    from modelscope import snapshot_download as ms_download
    local = MODELS_DIR / spec["dest"]
    for attempt in range(3):
        try:
            ms_download(spec["repo"], local_dir=str(local))
            miss = missing_files(name)
            if not miss:
                return True
            print(f"  [retry {attempt + 1}/3] {name}: 下完仍缺 {miss}")
        except Exception as e:
            print(f"  [retry {attempt + 1}/3] {name}: {str(e)[:150]}")
        time.sleep(5 * (attempt + 1))
    print(f"  [fail] {name}")
    return False


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            # 截断/损坏的状态文件只影响"哪些项被记成已下载"—— 下载器本来就会
            # 逐项真查文件（missing_files），所以当空表重来即可，别把下载链搞断
            print(f"[注意] {STATE_FILE.name} 损坏，按空状态处理（会重新核对每项）")
    return {}


def save_state(state: dict):
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, STATE_FILE)


def download_hf(name: str, spec: dict) -> bool:
    """HF 快照下载：镜像（env HF_ENDPOINT，缺省 hf-mirror.com）→ 官方源 → ModelScope 镜像。

    ⚠ hf_hub 的坑（2026-09-17 实测）：远端不可达但 local_dir 里已有文件时，
    `snapshot_download` **不抛异常**，只打印一句 "Returning existing local_dir `…`
    as remote repo cannot be accessed" 就正常返回（见 huggingface_hub/_snapshot_download.py）。
    所以每次调用后必须 missing_files() 真查一遍文件，否则会把"其实什么都没下到"报成 OK。
    """
    from huggingface_hub import snapshot_download
    local = MODELS_DIR / spec["dest"]
    local.mkdir(parents=True, exist_ok=True)
    channels = [(None, "镜像"), ("https://huggingface.co", "官方源")]   # None = 用 env 里的镜像
    for ep, label in channels:
        for attempt in range(2):
            try:
                kw = {"allow_patterns": spec.get("patterns") or None, "max_workers": 4}
                if ep:
                    kw["endpoint"] = ep
                snapshot_download(repo_id=spec["repo"], local_dir=local, **kw)
                miss = missing_files(name)
                if not miss:
                    if ep:
                        print(f"  [ok] {name}（回退官方源成功）")
                    return True
                print(f"  [warn] {label} 跑完仍缺 {miss}")
                break                      # 同一通道再试也是同样结果 → 换下一个
            except Exception as e:
                print(f"  [retry {label} {attempt + 1}/2] {name}: {str(e)[:120]}")
                time.sleep(3 * (attempt + 1))
    ms_repo = spec.get("ms_fallback")
    if ms_repo:
        print(f"  [hf 不通] {name} → 回退 ModelScope 镜像 {ms_repo}")
        if download_ms(name, {"repo": ms_repo, "dest": spec["dest"]}):
            miss = missing_files(name)
            if not miss:
                print(f"  [ok] {name}（ModelScope 镜像成功）")
                return True
            print(f"  [warn] ModelScope 镜像跑完仍缺 {miss}")
    miss = missing_files(name)
    print(f"  [fail] {name}" + (f"（缺 {'、'.join(miss)}）" if miss else ""))
    print(f"         手动补救：把该模型的权重文件放进 {MODELS_DIR / spec['dest']}")
    return False


def gh_download(name: str, spec: dict, tries: int = 3) -> bool:
    """GitHub 资产下载：直连失败自动切代理链。

    每次下完都过 missing_files() 那套校验（体积 + 内容），不合格就删掉、换下一个代理 ——
    curl 的 `--fail` 只挡得住 4xx/5xx，挡不住"200 + HTML 错误页"。
    """
    url = spec["url"]
    dest = MODELS_DIR / spec["dest"]
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
                if r.returncode == 0 and not missing_files(name):
                    return True
                if dest.exists():
                    dest.unlink()            # 别把半截/假文件留在 models/ 里
                if r.returncode == 0:
                    print(f"    [warn] {proxy or 'direct'} 返回的不是模型文件，换下一个通道")
            except Exception:
                pass
        time.sleep(3 * (attempt + 1))
    return False


def download_github_asset(name: str, spec: dict) -> bool:
    if not missing_files(name):              # 已下好（含体积/内容校验）
        return True
    ok = gh_download(name, spec)
    if ok:
        print(f"  [ok] {name} -> {spec['dest']} "
              f"({(MODELS_DIR / spec['dest']).stat().st_size // 1024} KB)")
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
        # 前面可能留着 hf_hub 的进度条（tqdm 用 \r 重写当前行、不换行），
        # 直接 print 会和它挤在同一行（实测：`Download complete: …| 1.62GB [down] silero_vad …`）。
        # 先换一行再打印，日志就是干净的。
        print(f"\n[down] {name} ...", flush=True)
        if name in HF_MODELS:
            spec = HF_MODELS[name]
            ok = False
            if prefer_model_scope(spec):
                print(f"  [换源] ModelScope 快得多 → 先用它下 {name}")
                ok = download_ms(name, {"repo": spec["ms_fallback"], "dest": spec["dest"]})
            if not ok:
                ok = download_hf(name, spec)
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
