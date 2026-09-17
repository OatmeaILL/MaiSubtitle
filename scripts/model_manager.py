"""MaiSubtitle 模型管理器：状态一览 / 下载 / 转换 / 删除。

用法：
    uv run python scripts/model_manager.py list             # 状态 + 缺什么、下一步怎么做
    uv run python scripts/model_manager.py download default # 下载默认档（必需项）
    uv run python scripts/model_manager.py download all     # 含可选项
    uv run python scripts/model_manager.py convert qwen_1_5b   # HF 权重 → CT2（翻译必需）
    uv run python scripts/model_manager.py delete <name>
    uv run python scripts/model_manager.py size

说明：识别/翻译的"能用"取决于四个必需项（silero_vad / whisper_turbo / qwen_1_5b /
qwen_1_5b_ct2）；其余是可选增强（更省 GPU 的标点、备选 VAD、备选识别、质量档翻译）。
"""
import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from maisubtitle.config import MODELS_DIR  # noqa: E402

# name → (下载项名 | "__local__" 手动放置 | "__converted__" 本工具转换,
#           展示名, 档位, 是否必需)
CATALOG = {
    "silero_vad":     ("silero_vad", "Silero VAD（语音分段）", "default", True),
    "whisper_turbo":  ("whisper_turbo", "Whisper large-v3-turbo（识别）", "default", True),
    "qwen_1_5b":      ("qwen_1_5b_hf", "Qwen2.5-1.5B 官方权重（仅转换时需要；转好可删）",
                       "default", False),
    "qwen_1_5b_ct2":  ("__converted__", "Qwen2.5-1.5B-CT2（备选翻译；快，约 0.22s/句）",
                       "quality", False),
    "firered":        ("firered", "FireRedVAD ONNX（默认 VAD）", "default", True),
    "hymt2":          ("hymt2", "Hy-MT2-1.8B（默认翻译引擎；需 torch）", "default", True),
    "punc_cpu":       ("punc_cpu", "接缝标点 CPU 模型（可选：省 GPU、中文 1~4ms）",
                       "quality", False),
    "fsmn_vad":       ("fsmn_vad", "FSMN-VAD（备选 VAD 引擎）", "quality", False),
    "qwen3_asr":      ("qwen3_asr_0_6b_onnx_int4",
                       "Qwen3-ASR-0.6B ONNX（备选识别后端；仅 HF 有）", "quality", False),
    "qwen3_1_7b_ct2": ("__converted__",
                       "Qwen3-1.7B-CT2（质量档翻译；需 HF 权重 + convert 1.7b）",
                       "quality", False),
}


def dir_size_mb(p: Path) -> float:
    if not p.exists():
        return 0.0
    if p.is_file():
        return p.stat().st_size / 1024 / 1024
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024 / 1024


def model_path(name: str) -> Path:
    paths = {
        "silero_vad": "silero_vad.onnx",
        "whisper_turbo": "faster-whisper-large-v3-turbo",
        "qwen_1_5b": "Qwen2.5-1.5B-Instruct-hf",
        "qwen_1_5b_ct2": "Qwen2.5-1.5B-Instruct-ct2",
        "firered": "fireredvad-onnx",
        "hymt2": "Hy-MT2-1.8B",
        "punc_cpu": "punc-ct-transformer-zh-en-onnx",
        "fsmn_vad": "fsmn-vad-onnx",
        "qwen3_asr": "qwen3-asr-0.6b-onnx-int4",
        "qwen3_1_7b_ct2": "Qwen3-1.7B-ct2",
    }
    return MODELS_DIR / paths[name]


def cmd_list():
    print(f"模型目录: {MODELS_DIR}\n")
    for tier in ("default", "quality"):
        title = "默认档（必需）" if tier == "default" else "可选增强"
        print(f"[{title}]")
        for name, (_dl, label, t, req) in CATALOG.items():
            if t != tier:
                continue
            mb = dir_size_mb(model_path(name))
            ok = "✔" if mb > 0.1 else ("✘ 必需" if req else "–  可选")
            print(f"  {ok:6s} {name:15s} {mb:8.0f} MB   {label}")
        print()
    missing = [n for n, e in CATALOG.items() if e[3] and dir_size_mb(model_path(n)) <= 0.1]
    if missing:
        dl_names = [CATALOG[n][0] for n in missing
                    if CATALOG[n][0] not in ("__local__", "__converted__")]
        print("⚠ 缺失的必需项：" + "、".join(CATALOG[n][1] for n in missing))
        if dl_names:
            print("  下载：python scripts/download_models.py --only " + " ".join(dl_names))
        if "qwen_1_5b_ct2" in missing:
            print("  转换：python scripts/model_manager.py convert qwen_1_5b"
                  "（需先有 qwen_1_5b 的 HF 权重）")
        if "firered" in missing:
            print("  FireRedVAD：先 scripts/download_models.py --only firered 下载权重，"
                  "再 scripts/model_manager.py export firered 导出 ONNX"
                  "（导出要 torch + fireredvad 包）")
        if "hymt2" in missing:
            print("  Hy-MT2-1.8B 约 4.1GB，下载后即用（走 transformers/PyTorch，需要 torch）")
    else:
        print("✔ 必需项齐全，直接启动即可（启动_MaiSubtitle.bat）")


def cmd_download(tier: str):
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    names = [n for n, e in CATALOG.items() if tier == "all" or e[2] == tier]
    to_run = []
    for n in names:
        dl = CATALOG[n][0]
        if dl in ("__converted__", "__local__"):
            continue
        if n == "qwen_1_5b" and dir_size_mb(model_path("qwen_1_5b_ct2")) > 100:
            continue          # 已经转换过 → 不再需要 HF 权重（省 3GB 下载）
        if dir_size_mb(model_path(n)) < 0.1:
            to_run.append(dl)
    if to_run:
        import subprocess
        print("下载:", to_run)
        r = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "download_models.py"),
                            "--only", *to_run])
        if r.returncode != 0:
            print("下载有失败项（网络/镜像问题），重跑即可断点续传")
    # 翻译模型：HF 权重已就位但 CT2 还没转换 → 自动转换（否则跑起来没译文）
    if (dir_size_mb(model_path("qwen_1_5b")) > 500
            and dir_size_mb(model_path("qwen_1_5b_ct2")) < 100):
        print("\n检测到 HF 权重但缺 CT2 版 → 自动转换 Qwen2.5-1.5B-CT2 …")
        import subprocess
        subprocess.run([sys.executable,
                        str(PROJECT_ROOT / "scripts" / "convert_qwen_ct2.py"), "1.5b"],
                       check=False)
    # FireRedVAD：权重下好了但 ONNX 还没导出 → 自动导出（否则 VAD 用不上 FireRed）
    if (dir_size_mb(MODELS_DIR / "fireredvad") > 1
            and dir_size_mb(model_path("firered")) <= 0.1):
        print("\n检测到 FireRedVAD 权重但缺 ONNX → 自动导出 …")
        import subprocess
        subprocess.run([sys.executable,
                        str(PROJECT_ROOT / "scripts" / "export_fireredvad_onnx.py")],
                       check=False)
    cmd_list()


def cmd_convert(name: str):
    import subprocess
    key = {"qwen_1_5b": "1.5b", "qwen3_1_7b_ct2": "1.7b"}.get(name)
    if key is None:
        print(f"无可转换项: {name}（可用: qwen_1_5b / qwen3_1_7b_ct2）")
        sys.exit(2)
    subprocess.run([sys.executable,
                    str(PROJECT_ROOT / "scripts" / "convert_qwen_ct2.py"), key], check=False)
    cmd_list()


def cmd_export(name: str):
    """把下载来的原始权重导出成运行时可用的 ONNX（目前只有 FireRedVAD）。"""
    if name != "firered":
        print(f"无可导出项: {name}（可用: firered）")
        sys.exit(2)
    import subprocess
    src = MODELS_DIR / "fireredvad"
    if dir_size_mb(src) < 1:
        print("缺权重：先运行 python scripts/download_models.py --only firered")
        sys.exit(2)
    subprocess.run([sys.executable,
                    str(PROJECT_ROOT / "scripts" / "export_fireredvad_onnx.py")], check=False)
    cmd_list()


def cmd_delete(name: str):
    if name not in CATALOG:
        print(f"未知模型: {name}，可选: {list(CATALOG)}")
        sys.exit(2)
    p = model_path(name)
    if p.exists():
        mb = dir_size_mb(p)
        shutil.rmtree(p, ignore_errors=True)
        print(f"已删除 {name}（释放 {mb:.0f} MB）")
    else:
        print("不存在")


def cmd_size():
    total = dir_size_mb(MODELS_DIR)
    print(f"models/ 总占用: {total / 1024:.2f} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "download", "convert", "export", "delete", "size"])
    ap.add_argument("target", nargs="?", default="all")
    a = ap.parse_args()
    if a.cmd == "list":
        cmd_list()
    elif a.cmd == "download":
        cmd_download(a.target if a.target in ("all", "default", "quality") else "all")
    elif a.cmd == "convert":
        cmd_convert(a.target)
    elif a.cmd == "export":
        cmd_export(a.target)
    elif a.cmd == "delete":
        cmd_delete(a.target)
    elif a.cmd == "size":
        cmd_size()


if __name__ == "__main__":
    main()