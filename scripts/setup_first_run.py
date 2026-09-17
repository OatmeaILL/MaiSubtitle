# -*- coding: utf-8 -*-
"""首次使用一键安装：装依赖 → 下必需模型 → 转换翻译模型 → 自检。

为什么单独一个脚本（而不是全写在 .bat 里）：
  · bat 只负责"找到/造出一个 Python 解释器"，真正的判断与步骤在这里 —— Python 比
    bat 好写、好测，还能 `--check` 空跑（不下载、不改盘）供用户和回归使用。
  · **只用标准库**：它必须在"依赖还没装的 venv"里跑得起来（第一步就是装依赖）。

典型用法：
    .venv\\Scripts\\python.exe scripts/setup_first_run.py            # 真装
    .venv\\Scripts\\python.exe scripts/setup_first_run.py --check    # 只体检 + 打印计划
    .venv\\Scripts\\python.exe scripts/setup_first_run.py --skip-models
（`安装_首次使用.bat` 会依次判断 uv / py / python，建好 .venv 后调本脚本。）

幂等：已装的依赖 pip 自己判、已下的模型判标志文件、转换过的判 model.bin；
重复运行不会重复下载。所有下载走**阿里云镜像**（含 torch 的 CPU 轮子）。
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
VENV_PY = VENV / "Scripts" / "python.exe"
REQ = ROOT / "requirements.txt"
PIP_MIRROR = "https://mirrors.aliyun.com/pypi/simple/"
PIP_HOST = "mirrors.aliyun.com"

# 启动必需的关键 import（与 requirements.txt 对应；缺任何一个都进不去主界面）
CORE_IMPORTS = [
    "PyQt6", "numpy", "av", "soundfile", "soxr", "faster_whisper", "ctranslate2",
    "onnxruntime", "kaldi_native_fbank", "pyaudiowpatch", "keyboard", "psutil",
    "requests", "zhconv", "rapidfuzz", "jinja2", "tokenizers",
    "torch", "transformers",          # 默认翻译引擎 hymt2（PyTorch）需要
]
# 默认档要准备的模型：识别 + 翻译 + VAD（就是设置里的推荐值）
MODEL_MARKS = [
    ("Whisper large-v3-turbo", "models/faster-whisper-large-v3-turbo/model.bin"),
    ("Hy-MT2-1.8B（翻译）", "models/Hy-MT2-1.8B/config.json"),
    ("FireRedVAD ONNX", "models/fireredvad-onnx/stream_vad.onnx"),
    ("Silero VAD（VAD 兜底）", "models/silero_vad.onnx"),
]
# 下载项（都走 ModelScope/镜像）：whisper 识别、Hy-MT2 翻译、FireRedVAD（下完要导出 ONNX）
DL_DEFAULT = ["whisper_turbo", "silero_vad", "firered", "hymt2"]
# FireRedVAD 的 ONNX 由原始权重导出（要 torch + fireredvad 包，两步都在本脚本里做）
FIRERED_ONNX = ROOT / "models" / "fireredvad-onnx" / "stream_vad.onnx"

CHUNK_GB = 7.0      # 默认档大致体积（依赖 ~3GB 含 torch + 模型 ~4GB），打印给用户看的预算


def say(msg: str = ""):
    print(msg, flush=True)


def run(cmd: list, timeout: float | None = None) -> int:
    """跑一个子进程并把输出直接透到控制台（保持顺序，便于用户跟着看）。"""
    say("  $ " + " ".join(str(c) for c in cmd))
    try:
        return subprocess.call([str(c) for c in cmd], cwd=str(ROOT), timeout=timeout)
    except subprocess.TimeoutExpired:
        say("  （超时中断）")
        return 124


def py_run(code: str, py: Path | None = None) -> int:
    return run([py or VENV_PY, "-c", code])


def missing_imports(py: Path | None = None) -> list:
    """在目标解释器里试 import；返回缺的模块名。"""
    code = ("import importlib.util as u;"
            "print(' '.join(m for m in %r if u.find_spec(m) is None))" % CORE_IMPORTS)
    r = subprocess.run([str(py or VENV_PY), "-c", code], capture_output=True,
                       text=True, cwd=str(ROOT))
    if r.returncode != 0:
        return list(CORE_IMPORTS)          # 解释器本身有问题 → 当作全缺
    return [m for m in (r.stdout or "").split() if m]


def torch_ok(py: Path | None = None) -> bool:
    return subprocess.run([str(py or VENV_PY), "-c", "import torch"],
                          capture_output=True, cwd=str(ROOT)).returncode == 0


def pip_install(args: list, py: Path | None = None) -> int:
    return run([py or VENV_PY, "-m", "pip", "install", *args,
                "-i", PIP_MIRROR, "--trusted-host", PIP_HOST,
                "--disable-pip-version-check"])


def models_state() -> tuple:
    """返回 (已有的模型名, 缺的模型名)。"""
    have, miss = [], []
    for name, rel in MODEL_MARKS:
        (have if (ROOT / rel).exists() else miss).append(name)
    return have, miss


def report(*, fix: bool, skip_deps: bool = False, skip_models: bool = False) -> bool:
    """体检并（fix=True 时）补齐；返回"是否已就绪"。"""
    say("=" * 62)
    say("MaiSubtitle 首次使用安装" + ("（--check：只看不动）" if not fix else ""))
    say("=" * 62)
    say(f"项目目录: {ROOT}")
    say(f"解释器  : {sys.executable}")
    if not VENV_PY.exists():
        say("[错误] 没找到 .venv\\Scripts\\python.exe —— 请先双击 安装_首次使用.bat")
        return False
    if Path(sys.executable).resolve() != VENV_PY.resolve():
        say(f"[注意] 当前不是用 .venv 的解释器在跑；装依赖会装错地方。"
            f"建议：\"{VENV_PY}\" scripts/setup_first_run.py")
        return False

    ok = True
    # ---- 1. 依赖 ----
    if skip_deps:
        say("\n[1/4] 依赖：跳过（--skip-deps）")
    else:
        miss = missing_imports()
        if miss:
            say(f"\n[1/4] 依赖：缺 {len(miss)} 个 → {' '.join(miss)}")
            if fix:
                say(f"      安装 requirements.txt（阿里云镜像，首次约 {CHUNK_GB / 2:.0f} 分钟）…")
                if pip_install(["-r", str(REQ)]) != 0:
                    say("      [错误] 装依赖失败：看上面的 pip 报错（网络/镜像问题居多）")
                    return False
                miss = missing_imports()
                if miss:
                    say(f"       装完仍缺：{' '.join(miss)}")
                    return False
                say("      [OK] 依赖齐了")
            else:
                ok = False
        else:
            say("\n[1/4] 依赖：[OK] 齐全")

    # ---- 2. 必需模型 ----
    have, mmiss = models_state()
    if skip_models:
        say("\n[2/4] 模型：跳过（--skip-models）")
    elif mmiss:
        say(f"\n[2/4] 模型：缺 {'、'.join(mmiss)} → 下载 {' '.join(DL_DEFAULT)}（约 4GB）")
        if fix:
            if run([VENV_PY, str(ROOT / "scripts" / "download_models.py"),
                    "--only", *DL_DEFAULT]) != 0:
                say("      [错误] 下载失败：见上面的报错（镜像回落也在 download_models 里）")
                return False
            say("      [OK] 模型下好了")
        else:
            ok = False
    else:
        say("\n[2/4] 模型：[OK] 齐全（" + "、".join(have) + "）")

    # ---- 3. FireRedVAD：原始权重 → ONNX（要 torch + fireredvad 包，CPU、几秒）----
    if skip_models:
        say("\n[3/4] FireRedVAD 导出：跳过（--skip-models）")
    elif FIRERED_ONNX.exists():
        say("\n[3/4] FireRedVAD 导出：[OK] 已就绪")
    else:
        say("\n[3/4] FireRedVAD 导出：把权重导成 ONNX（主程序用它做分句，不需要 torch）")
        if fix:
            if run([VENV_PY, str(ROOT / "scripts" / "model_manager.py"),
                    "export", "firered"]) != 0:
                say("      [错误] 导出失败：见上面的报错")
                return False
        else:
            ok = False

    # ---- 4. 自检 ----
    say("\n[4/4] 自检：")
    if fix or not ok:
        run([VENV_PY, str(ROOT / "scripts" / "model_manager.py"), "list"])
    if fix:
        say("\n其它可选模型（用不到就不用装）：")
        say("  · 后备翻译 Qwen2.5-CT2（快，0.22s/句）：scripts/download_models.py"
            " --only qwen_1_5b_hf，再 scripts/model_manager.py convert qwen_1_5b")
        say("  · 标点 CPU 模型（中文补标点免 GPU）：scripts/download_models.py --only punc_cpu")
        say("  · Qwen3-ASR 识别：scripts/download_models.py --only qwen3_asr_0_6b_onnx_int4")

    if not fix:
        say("\n--check 结束：以上就是要做的步骤（去掉 --check 即真执行）。")
        return False
    _, miss_now = models_state()         # 刚下/导出完 → 重新判一遍，别拿旧结果下结论
    if miss_now:
        say("\n[注意] 还缺：" + "、".join(miss_now))
        return False
    say("\n[OK] 就绪：双击 启动_MaiSubtitle.bat 即可开始")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="MaiSubtitle 首次使用安装/体检")
    ap.add_argument("--check", action="store_true", help="只体检 + 打印计划（不下载、不改盘）")
    ap.add_argument("--yes", action="store_true", help="不询问，直接开干")
    ap.add_argument("--skip-deps", action="store_true", help="跳过装依赖")
    ap.add_argument("--skip-models", action="store_true", help="跳过下模型/转换")
    a = ap.parse_args()

    if a.check:
        report(fix=False)
        return 0

    if not a.yes and sys.stdin is not None and sys.stdin.isatty():
        say(f"将要：装依赖 + 下载必需模型（约 {CHUNK_GB:.1f} GB）+ 转换翻译模型。")
        try:
            if input("继续？[Y/n] ").strip().lower() in ("n", "no"):
                say("已取消。")
                return 1
        except EOFError:
            pass

    t0 = time.perf_counter()
    ok = report(fix=True, skip_deps=a.skip_deps, skip_models=a.skip_models)
    say(f"\n用时 {time.perf_counter() - t0:.0f} 秒。")
    say("[OK] 全部就绪" if ok else "[注意] 还没就绪：按上面提示处理后重跑本脚本")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())