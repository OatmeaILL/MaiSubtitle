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
# CPU 版 torch 的国内轮子源（实测可列目录、含 cp312 win_amd64；只用于"转换翻译模型"
# 这一次性步骤，装完可以留着也可以卸）。官方 CPU 源作回落。
TORCH_LINKS = "https://mirrors.aliyun.com/pytorch-wheels/cpu/"
TORCH_FALLBACK = "https://download.pytorch.org/whl/cpu"

# 启动必需的关键 import（与 requirements.txt 对应；缺任何一个都进不去主界面）
CORE_IMPORTS = [
    "PyQt6", "numpy", "av", "soundfile", "soxr", "faster_whisper", "ctranslate2",
    "onnxruntime", "kaldi_native_fbank", "pyaudiowpatch", "keyboard", "psutil",
    "requests", "zhconv", "rapidfuzz", "jinja2", "tokenizers",
]
# 必需模型（对应默认/推荐配置：whisper 识别 + Silero VAD + Qwen2.5-CT2 翻译）
MODEL_MARKS = [
    ("Whisper large-v3-turbo", "models/faster-whisper-large-v3-turbo/model.bin"),
    ("Silero VAD", "models/silero_vad.onnx"),
    ("翻译 Qwen2.5-1.5B-CT2", "models/Qwen2.5-1.5B-Instruct-ct2/model.bin"),
]
# 转换用的 HF 权重（下完转好可以删；所以它是"过程件"不是"必需件"）
HF_DIR = ROOT / "models" / "Qwen2.5-1.5B-Instruct-hf"
# 可选增强（装了更准/更快，不装自动回落）：标点 CPU 模型 + FireRedVAD
DL_OPTIONAL = ["punc_cpu"]

CHUNK_GB = 3.0      # 必需模型的大致体积（打印给用户看的预算）


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
    """返回 (已有的必需模型名, 缺的必需模型名, HF 权重是否在)。"""
    have, miss = [], []
    for name, rel in MODEL_MARKS:
        (have if (ROOT / rel).exists() else miss).append(name)
    hf_ok = HF_DIR.exists() and any(HF_DIR.glob("*.safetensors"))
    return have, miss, hf_ok


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
    have, mmiss, hf_ok = models_state()
    # 翻译模型是"转换产物"：HF 权重只是转换的输入。**CT2 在就不需要 HF**
    #（否则会在"早就转好了"的机器上又要下 2.9GB —— 实测踩过）。
    need_ct2 = "翻译 Qwen2.5-1.5B-CT2" in mmiss
    dl = [m for m in mmiss if m != "翻译 Qwen2.5-1.5B-CT2"]
    if need_ct2 and not hf_ok:
        dl.append("qwen_1_5b_hf")
    if skip_models:
        say("\n[2/4] 模型：跳过（--skip-models）")
    elif dl:
        say(f"\n[2/4] 模型：缺 {'、'.join(mmiss)} → 下载 {' '.join(dl)}"
            f"（约 {CHUNK_GB:.1f} GB）")
        if fix:
            if run([VENV_PY, str(ROOT / "scripts" / "download_models.py"),
                    "--only", *dl]) != 0:
                say("      [错误] 下载失败：见上面的报错（HF 镜像回落也在 download_models 里）")
                return False
            say("      [OK] 模型下好了")
        else:
            ok = False
    else:
        say("\n[2/4] 模型：[OK] 齐全（" + "、".join(have) + "）"
            + ("（翻译模型待转换）" if need_ct2 else ""))

    # ---- 3. 翻译模型转换（需要 torch，仅这一次）----
    if skip_models:
        say("\n[3/4] 翻译模型转换：跳过（--skip-models）")
    elif not need_ct2:
        say("\n[3/4] 翻译模型转换：[OK] 已就绪"
            "（转换用的 HF 权重可以删：models/Qwen2.5-1.5B-Instruct-hf）")
    else:
        say("\n[3/4] 翻译模型转换：需要 torch（CPU 版即可，只用于转换这一次）")
        if not torch_ok():
            if fix:
                say("      装 CPU 版 torch（阿里云轮子源，约 200 MB）…")
                if pip_install(["--find-links", TORCH_LINKS, "torch"]) != 0:
                    say("      阿里云源失败 → 试官方 CPU 源 …")
                    if run([VENV_PY, "-m", "pip", "install", "torch",
                            "--index-url", TORCH_FALLBACK,
                            "--disable-pip-version-check"]) != 0:
                        say("      torch 装不上（不影响其它步骤；"
                            "装好后重跑本脚本即可继续转换）")
                        return False
                if not torch_ok():
                    say("       torch 装了但 import 不进来")
                    return False
            else:
                say("      （--check：会先装 CPU 版 torch，再转换）")
                ok = False
        if fix:
            if run([VENV_PY, str(ROOT / "scripts" / "model_manager.py"),
                    "convert", "qwen_1_5b"]) != 0:
                say("      [错误] 转换失败：见上面的报错")
                return False
            say("      [OK] 转换完成（models/Qwen2.5-1.5B-Instruct-ct2）")
        else:
            ok = False

    # ---- 4. 自检 ----
    say("\n[4/4] 自检：")
    if fix or not ok:
        run([VENV_PY, str(ROOT / "scripts" / "model_manager.py"), "list"])
    if fix:
        say("\n可选增强（不装也能跑，会自动回落）："
            f"标点 CPU 模型 → .venv\\Scripts\\python.exe scripts/download_models.py"
            f" --only {' '.join(DL_OPTIONAL)}")
        say("  FireRedVAD（推荐 VAD）需要自己导出：scripts/export_fireredvad_onnx.py"
            "（要 torch + 上游权重；缺失自动回落 Silero）")

    if not fix:
        say("\n--check 结束：以上就是要做的步骤（去掉 --check 即真执行）。")
        return False
    _, miss_now, _ = models_state()      # 转换刚做完 → 重新判一遍，别拿旧结果下结论
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