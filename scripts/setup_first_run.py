# -*- coding: utf-8 -*-
"""首次使用一键安装：装依赖 → 补两个必须单独装的包 → 下必需模型 → 导出 VAD → 自检。

为什么单独一个脚本（而不是全写在 .bat 里）：
  · bat 只负责"找到/造出一个 Python 解释器"，真正的判断与步骤在这里 —— Python 比
    bat 好写、好测，还能 `--check` 空跑（不下载、不改盘）供用户和回归使用。
  · **只用标准库**：它必须在"依赖还没装的 venv"里跑得起来（第一步就是装依赖）。
    （可用时会调 uv 命令行 —— 那是外部程序，不是 Python 依赖。）

典型用法：
    .venv\\Scripts\\python.exe scripts/setup_first_run.py            # 真装
    .venv\\Scripts\\python.exe scripts/setup_first_run.py --check    # 只体检 + 打印计划
    .venv\\Scripts\\python.exe scripts/setup_first_run.py --skip-models
（`安装_首次使用.bat` 会依次判断 uv / py / python，建好 .venv 后调本脚本。）

幂等：已装的依赖由装包器自己判、已下的模型判标志文件、导出过的判 onnx 文件；
重复运行不会重复下载。所有下载走**阿里云镜像**。

两个实测过的安装坑（新用户第一次装必撞，已在这里兜住）：
  ① **uv 建的 .venv 里没有 pip**：uv 用自己的 `uv pip` 管包，venv 的 site-packages 里
     只有它的 _virtualenv 垫片 → `python -m pip` 直接 "No module named pip"。
     兜底顺序见 installer_kind()：venv 自带 pip → uv pip → ensurepip 补一份 pip。
  ② **faster-whisper / funasr-onnx 不能跟着 requirements.txt 一起装**（第二步 NODEPS_*）：
     前者声明依赖 CPU 版 onnxruntime，会把我们钉的 onnxruntime-gpu 1.23.2 覆盖掉
     （同名包 onnxruntime，后装者覆盖，DLL/.pyd 还可能错配）；后者声明 numpy<=1.26.4，
     与本项目 numpy 2.x 解析无解。所以一律 `--no-deps` 单独装，理由也写在
     requirements.txt 末尾。
"""
import argparse
import shutil
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

# 启动必需的关键 import（对应 requirements.txt 里"能一把装上"的那些）
CORE_IMPORTS = [
    "PyQt6", "numpy", "av", "soundfile", "soxr", "ctranslate2",
    "onnxruntime", "kaldi_native_fbank", "pyaudiowpatch", "keyboard", "psutil",
    "requests", "zhconv", "rapidfuzz", "jinja2", "tokenizers",
    "torch", "transformers",          # 默认翻译引擎 hymt2（PyTorch）需要
]
# 第二步单独装的（--no-deps）：faster-whisper 是识别引擎（缺了没得用），
# funasr-onnx + jieba 只影响"中文补标点走 CPU"与 FSMN VAD（缺了自动回落，只提示）
NODEPS_IMPORTS = ["faster_whisper", "funasr_onnx"]
NODEPS_REQUIRED = ["faster-whisper==1.2.1"]
NODEPS_OPTIONAL = ["funasr-onnx==0.4.3", "jieba"]
# 默认档要准备的模型：识别 + 翻译 + VAD（就是设置里的推荐值）
# ⚠ 判据要用**真权重文件**，别用 config.json 之类的小文件 —— 那只能证明"目录建出来了"，
#   下了一半（3.9GB 的 safetensors 没到）也会被判成"已就绪"。
MODEL_MARKS = [
    ("Whisper large-v3-turbo", "models/faster-whisper-large-v3-turbo/model.bin"),
    ("Hy-MT2-1.8B（翻译）", "models/Hy-MT2-1.8B/model.safetensors"),
    ("FireRedVAD ONNX", "models/fireredvad-onnx/stream_vad.onnx"),
    ("Silero VAD（VAD 兜底）", "models/silero_vad.onnx"),
]
# 下载项（都走 ModelScope/镜像）：whisper 识别、Hy-MT2 翻译、FireRedVAD（下完要导出 ONNX）
DL_DEFAULT = ["whisper_turbo", "silero_vad", "firered", "hymt2"]
# FireRedVAD 的 ONNX 由原始权重导出（要 torch + fireredvad 包，两步都在本脚本里做）
FIRERED_ONNX = ROOT / "models" / "fireredvad-onnx" / "stream_vad.onnx"

CHUNK_GB = 8.0      # 默认档大致体积（依赖 ~3GB 含 torch + 必需模型 ~5.4GB；2026-09-17 实测校准）

# 装包器的人话说明（新用户看不懂 "No module named pip" 是怎么回事）
KIND_TEXT = {
    "pip": "venv 自带 pip",
    "uv": "uv pip（这个 .venv 里没有 pip —— uv 建的 venv 默认不带）",
    "ensurepip": "先 ensurepip 补一份 pip（这个 .venv 里没有 pip）",
    "": "没有可用的装包器（venv 无 pip、也没有 uv）",
}


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


def quiet_ok(cmd: list) -> bool:
    """静默跑一条命令，只看成功与否（用于探测，不刷屏）。"""
    try:
        r = subprocess.run([str(c) for c in cmd], capture_output=True, cwd=str(ROOT))
    except OSError:
        return False
    return r.returncode == 0


def py_run(code: str, py: Path | None = None) -> int:
    return run([py or VENV_PY, "-c", code])


def missing_imports(names: list, py: Path | None = None) -> list:
    """在目标解释器里试 import；返回缺的模块名。"""
    code = ("import importlib.util as u;"
            "print(' '.join(m for m in %r if u.find_spec(m) is None))" % list(names))
    r = subprocess.run([str(py or VENV_PY), "-c", code], capture_output=True,
                       text=True, cwd=str(ROOT))
    if r.returncode != 0:
        return list(names)                 # 解释器本身有问题 → 当作全缺
    return [m for m in (r.stdout or "").split() if m]


def torch_ok(py: Path | None = None) -> bool:
    return subprocess.run([str(py or VENV_PY), "-c", "import torch"],
                          capture_output=True, cwd=str(ROOT)).returncode == 0


# ---------------- 装包（三级兜底：pip → uv pip → ensurepip）----------------

def pip_ready(py: Path) -> bool:
    return quiet_ok([py, "-m", "pip", "--version"])


def uv_ready() -> bool:
    return shutil.which("uv") is not None


def installer_kind(py: Path | None = None) -> str:
    """往这个 .venv 里装包会走哪条路：'pip' / 'uv' / 'ensurepip' / ''（都不行）。

    uv 建的 venv **不带 pip**，所以不能想当然用 `-m pip`（新用户第一个卡点）。
    """
    py = py or VENV_PY
    if pip_ready(py):
        return "pip"
    if uv_ready():
        return "uv"
    if quiet_ok([py, "-m", "ensurepip", "--version"]):
        return "ensurepip"
    return ""


def _run_pip(py: Path, args: list, no_deps: bool) -> int:
    cmd = [py, "-m", "pip", "install", *args,
           "-i", PIP_MIRROR, "--trusted-host", PIP_HOST,
           "--disable-pip-version-check"]
    if no_deps:
        cmd.append("--no-deps")
    return run(cmd)


def pip_install(args: list, py: Path | None = None, no_deps: bool = False) -> int:
    """装包：venv 自带 pip → uv pip → ensurepip 补 pip 再装。都失败返回非 0。"""
    py = py or VENV_PY
    kind = installer_kind(py)
    if kind == "pip":
        return _run_pip(py, args, no_deps)
    if kind == "uv":
        say("      （这个 .venv 里没有 pip —— uv 建的 venv 默认不带；改用 uv pip 安装）")
        cmd = ["uv", "pip", "install", "--python", py, *args, "--index-url", PIP_MIRROR]
        if no_deps:
            cmd.append("--no-deps")
        if run(cmd) == 0:
            return 0
        say("      uv pip 没装成 → 改用 venv 里补一份 pip 重试")
    say("      （往 .venv 里补一份 pip：python -m ensurepip）")
    if not quiet_ok([py, "-m", "ensurepip", "--upgrade"]) or not pip_ready(py):
        say("      [错误] 这个 .venv 既没有 pip、ensurepip 也补不上；"
            "把 .venv 目录整个删掉再重跑 安装_首次使用.bat")
        return 1
    return _run_pip(py, args, no_deps)


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
    kind = installer_kind()
    say(f"装包器  : {KIND_TEXT.get(kind, kind)}")

    ok = True
    # ---- 1. 依赖（requirements.txt，能一把装上的那些）----
    if skip_deps:
        say("\n[1/5] 依赖：跳过（--skip-deps）")
    else:
        miss = missing_imports(CORE_IMPORTS)
        if miss:
            say(f"\n[1/5] 依赖：缺 {len(miss)} 个 → {' '.join(miss)}")
            if fix:
                say(f"      安装 requirements.txt（阿里云镜像，首次约 {CHUNK_GB / 2:.0f} 分钟）…")
                if pip_install(["-r", str(REQ)]) != 0:
                    say("      [错误] 装依赖失败：看上面的报错（网络/镜像问题居多）；"
                        "网络不通可改用其他 pip 源重试")
                    return False
                miss = missing_imports(CORE_IMPORTS)
                if miss:
                    say(f"       装完仍缺：{' '.join(miss)}")
                    return False
                say("      [OK] 依赖齐了")
            else:
                ok = False
        else:
            say("\n[1/5] 依赖：[OK] 齐全")

    # ---- 2. 两个必须 --no-deps 单独装的包（见文件头 ②）----
    if skip_deps:
        say("\n[2/5] faster-whisper / funasr-onnx：跳过（--skip-deps）")
    else:
        miss2 = missing_imports(NODEPS_IMPORTS)
        if miss2:
            say(f"\n[2/5] 单独装（--no-deps）：缺 {' '.join(miss2)}")
            say("      为什么不能跟上面一起装：faster-whisper 会拉 CPU 版 onnxruntime 覆盖"
                "onnxruntime-gpu；funasr-onnx 钉 numpy<=1.26.4 与本项目无解")
            if fix:
                rc = pip_install(NODEPS_REQUIRED, no_deps=True)
                if rc != 0 or missing_imports(["faster_whisper"]):
                    say("      [错误] faster-whisper 没装上：识别引擎靠它，缺了没法用")
                    return False
                say("      [OK] faster-whisper（识别引擎）")
                if pip_install(NODEPS_OPTIONAL, no_deps=True) != 0:
                    say("      [注意] funasr-onnx 没装上（可选）：中文补标点会改用 Qwen，"
                        "占 GPU 且慢一些，不影响其它功能")
                else:
                    say("      [OK] funasr-onnx（中文补标点走 CPU）+ jieba")
            else:
                ok = False
        else:
            say("\n[2/5] faster-whisper / funasr-onnx：[OK] 已就绪")

    # ---- 3. 必需模型 ----
    have, mmiss = models_state()
    if skip_models:
        say("\n[3/5] 模型：跳过（--skip-models）")
    elif mmiss:
        say(f"\n[3/5] 模型：缺 {'、'.join(mmiss)} → 下载 {' '.join(DL_DEFAULT)}"
            f"（约 5.4GB：whisper 1.5 + Hy-MT2 3.9）")
        if fix:
            if run([VENV_PY, str(ROOT / "scripts" / "download_models.py"),
                    "--only", *DL_DEFAULT]) != 0:
                say("      [错误] 下载失败：见上面的报错（镜像回落也在 download_models 里）")
                return False
            say("      [OK] 模型下好了")
        else:
            ok = False
    else:
        say("\n[3/5] 模型：[OK] 齐全（" + "、".join(have) + "）")

    # ---- 4. FireRedVAD：原始权重 → ONNX（要 torch + fireredvad 包，CPU、几秒）----
    if skip_models:
        say("\n[4/5] FireRedVAD 导出：跳过（--skip-models）")
    elif FIRERED_ONNX.exists():
        say("\n[4/5] FireRedVAD 导出：[OK] 已就绪")
    else:
        say("\n[4/5] FireRedVAD 导出：把权重导成 ONNX（主程序用它做分句，不需要 torch）")
        if fix:
            if run([VENV_PY, str(ROOT / "scripts" / "model_manager.py"),
                    "export", "firered"]) != 0:
                say("      [错误] 导出失败：见上面的报错")
                return False
        else:
            ok = False

    # ---- 5. 自检 ----
    say("\n[5/5] 自检：")
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
