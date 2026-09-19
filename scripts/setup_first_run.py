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
重复运行不会重复下载。

**PyPI 源先测速再选**：候选四家（清华 / 中科大 / 腾讯云 / 阿里云）各读几 MB 量一次速度，
用最快的那家装依赖 —— 实测同一 torch wheel 差 100 倍（0.22 vs 24 MB/s，2.5GB 差 3 小时），
所以不能写死一家。要指定就 `--mirror 清华`（或任何 index URL、或环境变量 MAISUB_PYPI_MIRROR）。

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
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
VENV_PY = VENV / "Scripts" / "python.exe"
REQ = ROOT / "requirements.txt"
LOGS_DIR = ROOT / "logs"          # 放生成物（如"去掉 torch 的清单"），不污染项目根目录

# ---- PyPI 源：**先测速再选**（2026-09-18 实测，同一 numpy wheel / torch 前 20MB）----
#   清华 24.2 / 中科大 18.9 / 腾讯云 11.0 / 阿里云 **0.22** MB/s —— 差 100 倍，
#   而 torch 一只就 2.5GB（阿里云要 3 小时，清华 1.7 分钟）。**别写死一家。**
PYPI_MIRRORS = [
    ("清华", "https://pypi.tuna.tsinghua.edu.cn/simple/", "pypi.tuna.tsinghua.edu.cn"),
    ("中科大", "https://mirrors.ustc.edu.cn/pypi/simple/", "mirrors.ustc.edu.cn"),
    ("腾讯云", "https://mirrors.cloud.tencent.com/pypi/simple/", "mirrors.cloud.tencent.com"),
    ("阿里云", "https://mirrors.aliyun.com/pypi/simple/", "mirrors.aliyun.com"),
]
PROBE_PKG = "numpy"                # 测速用的包：各镜像都有，wheel 十几 MB，够读几秒
PROBE_BYTES = 4 * 1024 * 1024      # 每个镜像最多读这么多
PROBE_SECONDS = 5.0                # 每个镜像最多花这么多秒

FORCED_MIRROR = os.environ.get("MAISUB_PYPI_MIRROR", "")   # 非空 = 跳过测速直接用
_MIRROR = None                     # 测速结果缓存（名称, index_url, host）

# 启动必需的关键 import（对应 requirements.txt 里"能一把装上"的那些）
# fireredvad / onnx / onnxscript 是"导出 FireRedVAD ONNX"那一步要的，一起验：
# 少了它们的表现是**导出崩**，不在启动路径上，最容易被漏掉（2026-09-18 实锤）。
CORE_IMPORTS = [
    "PyQt6", "numpy", "av", "soundfile", "soxr", "ctranslate2",
    "onnxruntime", "kaldi_native_fbank", "pyaudiowpatch", "keyboard", "psutil",
    "requests", "zhconv", "rapidfuzz", "jinja2", "tokenizers",
    "torch", "transformers",          # 默认翻译引擎 hymt2（PyTorch）需要
    "accelerate",                     # hymt2 用 device_map 加载 → transformers 硬要求它
    "fireredvad", "onnx", "onnxscript",   # 导出 FireRedVAD ONNX 需要
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
    ("标点 CPU 模型", "models/punc-ct-transformer-zh-en-onnx/model_quant.onnx"),
]
# 下载项（都走 ModelScope/镜像）：whisper 识别、Hy-MT2 翻译、FireRedVAD（下完要导出 ONNX）、
# 标点 CPU 小模型（每行定稿补标点：中文走它，1~4ms、不占 GPU；2026-09-18 进默认档）
DL_DEFAULT = ["whisper_turbo", "silero_vad", "firered", "hymt2", "punc_cpu"]
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


def torch_has_cuda(py: Path | None = None) -> bool:
    """torch 能不能用 CUDA。默认翻译引擎 hymt2 走 PyTorch —— CPU 上慢好几倍，
    所以装完必须体检一下（新版 PyPI 的 Windows 轮子是 CPU 版，很容易装上 2.14.0+cpu）。"""
    return quiet_ok([py or VENV_PY, "-c",
                     "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"])


# ---- CUDA 版 torch：能连官方源就装 ----
# 实测（2026-09-18）：国内 8 个镜像（阿里云/清华/中科大/上交/南大/北外/CERNET/腾讯）
# **都没有 Windows 的 CUDA torch 轮子**，官方索引才是唯一来源；而官方源国内直连可用
# （8.9 MB/s，2.4GB 约 5 分钟）。所以这里不"选源"，只做"探活 + 失败不阻塞"。
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu124"
TORCH_CUDA_SPEC = "torch==2.6.0+cu124"      # 与开发机同款，CUDA 12.4（驱动 ≥525 即可）


def has_nvidia_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None


def cuda_index_reachable(seconds: int = 8, tries: int = 3) -> bool:
    """官方 CUDA 索引通不通。不通就跳过 CUDA torch —— 绝不能因此让安装失败。

    官方源国内直连**时好时坏**（Fastly CDN）：2026-09-18 晚用户实装时 8s 探活失败 →
    静默跳过 → 装了 CPU 版 torch → hymt2 降级 CPU（"老问题又出现了"）。所以加重试：
    3 次 × 8s，间隔 2s —— 抖动一下也能等到。还不通就是真不通（此刻下载也必失败），
    放心走 CPU 版，另给 `安装_首次使用.bat --cuda-torch` 随时补装。
    """
    for i in range(tries):
        try:
            req = urllib.request.Request(TORCH_CUDA_INDEX + "/torch/",
                                         headers={"User-Agent": "MaiSubtitle-setup"})
            with urllib.request.urlopen(req, timeout=seconds) as r:
                if r.status == 200:
                    if i:
                        print(f"      官方 CUDA 源第 {i + 1} 次探活成功")
                    return True
        except Exception:
            pass
        if i < tries - 1:
            time.sleep(2)
    return False


def install_cuda_torch() -> bool:
    """补装/换装 CUDA 版 torch（`安装_首次使用.bat --cuda-torch`）。

    场景：首次安装时官方源恰好不可达 → 装了 CPU 版 torch（hymt2 降级 CPU）。
    官方源国内时好时坏，所以给一条**随时能重跑**的命令：网络通了跑一下，
    pip 会卸掉 CPU 版再装 CUDA 版（--no-deps 不动 sympy，避开卸载卡死的坑）。
    """
    if not has_nvidia_gpu():
        say("[注意] 本机没有 NVIDIA 显卡，不需要 CUDA 版 torch；hymt2 会跑 CPU，"
            "想更快就把翻译引擎换成 qwen（设置 → 翻译引擎）")
        return False
    if not cuda_index_reachable():
        say("[错误] 官方 CUDA 源（download.pytorch.org）此刻连不上（重试 3 次均失败）。"
            "过一会儿网络好了再跑一次本命令即可。")
        return False
    if torch_has_cuda():
        say("[OK] torch 已带 CUDA，无需补装")
        return True
    say(f"开始装 {TORCH_CUDA_SPEC}（约 2.4GB，实测 3~5 分钟）…")
    rc = pip_install([TORCH_CUDA_SPEC], index_url=TORCH_CUDA_INDEX, no_deps=True)
    if rc != 0 or not torch_has_cuda():
        say("[错误] CUDA 版 torch 没装上（看上面的报错）；"
            "网络恢复后重跑：安装_首次使用.bat --cuda-torch")
        return False
    say("[OK] torch 带 CUDA（hymt2 翻译走 GPU）")
    return True


def reqs_without_torch() -> Path:
    """requirements.txt 去掉裸 `torch` 那一行（写进 logs/，不改原文件）。

    为什么要这份：决定装 CUDA 版 torch 时，若先按原清单装了 CPU 版 torch，稍后换 CUDA 版
    就要 pip 去**卸载**已有的 torch/sympy —— 本机 pip 在卸载步骤会卡住不动
    （2026-09-18 实测：日志停在 `Uninstalling sympy-1.14.0:`，一停几个小时）。
    一开始就不装 CPU 版，既省 500MB 也绕过整个卸载路径。
    """
    out = LOGS_DIR / "requirements-notorch.txt"
    keep = []
    for line in REQ.read_text(encoding="utf-8").splitlines():
        if line.strip().lower() in ("torch",):     # 裸 torch 行才去掉
            continue
        keep.append(line)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(keep) + "\n", encoding="utf-8")
    return out


# ---------------- PyPI 源：先测速，再选最快的 ----------------

def speedtest_mirror(index_url: str) -> tuple:
    """量一个源的速度：从它的索引里挑个 wheel，读几 MB 计时。返回 (MB/s, 说明)。"""
    idx = index_url + PROBE_PKG + "/"
    ua = {"User-Agent": "MaiSubtitle-setup"}
    try:
        with urllib.request.urlopen(urllib.request.Request(idx, headers=ua), timeout=8) as r:
            html = r.read().decode("utf-8", "replace")
    except Exception as e:
        return 0.0, f"索引不通（{type(e).__name__}）"
    links = [urllib.parse.urljoin(idx, l) for l in re.findall(r'href="([^"#]+)', html)]
    url = ""
    for pref in ("cp312-cp312-win_amd64.whl", "cp312-abi3-win_amd64.whl", "py3-none-any.whl"):
        for link in links:
            if pref in link:
                url = link
                break
        if url:
            break
    if not url:
        return 0.0, "索引里没有可用 wheel"
    got = 0
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=ua), timeout=8) as r:
            while got < PROBE_BYTES:
                chunk = r.read(262144)
                if not chunk:
                    break
                got += len(chunk)
                if time.perf_counter() - t0 > PROBE_SECONDS:
                    break
    except Exception:
        if got == 0:                      # 一个字节都没读到才算失败（读到一半断按已读的算）
            return 0.0, "下载不通"
    dt = time.perf_counter() - t0
    if dt <= 0 or got == 0:
        return 0.0, "没读到数据"
    speed = got / 1048576 / dt
    return speed, f"{got / 1048576:4.1f} MB / {dt:4.1f}s = {speed:5.1f} MB/s"


def pick_mirror() -> tuple:
    """四个候选各测一次速，取最快的。返回 (名称, index_url, host)。"""
    say("      先测速挑 PyPI 源（每个源读几 MB，别写死一家）：")
    best = (0.0, "", "", "")
    for name, url, host in PYPI_MIRRORS:
        speed, note = speedtest_mirror(url)
        say(f"        {name:6} {note}")
        if speed > best[0]:
            best = (speed, name, url, host)
    if best[0] <= 0:
        name, url, host = PYPI_MIRRORS[0]
        say(f"        都没测出速度 → 回落到 {name}")
        return name, url, host
    say(f"        → 用 {best[1]}（{best[0]:.1f} MB/s）")
    return best[1], best[2], best[3]


def mirror() -> tuple:
    """当前要用的 PyPI 源（第一次调用时测速，之后缓存）。"""
    global _MIRROR
    if _MIRROR is None:
        if FORCED_MIRROR:
            url = FORCED_MIRROR
            if "//" not in url:                       # 允许只写源名
                for name, u, host in PYPI_MIRRORS:
                    if name == FORCED_MIRROR:
                        url, host = u, host
                        break
                else:
                    url = "https://" + FORCED_MIRROR + "/simple/"
            host = url.split("//")[-1].split("/")[0]
            say(f"      PyPI 源：{url}（手动指定，跳过测速）")
            _MIRROR = ("指定", url, host)
        else:
            _MIRROR = pick_mirror()
    return _MIRROR


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


def _run_pip(py: Path, args: list, no_deps: bool, index_url: str | None = None) -> int:
    _, mirror_url, host = mirror()
    idx = index_url or mirror_url
    cmd = [py, "-m", "pip", "install", *args, "-i", idx]
    if index_url:
        # 指定了特殊索引（如 torch 的 CUDA 索引）：torch 本体只可能来自它，
        # 但它缺的依赖可以走我们测速选出来的快源
        cmd += ["--extra-index-url", mirror_url]
    else:
        cmd += ["--trusted-host", host]
    cmd.append("--disable-pip-version-check")
    if no_deps:
        cmd.append("--no-deps")
    return run(cmd)


def pip_install(args: list, py: Path | None = None, no_deps: bool = False,
                index_url: str | None = None) -> int:
    """装包：venv 自带 pip → uv pip → ensurepip 补 pip 再装。都失败返回非 0。"""
    py = py or VENV_PY
    _, mirror_url, _ = mirror()
    kind = installer_kind(py)
    if kind == "pip":
        return _run_pip(py, args, no_deps, index_url)
    if kind == "uv":
        say("      （这个 .venv 里没有 pip —— uv 建的 venv 默认不带；改用 uv pip 安装）")
        cmd = ["uv", "pip", "install", "--python", py, *args,
               "--index-url", index_url or mirror_url]
        if index_url:
            cmd += ["--extra-index-url", mirror_url]
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
    return _run_pip(py, args, no_deps, index_url)


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
    if not fix:
        say("PyPI 源 : 真装时先测速，用最快的一家（清华 / 中科大 / 腾讯云 / 阿里云；"
            "可用 --mirror 指定）")

    ok = True
    # ---- 1. 依赖（requirements.txt，能一把装上的那些）----
    if skip_deps:
        say("\n[1/5] 依赖：跳过（--skip-deps）")
    else:
        miss = missing_imports(CORE_IMPORTS)
        if miss:
            say(f"\n[1/5] 依赖：缺 {len(miss)} 个 → {' '.join(miss)}")
            if fix:
                name, _, _ = mirror()          # 先测速选源（打印各源速度）
                # 有 N 卡 + 官方源可达 → **一开始就别装 CPU 版 torch**：
                # 先装 CPU 版再换 CUDA 版，pip 要"卸载"已装的 torch/sympy，而本机 pip 在卸载
                # 那一步会卡死（2026-09-18 实测：日志停在 Uninstalling sympy，停了几小时）。
                cuda_first = has_nvidia_gpu() and cuda_index_reachable()
                if cuda_first:
                    say(f"      检测到 N 卡 + 官方源可达 → 跳过 CPU 版 torch，"
                        f"直接装 CUDA 版（{TORCH_CUDA_SPEC}，约 2.4GB，实测约 3 分钟）")
                    rc = pip_install(["-r", str(reqs_without_torch())])
                else:
                    say(f"      用「{name}」装 requirements.txt（缺 {len(miss)} 个）…")
                    rc = pip_install(["-r", str(REQ)])
                if rc != 0:
                    say("      [错误] 装依赖失败：看上面的报错（网络/镜像问题居多）；"
                        "可换一家源重试：安装_首次使用.bat --mirror 清华")
                    return False
                if cuda_first:
                    # --no-deps：torch 的依赖（filelock/networkx/jinja2/fsspec/typing-extensions）
                    # 上一步都装好了；带上依赖反而会让 pip 去动 sympy（torch 2.6 钉 1.13.1，
                    # 本机是 1.14.0）—— 卸载那步正是会卡死的地方。开发机就是
                    # sympy 1.14 + torch 2.6.0+cu124 这么跑着的，没问题。
                    pip_install([TORCH_CUDA_SPEC], index_url=TORCH_CUDA_INDEX, no_deps=True)
                miss = missing_imports(CORE_IMPORTS)
                if miss:
                    say(f"       装完仍缺：{' '.join(miss)}")
                    return False
                say("      [OK] 依赖齐了")
            else:
                ok = False
        else:
            say("\n[1/5] 依赖：[OK] 齐全")

    # ---- 1b. 显卡体检（要不要装 CUDA torch 已在第 1 步决定并执行过）----
    if not missing_imports(["torch"]):
        if torch_has_cuda():
            say("      torch 带 CUDA（hymt2 翻译走 GPU）")
        else:
            say("      [注意] torch 不带 CUDA → hymt2 跑 CPU（能用，实测每句 1~2 秒）")
            if not has_nvidia_gpu():
                say("             本机没有 NVIDIA 显卡（nvidia-smi），这样是正常的；"
                    "想更快可换引擎：设置 → 翻译引擎 → qwen（CT2，约 0.22s/句）")
            else:
                say("             原因：官方源当时不可达（国内对它时好时坏），或 CUDA 版没装上。"
                    "三条替代：")
                say("               ① 随时补装：安装_首次使用.bat --cuda-torch"
                    "（官方源通了再跑，2.4GB，装完重启程序生效）")
                say("               ② 设置 → 外部 torch 目录：指一个已有的 CUDA torch，"
                    "零下载（机器上别的 AI 环境里有就行）")
                say("               ③ 设置 → 翻译引擎 → qwen（CT2，约 0.22s/句）")

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
            f"（约 5.7GB：whisper 1.5 + Hy-MT2 3.9 + 标点 0.3）")
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
            run([VENV_PY, str(ROOT / "scripts" / "model_manager.py"), "export", "firered"])
            if not FIRERED_ONNX.exists():
                # **验产物，别信退出码**：导出脚本崩了也可能一路 0（2026-09-18 实锤）
                say("      [错误] 没生成 stream_vad.onnx（原因见上面的报错）："
                    "最常见是缺 onnx / onnxscript")
                say("      [注意] 缺它主程序会自动回落到 Silero：能用，但句子切得更碎；"
                    "补好依赖后重跑本脚本即可")
            else:
                say("      [OK] 已导出 models/fireredvad-onnx/stream_vad.onnx")
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
    ap.add_argument("--mirror", default="", metavar="源",
                    help="强制 PyPI 源（清华 / 中科大 / 腾讯云 / 阿里云，或完整 index URL）；"
                         "默认自动测速选最快")
    ap.add_argument("--cuda-torch", action="store_true",
                    help="只补装 CUDA 版 torch（给 torch 不带 CUDA 的已装环境；"
                         "官方源国内时好时坏，网络通了跑一次即可）")
    a = ap.parse_args()

    global FORCED_MIRROR
    if a.mirror:
        FORCED_MIRROR = a.mirror

    if a.cuda_torch:
        return 0 if install_cuda_torch() else 1

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
