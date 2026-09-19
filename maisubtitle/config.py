"""路径与用户配置（配置文件 config.json，热更新由调用方决定）。"""
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# PyInstaller 冻结后资源在 exe 旁边（⚠ sys.frozen 只有冻结环境才有，普通解释器
# 连属性都不存在 —— 必须 getattr 带默认值，直写 sys.frozen 会 AttributeError）
if getattr(sys, "frozen", False):
    PROJECT_ROOT = Path(sys.executable).parent
MODELS_DIR = Path(os.environ.get("MAISUB_MODELS_DIR", PROJECT_ROOT / "models"))
LOGS_DIR = PROJECT_ROOT / "logs"
TESTDATA_DIR = PROJECT_ROOT / "testdata"
CONFIG_PATH = PROJECT_ROOT / "config.json"

for _d in (MODELS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def config_json_field(key: str, default=""):
    """直接读 config.json 的某个字段。

    用在"AppConfig 还没定义"或"必须在 import torch 之前就读到"的场景（本模块导入时）。
    读不到/文件坏了就返回 default —— **绝不因为配置问题让程序起不来**。
    """
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return data.get(key, default)
    except Exception:
        return default


def external_torch_site():
    """用户指定的外部 torch 目录 → 真正的 site-packages；没配/找不到返回 None。

    config.json 的 `torch_external_dir` 可以填 site-packages 目录，也可以填 venv 根目录
    （自动补 `Lib/site-packages`）。
    """
    raw = str(config_json_field("torch_external_dir") or "").strip()
    if not raw:
        return None
    p = Path(os.path.expandvars(os.path.expanduser(raw)))
    for c in (p, p / "Lib" / "site-packages"):
        if (c / "torch" / "__init__.py").is_file():
            return c
    return None


def _shadow_torch_metadata(site: Path) -> None:
    """让 importlib.metadata 报出**外部那份** torch 的版本（只改"版本号怎么报"）。

    为什么必须做：transformers 判定 torch 能力用的是 **importlib.metadata 的版本号**
    （`is_torch_flex_attn_available()` = `get_torch_version() >= 2.5`；
    `_TORCH_FLEX_USE_AUX` = `is_torch_greater_or_equal("2.9.0")`），**不是** `torch.__version__`。
    本 venv 里装着 CPU 版 torch（例如 2.14）时：
      · 元数据报 2.14 → transformers 以为在用"新版 torch"→ 去 import 只有 2.9+ 才有的 AuxRequest；
      · 而真正加载的是外部那份 2.6 → `ImportError: cannot import name 'AuxRequest'`
    （2026-09-18 实测：开发机能跑、本机挂了，差别就在"本地有没有另一份 torch 的元数据"。）

    做法：造一个**只含 dist-info** 的影子目录并插到 sys.path 最前 ——
    里面没有任何可导入的包，不会覆盖真正的模块，只影响版本号。
    """
    ver = ""
    try:
        for meta in site.glob("torch-*.dist-info/METADATA"):
            for line in meta.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("Version:"):
                    ver = line.split(":", 1)[1].strip()
                    break
            if ver:
                break
    except Exception:
        return
    if not ver:
        return
    try:
        d = PROJECT_ROOT / ".external_torch_meta" / ("torch-" + ver + ".dist-info")
        d.mkdir(parents=True, exist_ok=True)
        (d / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: torch\nVersion: " + ver + "\n",
            encoding="utf-8")
        root = str(d.parent)
        if root in sys.path:
            sys.path.remove(root)
        sys.path.insert(0, root)      # 最前：元数据以这份为准
    except Exception:
        pass


def mount_external_torch() -> str:
    """把用户指定的外部 CUDA torch 装成 sys.modules["torch"]，**不动 sys.path**。

    为什么这么绕：本项目 venv 里已经装了 CPU 版 torch（requirements 里的 torch）——
      · 只把外部目录**追加**到 sys.path → 本地那份 CPU 版仍然先被 import（等于没换）；
      · 把外部目录**插到最前** → 它的 numpy/transformers 也会一起顶上来，污染整个环境。
    所以只替换 `torch` 这一个模块。**必须在任何 import torch 之前调用。**
    （调用点：translate.HyMT2 构造、scripts/export_fireredvad_onnx.py。）

    返回一句人话（没配置 → 空串）；挂载失败会退回本地 torch，绝不抛异常。
    """
    if "torch" in sys.modules:
        return ""            # 已经导入过了：调用点保证在 import 之前，这里只是保险
    site = external_torch_site()
    if site is None:
        return ""
    import importlib.util
    pkg = site / "torch"
    _shadow_torch_metadata(site)      # 版本号也要对齐，否则 transformers 会选错代码路径
    try:
        spec = importlib.util.spec_from_file_location(
            "torch", pkg / "__init__.py", submodule_search_locations=[str(pkg)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules["torch"] = mod        # 先占位：torch 内部再 import torch 要能拿到自己
        spec.loader.exec_module(mod)
    except Exception as e:
        sys.modules.pop("torch", None)    # 退回本地 torch
        return f"外部 torch 挂载失败，已退回本地：{type(e).__name__}: {str(e)[:80]}"
    try:
        ok = bool(mod.cuda.is_available())
        tag = f"CUDA {mod.version.cuda}" if ok else "CUDA 不可用"
    except Exception:
        tag = "CUDA 状态未知"
    return f"外部 torch {mod.__version__}（{tag}）← {site}"


def ensure_external_torch() -> list:
    """冻结版（打包 exe）挂载源码环境里的 torch/transformers，让打包版也能用 Qwen。

    打包版按设计不含 torch（396MB vs 3GB），但 exe 部署在项目根目录时，
    旁边就有 .venv，可以借用：
      1. 用户配置的 torch_external_dir（最省事）
      2. .pth 指向的外部 site-packages（CUDA torch，如 E:\\applications\\0manbo）
      3. 项目 .venv 的 site-packages（transformers / tokenizers / safetensors）
      4. 基础 Python 的 Lib + DLLs（打包时 torch 被排除，torch 依赖的 timeit 等
         标准库没进 base_library.zip，需要从原解释器补齐）
    路径不存在就静默跳过。

    ⚠ 源码环境（非 frozen）**不在这里挂用户配置的目录**：源码 venv 里已经装了 CPU 版
    torch，"追加路径"不生效。那条走 mount_external_torch()（只替换 torch 模块）。
    """
    cands = []
    if getattr(sys, "frozen", False):
        ext = external_torch_site()
        if ext is not None:
            cands.append(ext)
        venv = PROJECT_ROOT / ".venv"
        site = venv / "Lib" / "site-packages"
        pth = site / "zz-maisub-external-torch.pth"
        if pth.exists():
            lines = pth.read_text(encoding="utf-8").strip().splitlines()
            if lines:
                cands.append(Path(lines[0].strip()))
        cands.append(site)
        cfg = venv / "pyvenv.cfg"
        if cfg.exists():
            for line in cfg.read_text(encoding="utf-8").splitlines():
                if line.strip().lower().startswith("home"):
                    home = Path(line.split("=", 1)[1].strip())
                    cands.extend([home / "Lib", home / "DLLs"])
                    break
    added = []
    for c in cands:
        if c.is_dir() and str(c) not in sys.path:
            sys.path.append(str(c))   # 追加在打包目录之后，不覆盖已打包模块
            added.append(str(c))
    return added


ensure_external_torch()


_DLL_HANDLES: list = []     # 持有 add_dll_directory 的返回对象，防止被 GC


def register_nvidia_dlls() -> bool:
    """让 CTranslate2/faster-whisper 找到 cuDNN/cuBLAS DLL。

    来源两处（都存在都注册）：
      1. venv 内 pip 安装的 nvidia-* 包
      2. 外部 torch（zz-maisub-external-torch.pth 指向的 site-packages）自带的 torch/lib
    """
    site = Path(__file__).parents[1] / ".venv" / "Lib" / "site-packages"
    dirs = [site / "nvidia" / d / "bin"
            for d in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc")]
    # 外部 torch（.pth 接线）
    pth = site / "zz-maisub-external-torch.pth"
    if pth.exists():
        ext_site = Path(pth.read_text(encoding="utf-8").strip().splitlines()[0].strip())
        dirs.append(ext_site / "torch" / "lib")
    ok = True
    for d in dirs:
        if d.is_dir():
            # ⚠ 两个坑（实测）：
            # 1. add_dll_directory() 的返回对象必须**持有**，否则被 GC 后目录立即失效，
            #    onnxruntime 的 CUDA EP 就会报 "cudnn64_9.dll is missing"。
            # 2. 同时追加到 PATH（有些原生库按传统搜索顺序找 DLL）。
            _DLL_HANDLES.append(os.add_dll_directory(str(d)))
            try:
                os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]
            except Exception:
                pass
        else:
            ok = ok and d.name not in ("cublas", "cudnn")
    return ok


def resolve_glossary_path(value: str | Path | None) -> Path | None:
    """术语库路径解析：相对路径按**项目根目录**解析，绝对路径原样返回。

    这样 config.json 里可以写 `glossary.csv`（相对），发布到别的机器照样能用；
    老配置里的绝对路径也继续有效。
    """
    s = str(value or "").strip()
    if not s:
        return None
    path = Path(s)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


@dataclass
class AppConfig:
    # 识别
    asr_model: str = "large-v3-turbo"   # 唯一保留的 whisper 权重（small/medium 已弃用）
    asr_device: str = "cuda"            # cuda / cpu
    beam_size: int = 5                  # 解码束宽（5 提升中文/嘈杂鲁棒性，实测不增延迟）
    # int8_float16 / float16 —— 2026-09-19 实测（RTX 4060 Laptop + 现版 CT2）：
    # int8_float16 快 7~8%、p95 更好（309 vs 347ms），且整段拼接文本与 float16
    # **逐词一致** → 改为默认（§八十一）。旧记录"float16 快约 15%"在当前环境不成立。
    asr_compute_type: str = "int8_float16"
    # 识别后端：whisper=本地 faster-whisper（默认）；http=交给本地/远端服务
    # （vLLM 的 vllm serve / Qwen3-ASR 的 qwen-asr-serve / FunASR 服务都提供
    #   /v1/audio/transcriptions 之类的 HTTP 接口，填 asr_http_url 即可对接，
    #   换模型不用改主程序代码）
    asr_backend: str = "whisper"             # whisper / qwen3-onnx / http / ws
    asr_http_url: str = ""                   # 例如 http://127.0.0.1:8000/v1/audio/transcriptions
    asr_http_model: str = ""                 # 例如 Qwen/Qwen3-ASR-0.6B（服务端模型名，可空）
    # asr_backend="qwen3-onnx" 时用的本地模型目录（含 encoder/decoder*.onnx + tokenizer.json）
    asr_qwen_dir: str = "models/qwen3-asr-0.6b-onnx-int4"
    # ---- asr_backend="ws"：火山引擎「大模型流式语音识别」（WebSocket 双向流式）----
    # 端点：bigmodel_async=双向流式优化版（推荐，支持二遍识别）；
    #      bigmodel=双向流式；bigmodel_nostream=流式输入（15s 后或负包才返回，更准更慢）
    # 默认值是开发机实测通过的一套（豆包流式识别 2.0 的 plan 端点 + seedasr 资源），
    # 新用户填上 API Key 即可用；换了服务版本要按控制台改回对应端点/资源 ID。
    asr_ws_url: str = "wss://openspeech.bytedance.com/api/v3/plan/sauc/bigmodel_nostream"
    asr_ws_model_name: str = "bigmodel"      # 请求体里的 model_name（文档：目前只有 bigmodel）
    asr_ws_api_key: str = ""                 # 新版控制台：X-Api-Key（只填它就够）
    asr_ws_app_key: str = ""                 # 旧版控制台：X-Api-App-Key（APP ID）
    asr_ws_access_key: str = ""              # 旧版控制台：X-Api-Access-Key（Access Token）
    # 资源 ID：豆包流式识别 1.0 = volc.bigasr.sauc.duration（小时版）/
    #         .concurrent（并发版）；2.0 = volc.seedasr.sauc.*
    asr_ws_resource_id: str = "volc.seedasr.sauc.duration"
    # 延迟
    loop_tick_ms: int = 100             # 实时循环节拍：越小字幕出得越快（250→100 中位快 82ms）
    progressive_display: bool = True    # 识别一出先显示原文，译文好了再补（仅双语模式）
    stream_translation: bool = True     # 译文边生成边上屏（仅 CT2 引擎支持，其他自动回落）
    # 翻译
    engine: str = "hymt2"               # hymt2（推荐，权重自备）/ qwen / qwen3
    context_sentences: int = 7
    # 外部 CUDA torch 目录（进阶）：填了就优先用它的 torch，省下 2.5GB 下载。
    # 可填 site-packages 或 venv 根目录；留空 = 用本环境的 torch。
    # 只在 HyMT2（走 PyTorch）上用得到；改了要重启（挂载发生在 import torch 之前）。
    torch_external_dir: str = ""
    # 显示
    bilingual: bool = True              # 由 display_mode 推导（导出用）：仅 bilingual 模式为真
    display_mode: str = "bilingual"     # bilingual 双语 / target 仅译文 / source 仅原文
    history_lines: int = 2
    # 每句字幕在当前行至少停留多久（ms）再让位给下一句（**等译文定稿后再计时**）。
    # 管线突发时一次会出 2~3 句，没这个停顿就"只看到最后一句"。
    subtitle_dwell_ms: float = 2000.0
    font_size_src: int = 14
    font_size_dst: int = 19
    opacity: float = 0.92
    screen: int = 0
    click_through: bool = False
    # 流式：realtime=边说边出（ASR 也实时出进度）/ sentence=整句识别完再出
    stream_mode: str = "realtime"       # realtime / sentence
    partial_interval_ms: int = 800      # 实时模式下"部分识别"的最小间隔
    partial_min_s: float = 0.8          # 语音不足这么长不做部分识别
    partial_max_s: float = 5.0          # 已说超过这么长就停止部分识别（等最终识别，省算力）
    # 智能分句
    end_silence_ms: float = 350.0       # 句尾静音多久判定一句话结束（Silero/FSMN 用）
    long_after_s: float = 6.0           # 超过这么长的句子，改用更短的静音阈值
    long_end_silence_ms: float = 220.0  # 长句场景下的句尾静音阈值
    max_sentence_s: float = 10.0        # 单句硬上限（无静音也切；Silero/FSMN 用）
    min_sentence_s: float = 0.6         # 短于此视为噪声，丢弃
    min_rms: float = 60.0               # 能量门：片段 RMS 低于此值视为没人说话（挡近静音幻觉）
    min_speech_ratio: float = 0.15      # 语音占比门：低于此值视为音乐/纯伴奏（听歌要放宽）
    merge_short_ms: float = 800.0       # 短片段挂起：说得比这短先不断句，等下一截连起来
    # 撞单句上限被强切后的"续接段"：合并前先让本地 Qwen 给两截拼起来的文本
    # **补标点**，接缝处有句末标点就另起一行（旧行为是无条件合并成一行）。
    # 实测 0.2~0.4s/次，只在这类续接段发生；关掉则回到"按长度硬拼"的老行为。
    seam_punct: bool = True
    # 补标点用哪个引擎：auto=中文走 CPU、其余走 Qwen（推荐）；
    #   cpu  = FunASR CT-Transformer（ONNX，1~4ms、不占 GPU，但只对中文可靠）；
    #   qwen = 本地 Qwen 对话模型（0.3~0.5s、占 GPU，中英日韩都能用）。
    punct_engine: str = "auto"
    punct_cpu_dir: str = "models/punc-ct-transformer-zh-en-onnx"
    # 每行**定稿前**补一次标点（2026-09-17）：whisper 在"撞上限强切"的片段上几乎
    # 不给句读（实测 4s 片段只有 1 个标点、10s 片段 5 个），屏幕上就是一串半句。
    # 中文走 CPU 小模型（1~4ms、不占 GPU），英/日/韩走本地 Qwen（0.3~0.5s、占 GPU）；
    # 带"不许改词"校验，校验不过就用原文（宁可没标点，也不能改字）。
    # Qwen3-ASR / 云端识别自己会给标点，用那些后端时可以关掉省时间。
    punct_final: bool = True
    # 前卷：撞上限强切出来的片段补 0.25s 前文音频再识别，找回被切掉的半个词
    # （实测：不加前卷时 "…show rover | around We've got…" 的 around 会被整段丢掉，
    #  字幕少一个词；加了就回来了）。关掉回到"从切点直接解码"的老行为。
    pre_roll: bool = True
    # VAD 引擎（可切换，缺依赖/缺模型自动回退 Silero）：
    #   firered = FireRedVAD（SOTA，100+ 语言；实测 9 段/平均 3.51s，覆盖最全）★默认
    #   fsmn    = FSMN-VAD（备选；实测 8 段/平均 3.73s）
    #   silero  = 现役逐帧 VAD（切得最碎：14 段/平均 1.78s，但零额外依赖）
    vad_engine: str = "firered"
    vad_firered_dir: str = "models/fireredvad-onnx"
    vad_fsmn_dir: str = "models/fsmn-vad-onnx"
    # FireRedVAD：句间静音超过这么久才断句（帧 10ms）。500 = 低延迟预设
    firered_min_silence_ms: float = 500.0
    # 单句上限：连续说话（播客/解说）没有停顿时靠它强制断句。
    # **体感延迟 ≈ 段长 + 0.7s**，所以它是最直接的旋钮：4s 是低延迟档（设置里
    # 「低延迟预设」会设成 4），7s 是默认档 —— 句子更完整、但要多等 3 秒左右。
    firered_max_speech_s: float = 7.0
    # 音频断流（暂停/结束）多久后把"最后一句"立刻定稿，ms。
    # 不加它：最后一句要等下一段音频才出现（字幕落后一句、不连贯）
    idle_flush_ms: float = 2500.0

    drop_hallucinations: bool = True    # 丢弃"ご視聴ありがとうございました"这类经典幻觉句
    # 源语言：auto=自动检测（默认）；en/ja/ko/zh=强制按该语言识别（不再自动检测）
    source_language: str = "auto"
    split_long_chars: int = 60          # 识别文本超过这么多字且含句末标点 → 按标点再分句
    # 术语（**你自己的数据，不进 git**；仓库里给的是 glossary.example.csv 样例）
    # **默认不启用**（空）：术语会进识别提示词与翻译术语保护，配错了会让模型"无中生有"
    # （整表喂识别侧就是老坑）。只有用户在设置里填了路径、或按 F8 在编辑器里保存过，
    # 才算"自己设置"，此前一律不读。相对路径按项目根解析。
    glossary_path: str = ""
    # 识别/翻译放进子进程（2026-09-16）：原生 GPU 调用卡死时只重启子进程，
    # 悬浮窗/采集/VAD 不受影响（详见 maisubtitle/gpu_proc.py）。
    gpu_subprocess: bool = True
    # 阶段5：蓝牙等音频外设延迟补偿（字幕显示延后毫秒）
    display_offset_ms: int = -2
    # 悬浮窗位置记忆（§八十二）：拖动释放即写回 "x,y"，启动时钳进可用区域恢复
    overlay_pos: str = ""
    # 字幕窗宽度占屏比（§八十三，设置页可调 50~95%）
    width_pct: float = 0.875
    # 静默期"监听中 mm:ss"角标开关（§八十二新增的可见性功能，允许关）
    idle_notice: bool = True
    # 用户改过的全局热键（§八十四）：action -> 键（如 {"export": "f4"}）；
    # 默认键与动作清单见 live_demo.HOTKEYS。空 = 全部默认。
    hotkey_map: dict = field(default_factory=dict)
    # 其他
    log_max_mb: int = 100               # 阶段5：日志限额

    def save(self, path: Path = CONFIG_PATH):
        # 原子写：先写临时文件再 os.replace —— 本项目有看门狗 taskkill 强杀路径，
        # 写一半被杀会留下截断的 json，下次 load() 静默回默认值、设置（含密钥）全丢
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "AppConfig":
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                cfg = cls()
                for k, v in data.items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
                # 显示模式是唯一来源，双语字段由它推导：老 config.json 里两者可能
                # 已经矛盾（老设置界面有独立的"双语显示"勾选框），加载时一次对齐。
                # 顺手把非法值挡回去（config.json 是用户可手改的边界输入）。
                if cfg.display_mode not in ("bilingual", "target", "source"):
                    cfg.display_mode = "bilingual"
                cfg.bilingual = (cfg.display_mode == "bilingual")
                return cfg
            except Exception:
                pass
        return cls()



    @property
    def glossary_file(self):
        """术语库文件（相对路径按项目根解析）；未配置返回 None。"""
        return resolve_glossary_path(self.glossary_path)

    def ws_opts(self) -> dict:
        """火山流式识别（asr_backend=ws）的参数包。

        主进程与 GPU 子进程各建一次 asr，字段口径必须一致 —— 放这里一份，免得
        两边各写一遍（漏一个字段的表现是"设置里填了却不生效"，很难查）。
        """
        return {"url": self.asr_ws_url, "api_key": self.asr_ws_api_key,
                "app_key": self.asr_ws_app_key,
                "access_key": self.asr_ws_access_key,
                "resource_id": self.asr_ws_resource_id,
                "model_name": self.asr_ws_model_name}

def save_preserving_cli(cfg: "AppConfig", overrides: dict,
                        path: Path = CONFIG_PATH):
    """退出保存：把 CLI 本次透传的字段还原成磁盘原值后再写文件。

    透传参数（--engine / --model / --glossary）只应影响本次运行。
    若不还原，用"兜底 bat 启动过一次"就会把用户在设置里的选择永久改掉
    （曾发生：bat 传 --engine qwen，退出回写把 config.json 里的 qwen3 覆盖成 qwen）。

    仅当该字段运行期未被再次修改（仍等于覆盖值）时才还原——用户在设置对话框里
    明确改过的值会正常保存。
    overrides: {字段名: (磁盘原值, 本次覆盖值)}
    """
    restored = {}
    for key, (orig, applied) in overrides.items():
        if getattr(cfg, key) == applied:
            restored[key] = applied
            setattr(cfg, key, orig)
    cfg.save(path)
    for key, val in restored.items():
        setattr(cfg, key, val)      # 还原内存值，不影响本次运行的收尾流程


