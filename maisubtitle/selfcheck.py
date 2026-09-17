"""启动自检：把"缺什么、怎么补"用人话说清楚（**不阻断启动、不改管线行为**）。

发布体验的一部分：以前缺权重时的表现是"卡在加载里 / 抛一句 CT2/ORT 的英文错"，
用户不知道该怎么办。这里只做检查与提示，实际行为仍由管线自己的回退逻辑决定
（例如 FireRedVAD 缺失 → 自动回落 Silero；hymt2 权重没装 → 回落 QwenCT2）。

三个入口都接这一份：启动日志、托盘通知、设置窗口的模型状态行。
`summary()/check()` 支持传入"界面上还没保存的选择"（engine/backend/vad），
这样设置里换模型时状态行能立刻跟着变。
"""
from __future__ import annotations

from .asr import WS_BACKENDS
from .config import MODELS_DIR

# 翻译引擎 → [(目录, 判据文件), …]：第一个是本引擎自己的模型，后面是回落链
ENGINE_CANDIDATES = {
    "qwen": [("Qwen2.5-1.5B-Instruct-ct2", "model.bin")],
    "qwen3": [("Qwen3-1.7B-ct2", "model.bin"),
              ("Qwen2.5-1.5B-Instruct-ct2", "model.bin")],
    "hymt2": [("Hy-MT2-1.8B", "config.json"),
              ("Qwen2.5-1.5B-Instruct-ct2", "model.bin")],
}


def _vad_state(cfg, vad: str | None = None) -> tuple[bool, str]:
    """返回 (是否就绪, 说明)。"""
    v = str(vad or getattr(cfg, "vad_engine", "firered") or "firered").lower()
    if v == "firered":
        ok = (MODELS_DIR / "fireredvad-onnx" / "stream_vad.onnx").exists()
        return ok, "FireRedVAD"
    if v == "fsmn":
        ok = (MODELS_DIR / "fsmn-vad-onnx" / "model_quant.onnx").exists()
        return ok, "FSMN-VAD"
    ok = (MODELS_DIR / "silero_vad.onnx").exists()
    return ok, "Silero VAD"


def _asr_state(cfg, backend: str | None = None) -> tuple[bool, str]:
    b = str(backend or getattr(cfg, "asr_backend", "whisper") or "whisper").strip().lower()
    if b in ("http", "vllm", "qwen3asr", "qwen3-asr", "funasr", "service"):
        return True, "HTTP 服务（远端）"
    if b in WS_BACKENDS:
        # 云端：本地无权重可查，能查的是"密钥填了没"（没填 = 每段都会失败）
        if _ws_has_key(cfg):
            host = str(getattr(cfg, "asr_ws_url", "") or "").split("//")[-1].split("/")[0]
            return True, f"火山流式识别（远端 {host or '未填地址'}）"
        return False, "火山流式识别（未填密钥 → 每段都会失败）"
    if b in ("qwen3-onnx", "qwen3onnx", "qwen3_asr_onnx", "qwenasr"):
        from .asr import find_qwen_dir
        d = find_qwen_dir(str(getattr(cfg, "asr_qwen_dir", "") or ""))
        return d is not None, f"Qwen3-ASR（{d.name if d else '未找到目录'}）"
    d = MODELS_DIR / "faster-whisper-large-v3-turbo"
    return (d / "model.bin").exists(), "Whisper large-v3-turbo"


def _ws_has_key(cfg) -> bool:
    """火山流式识别是否至少有一套密钥（新版 X-Api-Key，或旧版 App Key + Access Key）。"""
    if str(getattr(cfg, "asr_ws_api_key", "") or "").strip():
        return True
    return bool(str(getattr(cfg, "asr_ws_app_key", "") or "").strip()
                and str(getattr(cfg, "asr_ws_access_key", "") or "").strip())


def _mt_state(cfg, engine: str | None = None) -> tuple[bool, str]:
    """翻译引擎状态；**回落时明确指出"实际用哪个"**（否则 hymt2 后面跟着 Qwen 目录会误导）。"""
    eng = str(engine or getattr(cfg, "engine", "qwen") or "qwen").strip().lower()
    cands = ENGINE_CANDIDATES.get(eng, ENGINE_CANDIDATES["qwen"])
    for i, (d, marker) in enumerate(cands):
        if (MODELS_DIR / d / marker).exists():
            if i == 0:
                return True, f"翻译 {eng}（{d}）"
            return True, f"翻译 {eng}（权重未装 → 实际回落 {d}）"
    return False, f"翻译 {eng}（{cands[0][0]} 未找到 → 本次只出原文）"


def summary(cfg, engine: str | None = None, backend: str | None = None,
            vad: str | None = None) -> list[str]:
    """给日志/设置窗口用的一行行状态（✔/✘ + 名称）。"""
    out = []
    for ok, name in (_asr_state(cfg, backend), _mt_state(cfg, engine), _vad_state(cfg, vad)):
        out.append(f"{'✔' if ok else '✘'} {name}")
    return out


def check(cfg, engine: str | None = None, backend: str | None = None,
          vad: str | None = None) -> list[str]:
    """返回问题清单（空 = 齐全）；每条形如 "缺 X → 怎么办"。"""
    msgs: list[str] = []
    asr_ok, asr_name = _asr_state(cfg, backend)
    if not asr_ok:
        b = str(backend or getattr(cfg, "asr_backend", "whisper")
                or "whisper").strip().lower()
        if b in WS_BACKENDS:
            msgs.append("火山流式识别没填密钥（每段都会失败）→ 设置 → 识别引擎 → "
                        "火山引擎流式识别：填 X-Api-Key（新版控制台），"
                        "或旧版的 App Key + Access Key")
        elif "Qwen3" in asr_name:
            msgs.append("缺 Qwen3-ASR 模型目录（需含三个 int4 图）→ 放进 models/ 或 "
                        "python scripts/download_models.py --only qwen3_asr_0_6b_onnx_int4")
        else:
            msgs.append("缺 Whisper 识别权重 → python scripts/download_models.py "
                        "--only whisper_turbo")
    mt_ok, mt_name = _mt_state(cfg, engine)
    if not mt_ok:
        msgs.append(mt_name + "；补救：python scripts/download_models.py --only qwen_1_5b_hf，"
                    "再 python scripts/model_manager.py convert qwen_1_5b")
    vad_ok, vad_name = _vad_state(cfg, vad)
    if not vad_ok:
        msgs.append(f"缺 {vad_name} 模型（将自动回落其它 VAD）→ "
                    "python scripts/model_manager.py list 查看补救方式")
    return msgs