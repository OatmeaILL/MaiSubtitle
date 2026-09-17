"""faster-whisper 识别与语言检测。"""
import re
import time
from pathlib import Path

import numpy as np

from .config import MODELS_DIR

_MODEL_DIRS = {
    # 只保留这一个 whisper 权重：small/medium 已弃用并清理（quality 不如 turbo、
    # 还没它快），需要旧模型请自行下载并在此登记。
    "large-v3-turbo": "faster-whisper-large-v3-turbo",
    "turbo": "faster-whisper-large-v3-turbo",
}
TRANSLATABLE = {"en", "ja", "ko"}

# 识别后端键别名：ws（火山引擎大模型流式识别，见 asr_ws.py）。
# 单独提一个常量，是为了让 live / selfcheck / 设置界面共用一份口径 —— 别名散落
# 在各处时，加一个别名就会漏掉一处（表现：某个拼法下"注意事项"不弹、状态行不对）。
WS_BACKENDS = ("ws", "websocket", "wss", "volc", "volc-ws", "volcengine")


def local_model_dir(size: str) -> Path | None:
    """该模型的本地权重目录；没下过返回 None。

    用来避免"以为本地有、其实悄悄去 HuggingFace 下载"：按名字构造未下载的模型时
    faster-whisper 会转去下载，无网就要卡约 100s 连接超时（实测）才回滚。
    """
    d = MODELS_DIR / _MODEL_DIRS.get(size, size)
    return d if d.exists() else None


# Qwen3-ASR ONNX 目录的三个标志文件（自动侦测判据）
_QWEN_MARKERS = ("encoder.int4.onnx", "decoder_init.int4.onnx", "decoder_step.int4.onnx")


def find_qwen_dir(explicit: str = "") -> Path | None:
    """定位 Qwen3-ASR ONNX 模型目录（设置界面"自动侦测"与 make_asr 共用一份）。

    优先显式配置（asr_qwen_dir，相对路径按项目根解析），否则扫描 models/ 下同时
    含三个 int4 图的目录。都找不到返回 None —— 调用方给出明确报错，而不是等
    ONNX Runtime 抛文件缺失。
    """
    cands: list[Path] = []
    if explicit:
        p = Path(explicit)
        cands.append(p if p.is_absolute() else MODELS_DIR.parent / p)
    if MODELS_DIR.is_dir():
        cands += [d for d in sorted(MODELS_DIR.iterdir()) if d.is_dir()]
    for d in cands:
        if all((d / m).exists() for m in _QWEN_MARKERS):
            return d
    return None


def join_segments(pieces: list[str]) -> str:
    """拼接 faster-whisper 的多个 segment，**不丢段间空格**。

    实锤（2026-09-16 日志）：原来对每段先 `.strip()` 再 `"".join()`，
    段首空格被吃掉 → 一次吐出多段时粘成
    "…martial championshipsMy godPlus tons of"、"I'm gonna go betterI'm gonna go better"。
    whisper 的 segment 文本自带前导空格（英文），所以正确做法是**按原样拼接**；
    只在"ASCII 字母数字紧贴 ASCII 字母数字"处补一个空格兜底（模型偶尔不给前导空格），
    CJK 之间不加（中日文本来就不带空格）。
    """
    out = ""
    for p in pieces:
        if not p:
            continue
        if (out and not out[-1].isspace() and not p[0].isspace()
                and out[-1].isascii() and p[0].isascii()
                and out[-1].isalnum() and p[0].isalnum()):
            out += " "
        out += p
    return re.sub(r"[ \t]{2,}", " ", out).strip()


class WhisperASR:
    def __init__(self, size: str = "large-v3-turbo", device: str = "cuda",
                 compute_type: str = "int8_float16"):
        from faster_whisper import WhisperModel
        local = local_model_dir(size)
        target = str(local) if local else size
        self.model = WhisperModel(target, device=device, compute_type=compute_type)
        self.size = size
        self.device = device
        self.compute_type = compute_type

    def detect_lang(self, audio: np.ndarray) -> tuple[str, float]:
        x = audio.astype(np.float32) / 32768.0
        if len(x) > 16000:
            x = x[:16000]
        lang, prob, _meta = self.model.detect_language(x)
        return str(lang), float(prob)

    def transcribe(self, audio: np.ndarray, language: str | None = None,
                   prompt: str | None = None,
                   return_confidence: bool = False, beam_size: int = 5,
                   without_timestamps: bool = False):
        """audio: int16 16k 单声道。返回 (text, meta)。

        return_confidence=True 时 meta 附带 avg_logprob / no_speech_prob
        （低置信二次识别，阶段4）。
        beam_size 默认 5：实测在本机 GPU 上与 beam=1 耗时无差别（~0.39s/3.5s），
        但对中文、嘈杂/含糊音频的鲁棒性明显更好（beam=1 时中文易出错字）。
        without_timestamps=True 让模型不预测时间戳 token，解码步数更少 → 更快；
        字幕时间轴是我们自己按 VAD 段算的，用不到模型时间戳。
        """
        x = audio.astype(np.float32) / 32768.0
        t0 = time.perf_counter()
        segments, info = self.model.transcribe(
            x, language=language, initial_prompt=prompt,
            beam_size=beam_size, vad_filter=False, condition_on_previous_text=False,
            without_timestamps=without_timestamps)
        texts, probs = [], []
        for s in segments:
            texts.append(s.text)
            probs.append({"avg_logprob": s.avg_logprob,
                          "no_speech_prob": s.no_speech_prob})
        t1 = time.perf_counter()
        meta = {"latency_s": round(t1 - t0, 3),
                "audio_s": round(len(audio) / 16000, 2),
                "rtf": round((t1 - t0) / max(len(audio) / 16000, 1e-6), 3),
                "lang": info.language,
                "lang_prob": round(float(info.language_probability), 3)}
        if return_confidence:
            meta["segments"] = probs
            # 段级置信度：最差段指标
            if probs:
                meta["avg_logprob"] = min(p["avg_logprob"] for p in probs)
                meta["no_speech_prob"] = max(p["no_speech_prob"] for p in probs)
        return join_segments(texts), meta


class HttpASR:
    """把识别交给 HTTP 服务（vLLM / qwen-asr-serve / FunASR 都吃这一套）。

    这样"换识别模型"不用改主程序：起一个服务（例如
    `vllm serve Qwen/Qwen3-ASR-0.6B` 或 `qwen-asr-serve ...`），把地址填进
    config.json 的 `asr_http_url`、并把 `asr_backend` 设为 `http` 即可。

    ⚠ 两点限制：
      1. 服务端通常不返回"语种检测"结果，所以走 HTTP 后端时**建议固定源语言**
         （config 的 source_language，或运行时按 F5 选择），否则管线不知道把
         这段文本当什么语言处理。
      2. 多一次 HTTP 往返（本机约 1~10ms，可忽略；跨机另算）。
    """

    def __init__(self, url: str, model: str = "", timeout: float = 60.0):
        import requests  # 延迟导入：不用 HTTP 后端时不给启动增加依赖
        self._requests = requests
        self.url = url
        self.model = model
        self.timeout = timeout
        self.load_s = 0.0

    @staticmethod
    def _wav_bytes(pcm: np.ndarray) -> bytes:
        """16k/mono/int16 PCM → 内存里的 WAV 字节（服务端普遍要求带容器头）。"""
        import io
        import wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(np.asarray(pcm, dtype=np.int16).tobytes())
        return buf.getvalue()

    def transcribe(self, audio: np.ndarray, language: str | None = None,
                   prompt: str | None = None, return_confidence: bool = False,
                   beam_size: int | None = None, without_timestamps: bool = True):
        t0 = time.perf_counter()
        files = {"file": ("seg.wav", self._wav_bytes(audio), "audio/wav")}
        data = {}
        if self.model:
            data["model"] = self.model
        if language:
            data["language"] = language
        r = self._requests.post(self.url, files=files, data=data,
                                timeout=self.timeout)
        r.raise_for_status()
        try:
            js = r.json()
        except Exception:
            js = {"text": r.text}
        text = js.get("text") or ""
        if not text:
            # FunASR 风格：{"results":[{"text": "..."}]}
            res = js.get("results") or []
            if res and isinstance(res[0], dict):
                text = res[0].get("text", "") or ""
        dt = time.perf_counter() - t0
        meta = {"latency_s": round(dt, 3), "audio_s": round(len(audio) / 16000, 2),
                "rtf": round(dt / max(len(audio) / 16000, 1e-6), 3),
                "http": r.status_code}
        words = js.get("words") or (js.get("segments") or [])
        if words and isinstance(words[0], dict):
            meta["no_speech_prob"] = float(words[0].get("no_speech_prob", 0.0) or 0.0)
            meta["avg_logprob"] = float(words[0].get("avg_logprob", 0.0) or 0.0)
        return str(text).strip(), meta

    def detect_lang(self, audio: np.ndarray):
        """HTTP 服务不做语种检测 → 返回空，交由固定源语言处理。"""
        return "", 0.0


def make_asr(backend: str = "whisper", model: str = "large-v3-turbo",
             device: str = "cuda", compute_type: str = "int8_float16",
             http_url: str = "", http_model: str = "",
             qwen_dir: str | None = None, ws: dict | None = None):
    """按配置构造识别后端。

    backend：
      whisper（默认）   = 本地 faster-whisper
      qwen3-onnx        = 本地 Qwen3-ASR-0.6B INT4（ONNX Runtime，不需要 torch）
      http / vllm / funasr = 交给 HTTP 服务（见 HttpASR）
      ws / websocket / volc = 火山引擎大模型流式识别（WebSocket 双向流式，见 asr_ws）

    ws：{"url","api_key","app_key","access_key","resource_id"}（asr_backend=ws 时才用）
    """
    b = (backend or "whisper").strip().lower()
    if b in ("whisper", "local", ""):
        ctype = compute_type or ("int8_float16" if device == "cuda" else "int8")
        return WhisperASR(model, device=device, compute_type=ctype)
    if b in ("qwen3-onnx", "qwen3onnx", "qwen3_asr_onnx", "qwenasr"):
        from .asr_qwen_onnx import Qwen3AsrOnnx
        # 模型目录自动侦测（显式配置 > models/ 下扫描）；找不到直接报清楚
        d = find_qwen_dir(qwen_dir or "")
        if d is None:
            raise FileNotFoundError(
                "未找到 Qwen3-ASR 模型目录（需含 encoder.int4.onnx / "
                "decoder_init.int4.onnx / decoder_step.int4.onnx；放进 models/ 即可）")
        return Qwen3AsrOnnx(model_dir=str(d), device=device)
    if b in WS_BACKENDS:
        from .asr_ws import VolcWsAsr
        o = dict(ws or {})
        url = str(o.get("url") or "").strip()
        if not url:
            url = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async"
        if not (str(o.get("api_key") or "").strip()
                or (str(o.get("app_key") or "").strip()
                    and str(o.get("access_key") or "").strip())):
            raise ValueError(
                "asr_backend=ws 但没填密钥：填 X-Api-Key（新版控制台），"
                "或旧版的 App Key + Access Key（设置 → 识别引擎 → 火山流式）")
        return VolcWsAsr(url, api_key=o.get("api_key", ""),
                         app_key=o.get("app_key", ""),
                         access_key=o.get("access_key", ""),
                         resource_id=o.get("resource_id", ""),
                         model_name=o.get("model_name", "") or "bigmodel")
    if b in ("http", "vllm", "qwen3asr", "qwen3-asr", "funasr", "service"):
        if not http_url:
            raise ValueError("asr_backend=http 但 asr_http_url 为空："
                             "请填写服务地址（如 http://127.0.0.1:8000/v1/audio/transcriptions）")
        return HttpASR(http_url, http_model)
    raise ValueError(f"未知的 asr_backend: {backend!r}"
                     f"（可选 whisper / qwen3-onnx / http / ws）")
