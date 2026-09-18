# -*- coding: utf-8 -*-
"""Qwen3-ASR-0.6B 本地 ONNX 推理（int4，onnxruntime，CPU/GPU）。

**按官方参考实现逐行对齐**（github.com/andrewleech/qwen3-asr-onnx 的 src/），
三个此前踩过的坑全部照官方写法修正：

1. **mel 要丢掉最后一个 STFT 帧**（对齐 WhisperFeatureExtractor）——
   否则帧数多 1 → 音频 token 数与提示词对不上 → 输出复读。
2. **audio_offset = 第一个 `<|audio_pad|>` 的下标**（官方 inference.py 的
   get_audio_pad_range[0]；不是 audio_start，也不是 0）。
3. **提示词用固定结构**：system 空 / user 带音频 / assistant 直接生成；
   词 id 由本地 tokenizer 编码（"system"=8948、"user"=872，
   官方参考里硬编码的 9125/882 是旧 tokenizer 的值，会退化成 " Current"/" time"）。

音频 token 数 N = encoder 实际输出长度（并用官方公式 feat_extract_output_lengths
对账：N = conv3(len % 100) + (len // 100) * 13）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- mel 特征
def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    """Slaney 公式（librosa 同款）。

    ⚠ 高频段是**对数**关系：mel = 15 + ln(f/1000)/logstep，
    曾把它写成 (f-1000)/1000（线性）→ 高频 mel bin 全错、能量为零，
    特征等于废的（encoder 只在开头几秒"听"到东西）。这是转写退化的真凶。
    """
    f_min, f_sp = 0.0, 200.0 / 3
    mel = (f - f_min) / f_sp
    min_log_hz, min_log_mel, logstep = 1000.0, 15.0, np.log(6.4) / 27.0
    log_t = np.log(np.maximum(f, 1e-6) / min_log_hz)
    return np.where(f >= min_log_hz, min_log_mel + log_t / logstep, mel)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Slaney 归一化 mel 滤波器组（= librosa.filters.mel(norm='slaney')）。

    优先用 librosa（官方参考就是它）；没有则用等价的 numpy 实现。
    ⚠ 两个坑：①三角形要用 **Hz 差**归一化（用 bin 差会差 40 倍 = sr/n_fft）；
    ②Slaney 面积归一化 enorm = 2/(mel_f[i+2]-mel_f[i]) 也不能漏。
    """
    try:
        import librosa
        return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
                                   fmin=fmin, fmax=fmax, norm="slaney").astype(np.float32)
    except Exception:
        pass
    n_freqs = n_fft // 2 + 1
    mel_min, mel_max = _hz_to_mel(np.array([fmin]))[0], _hz_to_mel(np.array([fmax]))[0]
    mel_f = np.linspace(mel_min, mel_max, n_mels + 2)
    freqs = 200.0 / 3 * mel_f
    logstep = np.log(6.4) / 27.0
    freqs = np.where(mel_f >= 15.0, 1000.0 * np.exp((mel_f - 15.0) * logstep), freqs)  # Hz 轴
    fftfreqs = np.arange(n_freqs) * (sr / n_fft)
    fdiff = np.diff(freqs)
    ramps = np.subtract.outer(fftfreqs, freqs)          # [n_freqs, n_mels+2]
    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        lower = -ramps[:, i] / fdiff[i]
        upper = ramps[:, i + 2] / fdiff[i + 1]
        fb[i] = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (freqs[2:n_mels + 2] - freqs[:n_mels])   # Slaney 面积归一化
    return (fb * enorm[:, None]).astype(np.float32)


def log_mel(audio_f32: np.ndarray, sample_rate: int = 16000,
            n_mels: int = 128, n_fft: int = 400, hop: int = 160) -> np.ndarray:
    """Whisper 参数 log-mel：[n_mels, T]（**丢掉最后一帧**，与官方一致）。"""
    audio = np.asarray(audio_f32, dtype=np.float32)
    audio = np.pad(audio, (n_fft // 2, n_fft // 2), mode="reflect")
    win = np.hanning(n_fft + 1)[:n_fft].astype(np.float32)         # periodic Hann
    n_frames = 1 + (len(audio) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    spec = np.abs(np.fft.rfft(audio[idx] * win, axis=1)) ** 2      # [T, 201]
    fb = _mel_filterbank(sample_rate, n_fft, n_mels, 0.0, 8000.0)
    mel = spec @ fb.T
    log_spec = np.log10(np.maximum(mel, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec[:-1].T.astype(np.float32)     # ⚠ 丢最后一帧（Whisper 行为）


def _conv_out_len(t: int) -> int:
    return (t + 1) // 2


def feat_extract_output_lengths(mel_frames: int) -> int:
    """官方公式：mel 帧数 → 音频 token 数（CONV_WINDOW=100, TOKENS_PER_WINDOW=13）。"""
    leave = mel_frames % 100
    t = _conv_out_len(_conv_out_len(_conv_out_len(leave)))
    return t + (mel_frames // 100) * 13


# ---------------------------------------------------------------- 识别器
class Qwen3AsrOnnx:
    """与 WhisperASR 同接口：transcribe(pcm, language=..., ...) -> (text, meta)。"""

    LANG_NAME = {"en": "English", "ja": "Japanese", "ko": "Korean", "zh": "Chinese"}
    NEWLINE_TOKEN_ID = 198

    def __init__(self, model_dir: str = "models/qwen3-asr-0.6b-onnx-int4",
                 device: str = "cuda", max_new_tokens: int = 448, context: str = ""):
        import onnxruntime as ort
        d = Path(model_dir)
        if not (d / "encoder.int4.onnx").exists():
            raise FileNotFoundError(f"缺少 {d/'encoder.int4.onnx'}")
        self._dir = str(d)
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        if "mel" not in cfg or "special_tokens" not in cfg:
            # 实锤（2026-09-18）：该目录的 config.json 曾被一次**下错仓库**的下载覆盖成
            # transformers 版（只有 architectures/model_type/…），运行期抛 `KeyError: 'mel'`，
            # 完全看不出是配置被换掉了。这里直接说清楚缺什么、从哪补。
            raise ValueError(
                f"{d/'config.json'} 不是 ONNX 导出方的配置（缺 mel / special_tokens）："
                "该文件被别的仓库覆盖过。修复：重新下载该模型 "
                "（scripts/download_models.py --only qwen3_asr_0_6b_onnx_int4，"
                "源为 andrewleech/qwen3-asr-0.6b-onnx）")
        self.spec = cfg["mel"]
        st = cfg["special_tokens"]
        self.eos_ids = set(st["eos_token_ids"]) | {st["im_end_token_id"]}
        self.audio_pad_id = st["audio_pad_token_id"]
        self.asr_text_id = st["asr_text_token_id"]
        self.im_start_id = st["im_start_token_id"]
        self.im_end_id = st["im_end_token_id"]
        self.audio_start_id = st["audio_start_token_id"]
        self.audio_end_id = st["audio_end_token_id"]
        self.max_new_tokens = max_new_tokens
        self.context = context                     # system 轮文本（术语偏置用）

        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        try:                                       # 外部 torch 的 CUDA DLL 必须先注册
            from .config import register_nvidia_dlls
            register_nvidia_dlls()
        except Exception:
            pass
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" \
            else ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(str(d / "encoder.int4.onnx"), so, providers=prov)
        self.dec_init = ort.InferenceSession(str(d / "decoder_init.int4.onnx"), so, providers=prov)
        self.dec_step = ort.InferenceSession(str(d / "decoder_step.int4.onnx"), so, providers=prov)

        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self._system_ids = self.tok.encode("system\n", add_special_tokens=False).ids
        self._user_ids = self.tok.encode("user\n", add_special_tokens=False).ids
        self._assistant_ids = self.tok.encode("assistant\n", add_special_tokens=False).ids
        self._embed_fp16 = None

    # ---------- 内部 ----------
    def _load_embed(self) -> np.ndarray:
        """embed_tokens.bin：裸 fp16 数组 [151936,1024]，保留 fp16、按行 cast。"""
        raw = np.fromfile(Path(self._dir) / "embed_tokens.bin", dtype=np.float16)
        return raw[:151936 * 1024].reshape(151936, 1024)

    def _embed_rows(self, ids_row: np.ndarray) -> np.ndarray:
        return self._embed_fp16[np.asarray(ids_row, dtype=np.int64)].astype(np.float32)

    def _prompt_ids(self, n_audio: int, language: str | None) -> tuple[np.ndarray, int]:
        """官方结构：
        <|im_start|>system\\n<|im_end|>\\n<|im_start|>user\\n<audio>…<|im_end|>\\n
        <|im_start|>assistant\\n[language X]<asr_text>"""
        ids = [self.im_start_id]
        if self.context:
            ids += self.tok.encode("system\n" + self.context, add_special_tokens=False).ids
        else:
            ids += self._system_ids
        ids += [self.im_end_id, self.NEWLINE_TOKEN_ID, self.im_start_id]
        ids += self._user_ids
        ids += [self.audio_start_id] + [self.audio_pad_id] * n_audio + [self.audio_end_id]
        ids += [self.im_end_id, self.NEWLINE_TOKEN_ID, self.im_start_id]
        ids += self._assistant_ids
        if language:                     # 语言前导（官方格式 "language {Name}"）
            ids += self.tok.encode("language " + self.LANG_NAME.get(language, "English"),
                                   add_special_tokens=False).ids
        ids += [self.asr_text_id]
        # ⚠ audio_offset = 第一个 <|audio_pad|> 的下标（官方 get_audio_pad_range[0]）
        audio_offset = ids.index(self.audio_pad_id)
        return np.array([ids], dtype=np.int64), audio_offset

    # ---------- 对外 ----------
    def transcribe(self, audio: np.ndarray, language: str | None = None,
                   prompt: str | None = None, return_confidence: bool = False,
                   beam_size: int | None = None, without_timestamps: bool = True):
        t0 = time.perf_counter()
        if self._embed_fp16 is None:
            self._embed_fp16 = self._load_embed()
        wav = (audio.astype(np.float32) / 32768.0 if audio.dtype == np.int16
               else np.asarray(audio, dtype=np.float32))
        if prompt:                                   # 术语/词汇偏置进 system 轮
            self.context = (self.context + "\n" + prompt).strip()

        mel = log_mel(wav, sample_rate=self.spec["sample_rate"],
                      n_fft=self.spec["n_fft"], hop=self.spec["hop_length"],
                      n_mels=self.spec["n_mels"])
        t_mel = time.perf_counter() - t0
        t1 = time.perf_counter()
        feats = self.enc.run(None, {"mel": mel[None, :, :]})[0]        # [1,N,1024]
        t_enc = time.perf_counter() - t1
        n_audio = int(feats.shape[1])
        expect = feat_extract_output_lengths(mel.shape[1])    # 对账用（不一致=特征管线有问题）

        ids, audio_offset = self._prompt_ids(n_audio, language)
        pos = np.arange(ids.shape[1], dtype=np.int64)[None, :]
        t1 = time.perf_counter()
        logits, pk, pv = self.dec_init.run(None, {
            "input_ids": ids, "position_ids": pos,
            "audio_features": feats.astype(np.float32),
            "audio_offset": np.array([audio_offset], dtype=np.int64)})
        t_prefill = time.perf_counter() - t1
        next_id = int(np.argmax(logits[0, -1]))

        out_ids: list[int] = []
        pos_now = ids.shape[1]
        t_decode = 0.0
        while len(out_ids) < self.max_new_tokens:
            if next_id in self.eos_ids:
                break
            out_ids.append(next_id)
            emb = self._embed_rows(np.array([next_id]))[None, :, :]
            t2 = time.perf_counter()
            logits, pk, pv = self.dec_step.run(None, {
                "input_embeds": emb,
                "position_ids": np.array([[pos_now]], dtype=np.int64),
                "past_keys": pk, "past_values": pv})
            t_decode += time.perf_counter() - t2
            next_id = int(np.argmax(logits[0, -1]))
            pos_now += 1
        text = self.tok.decode(out_ids).strip()
        dt = time.perf_counter() - t0
        n_tok = len(out_ids)
        return text, {"latency_s": round(dt, 3), "audio_s": round(len(wav) / 16000, 2),
                      "rtf": round(dt / max(len(wav) / 16000, 1e-6), 3),
                      "n_audio": n_audio, "expect": expect,
                      # —— 分阶段耗时（ms）：定位"延迟花在哪"用 ——
                      "mel_ms": round(t_mel * 1000, 1),
                      "enc_ms": round(t_enc * 1000, 1),
                      "prefill_ms": round(t_prefill * 1000, 1),
                      "decode_ms": round(t_decode * 1000, 1),
                      "n_tokens": n_tok,
                      "ms_per_token": round(t_decode * 1000 / max(n_tok, 1), 2),
                      "engine": "qwen3-asr-onnx-int4"}

    def detect_lang(self, audio: np.ndarray):
        return "", 0.0        # 不做语种检测（配合固定源语言使用）
