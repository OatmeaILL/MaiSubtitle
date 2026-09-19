"""Silero VAD v4 流式分段（onnxruntime CPU，参数为阶段0冻结值）。"""
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .config import MODELS_DIR

SAMPLE_RATE = 16000
FRAME_MS = 32
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)  # 512


def _make_session(model_path: Path):
    """建 VAD 推理会话——**必须单线程 + 关闭自旋等待**。

    踩坑记录（2026-09-13 实测）：onnxruntime 默认按物理核数（本机 14）建 intra-op
    线程池，而 Silero VAD 一帧只有 512 采样的小 LSTM，多线程纯属反效果；更糟的是
    线程池在两次 run 之间**忙等自旋**，而实时循环每 250ms 就调一次，于是自旋永远
    不会退化成睡眠。

    A/B 实测（每 250ms 处理 8 帧，psutil 进程 CPU%）：
        默认线程池          → 1206%（12 个核持续空烧）
        单线程 + 关自旋      →   ~0%
    即单这一处就白烧掉约 12 个核。关掉后 VAD 精度/延迟无变化。
    """
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    for key in ("session.intra_op.allow_spinning", "session.inter_op.allow_spinning"):
        try:
            so.add_session_config_entry(key, "0")
        except Exception:
            pass
    return ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])


class SileroVAD:
    def __init__(self, threshold: float = 0.6):
        model_path = MODELS_DIR / "silero_vad.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"未找到 {model_path}，运行 scripts/download_models.py --only silero_vad")
        self.session = _make_session(model_path)
        self.threshold = threshold
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def reset(self):
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def frame_prob(self, frame: np.ndarray) -> float:
        x = frame.astype(np.float32) / 32768.0
        outs = self.session.run(None, {"x": x[None, :], "h": self._h, "c": self._c})
        self._h, self._c = outs[1], outs[2]
        return float(outs[0][0][0])

    def frames_probs(self, audio: np.ndarray):
        n = len(audio) // FRAME_SIZE
        for i in range(n):
            yield i * FRAME_SIZE, self.frame_prob(audio[i * FRAME_SIZE:(i + 1) * FRAME_SIZE])


def segment_audio(audio: np.ndarray, vad: SileroVAD,
                  start_hangover_ms: float = 150.0,
                  end_silence_ms: float = 300.0,
                  max_segment_ms: float = 12000.0,
                  pad_start_ms: float = 100.0,
                  pad_end_ms: float = 150.0) -> list:
    """离线/准实时分段：返回 [{start,end,max_prob}]（样本索引，含保护垫）。"""
    start_need = max(1, int(start_hangover_ms / FRAME_MS))
    end_need = max(1, int(end_silence_ms / FRAME_MS))
    max_frames = int(max_segment_ms / FRAME_MS)

    segments = []
    in_speech = False
    run_start = 0
    speech_run = 0
    silence_run = 0
    seg_frames = 0
    max_prob = 0.0

    def emit(end_s, silence_frames, pad_e):
        pad_s = int(pad_start_ms / 1000 * SAMPLE_RATE)
        pad_e_n = int(pad_e / 1000 * SAMPLE_RATE)   # pad_e 是**毫秒**，要换算成样本
        s = max(0, int(run_start) - pad_s)
        e = min(len(audio), int(end_s) - int(silence_frames) * FRAME_SIZE + pad_e_n)
        return {"start": s, "end": e, "max_prob": round(max_prob, 3)}

    for pos, prob in vad.frames_probs(audio):
        voiced = prob >= vad.threshold
        if not in_speech:
            if voiced:
                speech_run += 1
                if speech_run == 1:
                    run_start = pos
                if speech_run >= start_need:
                    in_speech = True
                    seg_frames = speech_run
                    max_prob = prob
                    silence_run = 0
            else:
                speech_run = 0
        else:
            seg_frames += 1
            max_prob = max(max_prob, prob)
            if voiced:
                silence_run = 0
            else:
                silence_run += 1
                if silence_run >= end_need:
                    segments.append(emit(pos, silence_run, pad_end_ms))
                    in_speech, speech_run, max_prob = False, 0, 0.0
                elif seg_frames >= max_frames:
                    segments.append(emit(pos, 0, 0))
                    in_speech, speech_run, max_prob = False, 0, 0.0

    if in_speech:
        segments.append(emit(len(audio), 0, pad_end_ms))
    return segments


class StreamingSegmenter:
    """实时流式分段（阶段2 修订）。

    与 segment_audio 的区别：跨 feed 调用保持分段状态，只有当句尾静音达到
    end_silence_ms 或单句长度达到 max_segment_ms 时才输出，因此不会在分析
    窗口边界把长句截断；没说完的语句留在缓冲区，下次调用继续累积。

    feed(pcm) 返回已完成的语句段列表，每项：
      {"pcm": np.ndarray, "start_sample": int, "end_sample": int, "max_prob": float}
    start_sample / end_sample 是相对本对象累计喂入样本流的绝对样本号。
    """

    def __init__(self, threshold: float = 0.6,
                 start_hangover_ms: float = 150.0,
                 hold_ms: float = 1200.0,
                 hold_extra_ms: float = 700.0,
                 end_silence_ms: float = 300.0,
                 max_segment_ms: float = 12000.0,
                 pad_start_ms: float = 100.0,
                 pad_end_ms: float = 150.0,
                 long_after_ms: float = 6000.0,
                 long_end_silence_ms: float = 260.0):
        self.vad = SileroVAD(threshold=threshold)
        self.start_need = max(1, int(start_hangover_ms / FRAME_MS))
        self.end_need = max(1, int(end_silence_ms / FRAME_MS))
        # 短片段挂起：说得很短（<hold_ms）时先别急着断句——等一小会儿，
        # 若马上又说话就把两截合成一句（解决 "Yeah." 这类碎片单独成行）。
        # 静音累计超过 end_need+hold_extra 帧就照样输出，保证短句也能及时上屏。
        self.hold_frames = max(1, int(hold_ms / FRAME_MS))
        self.hold_extra_frames = max(1, int(hold_extra_ms / FRAME_MS))
        # 长句专用：已经说了很多以后，遇到更短的停顿就切（避免攒成一大段）
        self.long_after_ms = long_after_ms
        self.long_after_frames = int(long_after_ms / FRAME_MS)
        self.long_end_need = max(1, int(long_end_silence_ms / FRAME_MS))
        self.max_frames = int(max_segment_ms / FRAME_MS)
        self._seg_counter = 0            # 语句序号（每次开始说话 +1）
        self.current_seg_id = 0
        self.pad_start = int(pad_start_ms / 1000 * SAMPLE_RATE)
        self.pad_end = int(pad_end_ms / 1000 * SAMPLE_RATE)
        self.buffer = np.zeros(0, dtype=np.int16)   # 已喂入、尚未丢弃的样本
        self.buffer_start = 0                        # buffer[0] 在样本流中的绝对号
        self.frame_pos = 0                           # buffer 中下一个待处理帧的下标
        self.total_samples = 0                       # 累计喂入样本数
        self.in_speech = False
        self.speech_run = 0
        self.silence_run = 0
        self.seg_frames = 0
        self.voiced_frames = 0
        self.max_prob = 0.0
        self.run_start = 0                           # 当前语句起点绝对样本号

    def feed(self, pcm: np.ndarray) -> list:
        if len(pcm):
            self.buffer = np.concatenate([self.buffer, pcm.astype(np.int16)])
            self.total_samples += len(pcm)
        segments = []
        while self.frame_pos + FRAME_SIZE <= len(self.buffer):
            frame = self.buffer[self.frame_pos:self.frame_pos + FRAME_SIZE]
            frame_start_abs = self.buffer_start + self.frame_pos
            prob = self.vad.frame_prob(frame)
            seg = self._step(frame_start_abs, prob)
            if seg is not None:
                segments.append(seg)
            self.frame_pos += FRAME_SIZE
        self._discard_consumed()
        return segments

    def _step(self, frame_start_abs: int, prob: float):
        voiced = prob >= self.vad.threshold
        if not self.in_speech:
            if voiced:
                self.speech_run += 1
                if self.speech_run == 1:
                    self.run_start = frame_start_abs
                if self.speech_run >= self.start_need:
                    self.in_speech = True
                    self.seg_frames = self.speech_run
                    self.voiced_frames = self.speech_run
                    self.max_prob = prob
                    self.silence_run = 0
                    # 新的一句：分配 id（实时模式用它把"部分识别"和最终识别对上）
                    self._seg_counter += 1
                    self.current_seg_id = self._seg_counter
            else:
                self.speech_run = 0
            return None
        self.seg_frames += 1
        self.max_prob = max(self.max_prob, prob)
        if voiced:
            self.voiced_frames += 1
            self.silence_run = 0
        else:
            self.silence_run += 1
        # 已经说够长时改用更短的静音阈值：长句里的自然停顿也能断句，
        # 短句仍用常规阈值（避免把正常换气切成两句）
        need = self.long_end_need if self.seg_frames >= self.long_after_frames \
            else self.end_need
        if self.silence_run >= need:
            # 短片段挂起：说得太短且静音还没久到"确实说完了" → 先不切继续等
            if (self.seg_frames < self.hold_frames
                    and self.silence_run < need + self.hold_extra_frames
                    and self.seg_frames < self.max_frames):
                return None
            # 句尾静音够了：静音第一帧处即语句结束点，再加尾部保护垫
            speech_end_abs = frame_start_abs - (self.silence_run - 1) * FRAME_SIZE
            seg = self._make_segment(speech_end_abs + self.pad_end)
            self._reset_speech()
            return seg
        if self.seg_frames >= self.max_frames:
            # 单句过长（连续说话无静音）：在当前位置强制切分
            seg = self._make_segment(frame_start_abs + FRAME_SIZE, cut="max")
            self._reset_speech()
            return seg
        return None

    def _make_segment(self, end_abs: int, cut: str = "silence"):
        start_abs = max(0, self.run_start - self.pad_start)
        end_abs = min(end_abs, self.total_samples)
        i = max(0, start_abs - self.buffer_start)
        j = min(len(self.buffer), end_abs - self.buffer_start)
        if j - i < FRAME_SIZE:
            return None
        ratio = (self.voiced_frames / self.seg_frames) if self.seg_frames else 0.0
        # cut：这一段为什么结束（"max"=撞上限被强切 → live.py 会把下一段续接上来）
        return {"pcm": self.buffer[i:j].copy(), "start_sample": start_abs,
                "end_sample": end_abs, "max_prob": round(self.max_prob, 3),
                "speech_ratio": round(ratio, 3), "cut": cut,
                "seg_id": self.current_seg_id}

    def pending(self):
        """正在说的那一句（还没判定结束）：{"seg_id", "pcm"} 或 None。

        实时模式用它做"部分识别"——边说边把已说部分先识别出来上屏。
        """
        if not self.in_speech:
            return None
        i = max(0, self.run_start - self.buffer_start - self.pad_start)
        j = len(self.buffer)
        if j - i < FRAME_SIZE:
            return None
        return {"seg_id": self.current_seg_id, "pcm": self.buffer[i:j].copy()}

    def flush(self) -> dict | None:
        """音频断流时立刻定稿"正在说的这一句"（不等句尾静音）。

        没有它：视频一暂停、一段 CG 结束、或播放器断流后不再送音频，
        最后一句会一直挂着；等下一段音频进来、静音累计够了才收尾 ——
        表现为"字幕不连贯：最后一句要等下一句音频才蹦出来"。
        """
        if not self.in_speech:
            return None
        seg = self._make_segment(self.total_samples, cut="flush")
        self._reset_speech()
        return seg

    def _reset_speech(self):
        self.in_speech = False
        self.speech_run = 0
        self.silence_run = 0
        self.seg_frames = 0
        self.voiced_frames = 0
        self.max_prob = 0.0

    def _discard_consumed(self):
        """丢弃已处理且不再需要的样本，同时保留语音起点前 pad_start 和未对齐尾帧。"""
        if self.in_speech:
            keep_from_abs = max(0, self.run_start - self.pad_start)
        else:
            keep_from_abs = max(0, self.total_samples - (self.pad_start + FRAME_SIZE))
        drop = keep_from_abs - self.buffer_start
        if drop > self.frame_pos:
            drop = self.frame_pos
        if drop <= 0:
            return
        self.buffer = self.buffer[drop:]
        self.buffer_start += drop
        self.frame_pos -= drop
