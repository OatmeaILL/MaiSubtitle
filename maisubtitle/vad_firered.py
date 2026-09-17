# -*- coding: utf-8 -*-
"""FireRedVAD 流式分段器（ONNX + CPU，主程序**不需要 torch**）。

来源：官方权重是 PyTorch(.pth.tar)，用 scripts/export_fireredvad_onnx.py 导出成
ONNX（已做 torch↔ONNX 数值对齐：prob 差 6e-8 / cache 差 5.7e-5）。运行时依赖：
    onnxruntime + numpy + kaldi_native_fbank（C++ 扩展，不含 torch）

接口与 vad.StreamingSegmenter 一致，可直接替换。

实测（高考听力 250s 起 60s、100ms 分块，min_silence=700ms）：
    Silero 14 段/平均 1.78s ｜ FSMN 8 段/平均 3.73s ｜ **FireRedVAD 9 段/平均 3.51s**
（覆盖 31.6s，比 FSMN 的 29.9s 更全）

三个已踩的坑（都写在注释里）：
 1. **样本尺度**：kaldi fbank 吃的是 ±32768 量级（和 sherpa-onnx 一样），
    不要除以 32768——那样 prob 全部接近 0，一段都切不出来。
 2. **缓存维度**：ONNX 的 8 个 cache 形状是 (1,128,T)，T 是 FSMN 记忆窗（≤19），
    不是 feat 的时间轴；导出时动态轴要设在 axis 2。
 3. **后处理状态机**：官方用 POSSIBLE_SPEECH/POSSIBLE_SILENCE 四态机，
    这里按官方逻辑移植（含 pad_start_frame 与 hit_max_speech 续段）。
"""
from __future__ import annotations

import json
from collections import deque

import numpy as np

from .vad import FRAME_SIZE, SileroVAD

FRAME_SHIFT = 160          # 10ms @16k：每帧对应的样本数
SAMPLE_RATE = 16000


class _State:
    SILENCE, POSSIBLE_SPEECH, SPEECH, POSSIBLE_SILENCE = 0, 1, 2, 3


class _PostProcessor:
    """官方 StreamVadPostprocessor 的无 torch 移植（行为保持一致）。"""

    def __init__(self, smooth_window_size=5, speech_threshold=0.5,
                 pad_start_frame=5, min_speech_frame=8,
                 max_speech_frame=2000, min_silence_frame=70):
        self.win = max(1, smooth_window_size)
        self.thr = speech_threshold
        self.pad = max(self.win, pad_start_frame)
        self.min_speech = min_speech_frame
        self.max_speech = max_speech_frame
        self.min_silence = min_silence_frame
        self.reset()

    def reset_state(self):
        """只重置状态机/平滑窗，**保留帧号计数**。

        ⚠ 不能直接 reset()：帧号是"帧 → 样本"的唯一映射依据，清零后
        后续段的 speech_start_frame 会变成很小的值，切到早已发过的音频
        （实测 flush 后新段起点跑到 0、与已交出的段重叠）。
        """
        self._q: deque[float] = deque()
        self._sum = 0.0
        self.state = _State.SILENCE
        self.speech_cnt = 0
        self.silence_cnt = 0
        self.hit_max = False
        self.last_start = -1
        self.last_end = self.n          # 下一句只能从当前帧之后开始

    def reset(self):
        self.n = 0
        self._q: deque[float] = deque()
        self._sum = 0.0
        self.state = _State.SILENCE
        self.speech_cnt = 0
        self.silence_cnt = 0
        self.hit_max = False
        self.last_start = -1
        self.last_end = -1

    def process(self, raw_prob: float) -> dict:
        self.n += 1
        self._q.append(raw_prob)
        self._sum += raw_prob
        if len(self._q) > self.win:
            self._sum -= self._q.popleft()
        smoothed = self._sum / len(self._q)
        is_speech = int(smoothed >= self.thr)
        r = {"frame_idx": self.n, "is_speech": is_speech,
             "is_speech_start": False, "is_speech_end": False,
             "speech_start_frame": -1, "speech_end_frame": -1}

        if self.hit_max:                       # 上一句被 max_speech 截断 → 立刻续开新句
            r["is_speech_start"] = True
            r["speech_start_frame"] = self.n
            self.last_start = self.n
            self.hit_max = False

        st = self.state
        if st == _State.SILENCE:
            if is_speech:
                self.state = _State.POSSIBLE_SPEECH
                self.speech_cnt += 1
            else:
                self.silence_cnt += 1
                self.speech_cnt = 0
        elif st == _State.POSSIBLE_SPEECH:
            if is_speech:
                self.speech_cnt += 1
                if self.speech_cnt >= self.min_speech:
                    self.state = _State.SPEECH
                    r["is_speech_start"] = True
                    r["speech_start_frame"] = max(
                        1, self.n - self.speech_cnt + 1 - self.pad,
                        self.last_end + 1)
                    self.last_start = r["speech_start_frame"]
                    self.silence_cnt = 0
            else:
                self.state = _State.SILENCE
                self.silence_cnt = 1
                self.speech_cnt = 0
        elif st == _State.SPEECH:
            self.speech_cnt += 1
            if is_speech:
                self.silence_cnt = 0
                if self.speech_cnt >= self.max_speech:
                    self.hit_max = True
                    r["hit_max_cut"] = True
                    self.speech_cnt = 0
                    r["is_speech_end"] = True
                    r["speech_end_frame"] = self.n
                    r["speech_start_frame"] = self.last_start
                    self.last_start = -1
                    self.last_end = self.n
            else:
                self.state = _State.POSSIBLE_SILENCE
                self.silence_cnt += 1
        elif st == _State.POSSIBLE_SILENCE:
            self.speech_cnt += 1
            if is_speech:
                self.state = _State.SPEECH
                self.silence_cnt = 0
                if self.speech_cnt >= self.max_speech:
                    self.hit_max = True
                    r["hit_max_cut"] = True
                    self.speech_cnt = 0
                    r["is_speech_end"] = True
                    r["speech_end_frame"] = self.n
                    r["speech_start_frame"] = self.last_start
                    self.last_start = -1
                    self.last_end = self.n
            else:
                self.silence_cnt += 1
                if self.silence_cnt >= self.min_silence:
                    self.state = _State.SILENCE
                    r["is_speech_end"] = True
                    r["speech_end_frame"] = self.n - self.silence_cnt
                    r["speech_start_frame"] = self.last_start
                    self.last_start = -1
                    self.last_end = r["speech_end_frame"]
                    self.speech_cnt = 0
                    self.silence_cnt = 0
        return r


class FireRedVadSegmenter:
    """FireRedVAD 流式分段（接口同 StreamingSegmenter）。"""

    def __init__(self, model_dir: str = "models/fireredvad-onnx",
                 speech_threshold: float = 0.5,
                 min_silence_ms: float = 700.0,
                 min_speech_ms: float = 80.0,
                 max_speech_s: float = 7.0,
                 pad_start_ms: float = 50.0,
                 buffer_s: float = 90.0,
                 ratio: bool = True):
        from pathlib import Path
        import onnxruntime as ort
        d = Path(model_dir)
        onnx_path = d / "stream_vad.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"缺少 {onnx_path}：先跑 scripts/export_fireredvad_onnx.py")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        cm = np.load(d / "cmvn.npz")
        # 兼容两种键名（导出脚本写的是 means/istd，官方字段名叫 inverse_std_variances）
        # 必须转成 float32：kaldiio 读出来的 cmvn 是 float64，减完会把 feats 抬成
        # double，ORT 会报 "Actual: (tensor(double)), expected: (tensor(float))"
        self.means = np.asarray(cm["means"], dtype=np.float32)
        _istd = cm["istd"] if "istd" in cm else cm["inverse_std_variances"]
        self.istd = np.asarray(_istd, dtype=np.float32)
        self.n_cache = int(meta["n_cache"])
        self.idim = int(meta["idim"])

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(onnx_path), so,
                                         providers=["CPUExecutionProvider"])
        self.in_names = [i.name for i in self.sess.get_inputs()]

        # 特征提取（kaldi fbank，C++ 扩展，不需要 torch）
        import kaldi_native_fbank as knf
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = SAMPLE_RATE
        opts.frame_opts.frame_length_ms = float(meta.get("frame_length_ms", 25.0))
        opts.frame_opts.frame_shift_ms = float(meta.get("frame_shift_ms", 10.0))
        opts.mel_opts.num_bins = self.idim
        self.fbank = knf.OnlineFbank(opts)
        self._frames_done = 0

        self.post = _PostProcessor(
            speech_threshold=speech_threshold,
            min_silence_frame=max(1, int(min_silence_ms / 10)),
            min_speech_frame=max(1, int(min_speech_ms / 10)),
            max_speech_frame=max(1, int(max_speech_s * 100)),
            pad_start_frame=max(1, int(pad_start_ms / 10)))
        self._sil = SileroVAD(threshold=0.6) if ratio else None

        self._caches = [np.zeros((1, 128, 0), dtype=np.float32)
                        for _ in range(self.n_cache)]
        self._buf = np.zeros(0, dtype=np.int16)
        self._buf_start = 0
        self.total_samples = 0
        self._cur_start: int | None = None
        self._seg_id = 0
        self._buf_max = int(buffer_s * SAMPLE_RATE)

    # ---------- 内部 ----------
    def _slice(self, a: int, b: int) -> np.ndarray:
        i = max(0, a - self._buf_start)
        j = max(i, b - self._buf_start)
        return self._buf[i:j]

    def _speech_ratio(self, pcm: np.ndarray) -> float:
        if self._sil is None or len(pcm) < FRAME_SIZE:
            return 1.0
        self._sil.reset()
        n = hit = 0
        for off in range(0, len(pcm) - FRAME_SIZE + 1, FRAME_SIZE):
            n += 1
            if self._sil.frame_prob(pcm[off:off + FRAME_SIZE]) >= self._sil.threshold:
                hit += 1
        return round(hit / n, 3) if n else 1.0

    # ---------- 对外 ----------
    def feed(self, pcm: np.ndarray) -> list[dict]:
        if pcm is None or len(pcm) == 0:
            return []
        pcm = np.asarray(pcm, dtype=np.int16)
        self._buf = np.concatenate([self._buf, pcm])
        self.total_samples += len(pcm)
        # 坑①：kaldi fbank 要 ±32768 量级，不能归一化到 ±1
        self.fbank.accept_waveform(SAMPLE_RATE, pcm.astype(np.float32).tolist())

        ready = self.fbank.num_frames_ready - self._frames_done
        out: list[dict] = []
        if ready > 0:
            feats = np.stack([np.asarray(self.fbank.get_frame(self._frames_done + i))
                              for i in range(ready)]).astype(np.float32)
            self._frames_done += ready
            self.fbank.pop(self._frames_done)
            feats = (feats - self.means) * self.istd
            feed = {"feat": feats[None, :, :]}
            for i, c in enumerate(self._caches):
                feed[f"cache{i}"] = c
            res = self.sess.run(None, feed)
            probs = np.asarray(res[0]).reshape(-1)
            self._caches = [np.asarray(res[1 + i]) for i in range(self.n_cache)]
            carry_start = None        # hit_max 智能切点：下一段从这里无缝续上
            for p in probs:
                r = self.post.process(float(p))
                if r["is_speech_start"]:
                    self._seg_id += 1
                    start = (r["speech_start_frame"] - 1) * FRAME_SHIFT
                    if carry_start is not None:
                        start = carry_start       # 上句被截断 → 从智能切点接着算
                        carry_start = None
                    self._cur_start = start
                if r["is_speech_end"] and self._cur_start is not None:
                    end = r["speech_end_frame"] * FRAME_SHIFT
                    cut = "silence"
                    if r.get("hit_max_cut"):
                        # 智能断句：不在上限处硬切，而是在最后 1.2s 里找
                        # 能量最低（最像停顿）的位置切，下一段从那里接上
                        end = self._refine_cut(end)
                        carry_start = end
                        cut = "max"
                    seg = self._finish(self._cur_start, end, cut)
                    if seg:
                        out.append(seg)
                    self._cur_start = None
        self._trim()
        return out

    def _refine_cut(self, end_abs: int, lookback_ms: float = 1500.0) -> int:
        """在 [end-lookback, end] 里找**一段安静区**的中间作为切点。

        "智能断句"：连续语音没有停顿时靠单句上限强切，但硬切很容易把词劈开；
        临近上限的最后一段里总有一个相对最安静的瞬间（音节间隙），在那里切才自然。

        2026-09-16 改进：原来是取"最安静的**单个 10ms 窗**"，而元音内部的
        瞬时过零也可能很安静 → 切在词中间，whisper 会转出 `show ro-` 这种断词残尾。
        现在先对 10ms 能量做 50ms 滑动平均，再取最小值 —— 只有**连续 50ms
        都安静**的位置才算候选，基本只会落在词与词之间。
        """
        a = max(self._buf_start, end_abs - int(lookback_ms * FRAME_SHIFT / 10))
        piece = self._slice(a, end_abs)
        win = 160                                   # 10ms
        if len(piece) < win * 7:                    # 至少 70ms 才有滑动平均的余地
            return end_abs
        x = piece.astype(np.float32)
        n = (len(x) - win) // win + 1
        e = np.empty(n, dtype=np.float32)
        for i in range(n):
            seg = x[i * win:(i + 1) * win]
            e[i] = float(np.mean(seg * seg))
        pad = 2                                     # ±2 个窗 = 50ms 跨度
        best_i, best_v = None, None
        for i in range(n):
            lo, hi = max(0, i - pad), min(n, i + pad + 1)
            v = float(np.mean(e[lo:hi]))
            if best_v is None or v < best_v:
                best_v, best_i = v, i
        if best_i is None:
            return end_abs
        return a + (best_i + 1) * win

    def _finish(self, start_abs: int, end_abs: int, cut: str = "silence") -> dict | None:
        piece = self._slice(max(0, start_abs - FRAME_SHIFT), end_abs + FRAME_SHIFT)
        if len(piece) < FRAME_SIZE:
            return None
        # cut：这一段是**为什么**结束的 —— live.py 靠它决定要不要把下一段
        # 续接到同一行（"max" = 撞单句上限被强切，后半截还在后面）。
        return {"pcm": piece.copy(), "start_sample": start_abs,
                "end_sample": end_abs, "max_prob": 1.0, "cut": cut,
                "speech_ratio": self._speech_ratio(piece),
                "seg_id": self._seg_id}

    def _trim(self):
        keep = self.total_samples - self._buf_max
        if self._cur_start is not None:
            keep = min(keep, self._cur_start - FRAME_SHIFT)
        if keep > self._buf_start:
            cut = int(keep - self._buf_start)
            self._buf = self._buf[cut:]
            self._buf_start += cut

    def flush(self) -> dict | None:
        """音频断流时立刻定稿当前句（不等句尾静音）。

        关键修复：以前音频一断（视频暂停 / CG 结束 / 播放器断流），
        这一句就一直挂着，直到下一段音频进来才收尾 —— 表现为
        "最后一句不显示，后面一有声音就马上跳出来"，字幕很不连贯。
        """
        if self._cur_start is None:
            return None
        end = self._trim_tail_silence(self._cur_start, self.total_samples)
        seg = self._finish(self._cur_start, end, "flush")
        self._cur_start = None
        self.post.reset_state()    # 只清状态，保留帧号（见 reset_state 的注释）
        return seg

    def _trim_tail_silence(self, start_abs: int, end_abs: int,
                           max_ms: float = 1200.0) -> int:
        """从尾部往回找最后一个"有声"的 10ms 窗，切掉后面的静音（留 120ms 余量）。"""
        a = max(start_abs, end_abs - int(max_ms * FRAME_SHIFT / 10))
        piece = self._slice(a, end_abs)
        if len(piece) < 320:
            return end_abs
        x = piece.astype(np.float32)
        win = FRAME_SHIFT          # 10ms
        peak = float(np.sqrt(np.mean(x ** 2))) or 1.0
        thr = peak * 0.08
        last = None
        for off in range(0, len(x) - win + 1, win):
            if float(np.sqrt(np.mean(x[off:off + win] ** 2))) >= thr:
                last = off
        if last is None:
            return end_abs
        return min(end_abs, a + last + win + int(120 * FRAME_SHIFT / 10))

    def pending(self) -> dict | None:
        if self._cur_start is None:
            return None
        piece = self._slice(self._cur_start, self.total_samples)
        return {"pcm": piece, "seg_id": self._seg_id} if len(piece) >= FRAME_SIZE else None

    def reset(self):
        import kaldi_native_fbank as knf
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = SAMPLE_RATE
        opts.frame_opts.frame_length_ms = 25.0
        opts.frame_opts.frame_shift_ms = 10.0
        opts.mel_opts.num_bins = self.idim
        self.fbank = knf.OnlineFbank(opts)
        self._frames_done = 0
        self.post.reset()
        self._caches = [np.zeros((1, 128, 0), dtype=np.float32)
                        for _ in range(self.n_cache)]
        self._buf = np.zeros(0, dtype=np.int16)
        self._buf_start = self.total_samples
        self._cur_start = None
