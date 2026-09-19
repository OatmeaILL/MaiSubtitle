# -*- coding: utf-8 -*-
"""FSMN-VAD 流式分段器（可选 VAD 引擎，CPU/ONNX）。

为什么单独做这个：
    现役 Silero 分段在实测里**切得过碎**（60s 音频 16 段、平均 1.44s），
    而 FSMN-VAD（damo/speech_fsmn_vad_zh-cn-16k-common-onnx，int8 ONNX）
    同样区间切出 9 段、平均 3.32s——更接近"一句人话"的长度，CPU 开销相当
    （RTF 0.0041 vs Silero 0.0039，单块最大约 2ms）。

接口与 `vad.StreamingSegmenter` 完全一致（可直接替换）：
    feed(pcm_int16) -> [{"pcm", "start_sample", "end_sample", "max_prob",
                         "speech_ratio", "seg_id"}, ...]
    pending() -> {"pcm", "seg_id"} | None      # 正在说的部分（给部分识别用）
    total_samples / reset()

两个实现细节（都踩过）：
 1. **状态必须由调用方持有**：`Fsmn_vad_online(chunk, param_dict=...)` 会把
    frontend/in_cache/vad_scorer 写回传入的 dict；首块也要传，否则状态丢失 → 0 段。
 2. **流式协议是"起点/终点分两次给"**：`[[1980, -1]]`=开始，`[[-1, 9290]]`=结束，
    单位毫秒，相对整条流起点。两者配对才是一个完整段。

speech_ratio（音乐门要用）：FSMN 只给段边界、不给逐帧概率，
所以对**已完成的段**跑一遍 Silero 帧概率来补这个比例（只在段结束时算，
一段约 3s 音频 ≈ 十几毫秒 CPU，可忽略）。
"""
from __future__ import annotations

import sys
import time

import numpy as np

from .vad import FRAME_SIZE, SAMPLE_RATE, SileroVAD

_last_feed_err = [0.0]    # 单块识别失败的限频报告（喂音频是持续流，别刷屏）

# 毫秒 → 样本（16k 采样率下 1ms = 16 样本）
_MS2S = SAMPLE_RATE // 1000


def _flatten(res) -> list[tuple[int, int]]:
    """把 funasr 返回的嵌套结构摊平成 [(beg_ms, end_ms), ...]（-1 表示该侧未定）。"""
    out: list[tuple[int, int]] = []
    for item in res or []:
        if isinstance(item, (list, tuple, np.ndarray)):
            vals = list(item)
            if len(vals) == 2 and all(isinstance(v, (int, float, np.integer, np.floating))
                                      for v in vals):
                out.append((int(vals[0]), int(vals[1])))
            else:
                out.extend(_flatten(vals))
    return out


class FsmnSegmenter:
    """用 FSMN-VAD 做实时分句（接口对齐 StreamingSegmenter）。"""

    def __init__(self, model_dir: str = "models/fsmn-vad-onnx",
                 max_segment_ms: float = 12000.0,
                 pad_start_ms: float = 100.0,
                 pad_end_ms: float = 150.0,
                 buffer_s: float = 90.0,
                 ratio: bool = True,
                 ratio_threshold: float = 0.6):
        try:
            from funasr_onnx import Fsmn_vad_online
        except Exception as e:      # pragma: no cover - 依赖缺失时由调用方回退
            raise ImportError(
                "FSMN-VAD 需要 funasr-onnx：pip install funasr-onnx "
                f"（{type(e).__name__}: {e}）") from e
        self._model = Fsmn_vad_online(model_dir, quantize=True, device_id="-1",
                                      intra_op_num_threads=1)
        self._params: dict = {}          # 流式状态容器（必须全程复用同一个）
        self._sil = SileroVAD(threshold=ratio_threshold) if ratio else None
        self.max_samples = int(max_segment_ms / 1000 * SAMPLE_RATE)
        self.pad_start = int(pad_start_ms / 1000 * SAMPLE_RATE)
        self.pad_end = int(pad_end_ms / 1000 * SAMPLE_RATE)
        self._buf_max = int(buffer_s * SAMPLE_RATE)

        self._buf = np.zeros(0, dtype=np.int16)
        self._buf_start = 0              # _buf[0] 对应的绝对样本号
        self.total_samples = 0           # 累计喂入样本数（= 流的绝对长度）
        self._cur_start: int | None = None   # 正在说的这一段的起点（绝对样本）
        self._seg_id = 0

    # ---------- 内部工具 ----------
    def _slice(self, a: int, b: int) -> np.ndarray:
        """取绝对样本区间 [a, b) 的 PCM（超出缓冲区部分自动截断）。"""
        i = max(0, a - self._buf_start)
        j = max(i, b - self._buf_start)
        return self._buf[i:j]

    def _speech_ratio(self, pcm: np.ndarray) -> float:
        """段内"有声帧"占比（补 FSMN 不提供逐帧概率的缺口）。"""
        if self._sil is None or len(pcm) < FRAME_SIZE:
            return 1.0
        self._sil.reset()
        n = hit = 0
        for off in range(0, len(pcm) - FRAME_SIZE + 1, FRAME_SIZE):
            n += 1
            if self._sil.frame_prob(pcm[off:off + FRAME_SIZE]) >= self._sil.threshold:
                hit += 1
        return round(hit / n, 3) if n else 1.0

    # ---------- 对外接口 ----------
    def feed(self, pcm: np.ndarray) -> list[dict]:
        if pcm is None or len(pcm) == 0:
            return []
        pcm = np.asarray(pcm, dtype=np.int16)
        self._buf = np.concatenate([self._buf, pcm])
        self.total_samples += len(pcm)

        try:
            res = self._model((pcm.astype(np.float32) / 32768.0),
                              param_dict=self._params)
        except Exception as e:
            # 单块出错不应打断采集循环 —— 但也不能零留证：模型目录坏/输入形状不符时
            # 表现就是"永远不切句、字幕全无"而日志一个字都没有（假成功类坑）。
            now = time.time()
            if now - _last_feed_err[0] > 30.0:
                _last_feed_err[0] = now
                print(f"[vad-fsmn] 单块识别失败（30s 内只报这一次，采集继续）："
                      f"{type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
            return []                     # 单块出错不应打断采集循环

        out: list[dict] = []
        for beg_ms, end_ms in _flatten(res):
            if beg_ms >= 0:
                self._seg_id += 1
                self._cur_start = beg_ms * _MS2S
            if end_ms >= 0 and self._cur_start is not None:
                seg = self._finish(self._cur_start, end_ms * _MS2S)
                if seg:
                    out.append(seg)
                self._cur_start = None
        self._trim()
        return out

    def _finish(self, start_abs: int, end_abs: int, cut: str = "silence") -> dict | None:
        a = max(0, start_abs - self.pad_start)
        b = end_abs + self.pad_end
        piece = self._slice(a, b)
        if len(piece) < FRAME_SIZE:
            return None
        return {"pcm": piece.copy(),
                "start_sample": start_abs,
                "end_sample": end_abs,
                "max_prob": 1.0,
                "cut": cut,
                "speech_ratio": self._speech_ratio(piece),
                "seg_id": self._seg_id}

    def _trim(self):
        """控制缓冲：保留当前句起点之后的内容，最多 _buf_max 样本。"""
        keep_from = self.total_samples - self._buf_max
        if self._cur_start is not None:
            keep_from = min(keep_from, self._cur_start - self.pad_start)
        if keep_from > self._buf_start:
            cut = int(keep_from - self._buf_start)
            self._buf = self._buf[cut:]
            self._buf_start += cut

    def pending(self) -> dict | None:
        """正在说的部分（给"边说边识别"用）。"""
        if self._cur_start is None:
            return None
        piece = self._slice(self._cur_start, self.total_samples)
        if len(piece) < FRAME_SIZE:
            return None
        return {"pcm": piece, "seg_id": self._seg_id}

    def flush(self) -> dict | None:
        """音频断流时立刻定稿当前句（不等句尾静音）——否则最后一句要等下一段
        音频才蹦出来，字幕不连贯。"""
        if self._cur_start is None:
            return None
        seg = self._finish(self._cur_start, self.total_samples, "flush")
        self._cur_start = None
        return seg

    def reset(self):
        """换音频设备/恢复采集后调用（FSMN 的流式状态不能跨流复用）。"""
        self._params = {}
        self._buf = np.zeros(0, dtype=np.int16)
        self._buf_start = self.total_samples
        self._cur_start = None
