"""音频：WASAPI loopback 捕获 + 媒体文件解码（统一 16kHz 单声道 int16）。"""
import threading
from pathlib import Path

import numpy as np

TARGET_SR = 16000

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    import pyaudio


def to_16k_mono(samples: np.ndarray, src_sr: int, src_channels: int) -> np.ndarray:
    """设备音频 → 16kHz 单声道 int16。输入为 int16 原始 PCM。"""
    import soxr
    x = samples.astype(np.float32) / 32768.0
    if src_channels > 1:
        x = x.reshape(-1, src_channels).mean(axis=1)
    if src_sr != TARGET_SR:
        x = soxr.resample(x, src_sr, TARGET_SR, quality="HQ")
    return np.clip(x * 32767.0, -32768, 32767).astype(np.int16)


def get_default_loopback(pa) -> dict:
    try:
        return pa.get_default_wasapi_loopback()
    except AttributeError:
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_out = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get("isLoopbackDevice") and default_out["name"].split(" (")[0] in info["name"]:
                return info
        raise RuntimeError("找不到默认输出设备对应的 loopback 设备")


class LoopbackCapture:
    """回调式 WASAPI loopback 捕获。start()/stop() 返回 16k mono int16。"""

    def __init__(self, device_index: int | None = None):
        self.pa = pyaudio.PyAudio()
        self.device_index = device_index
        self._frames = []
        self._lock = threading.Lock()
        self._n_samples_device = 0
        self._t_start = None
        self._stream = None
        self._overflow_count = 0
        self._device_info = None

    def start(self):
        info = (self.pa.get_device_info_by_index(self.device_index)
                if self.device_index is not None else get_default_loopback(self.pa))
        self._device_info = info
        channels = min(info["maxInputChannels"], 2)
        sr = int(info["defaultSampleRate"])
        self._t_start = time.perf_counter()

        def cb(in_data, frame_count, time_info, status):
            if status:
                self._overflow_count += 1
            with self._lock:
                self._frames.append(np.frombuffer(in_data, dtype=np.int16).copy())
                self._n_samples_device += frame_count
            return (None, pyaudio.paContinue)

        self._stream = self.pa.open(
            format=pyaudio.paInt16, channels=channels, rate=sr,
            input=True, input_device_index=info["index"],
            frames_per_buffer=1024, stream_callback=cb)
        self._stream.start_stream()
        return self

    def stop(self):
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        import time as _t
        wall = _t.perf_counter() - self._t_start
        with self._lock:
            raw = (np.concatenate(self._frames) if self._frames
                   else np.zeros(0, dtype=np.int16))
            self._frames = []
        sr_dev = int(self._device_info["defaultSampleRate"])
        ch = min(self._device_info["maxInputChannels"], 2)
        audio16k = to_16k_mono(raw, sr_dev, ch)
        meta = {"device_name": self._device_info["name"],
                "device_rate": sr_dev, "channels": ch,
                "wall_seconds": wall, "overflows": self._overflow_count}
        return audio16k, meta

    def terminate(self):
        self.pa.terminate()

    @property
    def device_name(self) -> str:
        return self._device_info["name"]

    def drain(self) -> np.ndarray:
        """取出并清空自上次调用以来累积的 16k 单声道样本（音频流保持打开）。

        实时管线原来每 0.25s 就 stop()+新建 start() 一次，PyAudio/WASAPI 反复
        初始化实测约吃掉 46% 单核；改成持续打开 + 定时 drain 后降到个位数。
        """
        with self._lock:
            frames = self._frames
            self._frames = []
        if not frames:
            return np.zeros(0, dtype=np.int16)
        raw = np.concatenate(frames)
        sr_dev = int(self._device_info["defaultSampleRate"])
        ch = min(self._device_info["maxInputChannels"], 2)
        return to_16k_mono(raw, sr_dev, ch)


import time  # noqa: E402


def decode_media(path: str | Path) -> tuple[np.ndarray, float]:
    """任意音频/视频文件 → (16k mono int16, 时长秒)。基于 PyAV，无需 ffmpeg。"""
    import av
    with av.open(str(path)) as container:
        streams = [s for s in container.streams if s.type == "audio"]
        if not streams:
            raise ValueError(f"文件中没有音频流: {path}")
        stream = streams[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=TARGET_SR)
        chunks = []
        for frame in container.decode(stream):
            for rf in resampler.resample(frame):
                chunks.append(rf.to_ndarray().reshape(-1))
    if not chunks:
        raise ValueError(f"音频解码为空: {path}")
    pcm = np.concatenate(chunks)
    return pcm.astype(np.int16), len(pcm) / TARGET_SR
