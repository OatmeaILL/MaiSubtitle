# -*- coding: utf-8 -*-
"""GPU 子进程客户端：把识别/翻译挪出主进程，原生卡死不再拖死界面。

背景（HANDOVER 七节，两条实锤）：
  A. CTranslate2 的 CUDA 调用偶发**不返回**（faulthandler 连续 3 份转储 45s 都停在
     faster_whisper/transcribe.py:1400 in encode），主线程被阻塞 → Windows 判"未响应"；
  B. 原生 access violation（崩栈里 Python 线程全空闲）→ 进程直接消失。
原生调用**无法从 Python 层中断**，所以以前只能靠进程外守护（supervisor.py）杀掉
整个程序重启。用户看到的是"字卡住 → 整个程序被重启"。

子进程化之后：
  * 模型活在工作进程里，主进程只做采集/VAD/UI —— 主线程再不会被原生调用冻住；
  * 请求带超时：超时/崩溃 = 卡死 → **只杀工作进程**并重新拉起（模型重载），
    悬浮窗、VAD 状态、会话上下文全部保留，字幕在重载完成后继续。

为什么不用 multiprocessing：Windows 只有 spawn，子进程会导入父进程的 `__main__`，
而 scripts/live_demo.py 是脚本式的（模块级就建 Qt/管线）→ 子进程会把整个程序
再跑一遍并撞上单实例守卫直接退出（实测：子进程秒退、请求全超时）。因此改用
`python -m maisubtitle.gpu_child` 独立进程 + stdin/stdout JSON 行协议
（协议细节见 gpu_child.py 顶部）。

对外接口与本地对象完全一致（RemoteASR / RemoteMT），live.py 只换构造那一行。
"""
from __future__ import annotations

import base64
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CFG_DUMP = PROJECT_ROOT / "logs" / "gpu_child_cfg.json"

DEFAULT_TIMEOUT = 30.0          # 单次识别/翻译的容忍上限（正常 0.3~1.5s）
PRELOAD_TIMEOUT = 600.0         # 模型加载（含 ORT CUDA 会话）可能很慢
STREAM_TIMEOUT = 30.0           # 流式翻译：两个 token 之间的容忍上限


class GpuTimeout(RuntimeError):
    """子进程没在超时内应答 —— 判为 GPU 原生卡死，已触发重启。"""


class GpuDead(RuntimeError):
    """子进程已退出或正在重启（模型重载中）。"""


def _pack(audio):
    """int16 音频 → JSON 可传的 {b64, shape}。"""
    return {"b64": base64.b64encode(audio.tobytes()).decode("ascii"),
            "shape": list(audio.shape)}


class GpuWorker:
    """工作子进程的客户端：请求串行化 + 超时判死 + 自动重启。"""

    def __init__(self, cfg, on_state=None, preload_on_restart: bool = True):
        self.cfg = cfg
        self.on_state = on_state or (lambda s: None)
        self._preload_on_restart = bool(preload_on_restart)
        self._lock = threading.Lock()
        self._down = threading.Event()
        self._n = 0
        self._proc = None
        self._q: dict = {}                  # id → Queue（该请求的响应/流块）
        self._q_lock = threading.Lock()
        self._reader = None

    # ---- 生命周期 ----
    def start(self):
        CFG_DUMP.parent.mkdir(parents=True, exist_ok=True)
        self._kill_orphans()
        try:
            self.cfg.save(CFG_DUMP)
        except Exception:
            # save() 不可用时退回 dataclass 序列化（子进程用 AppConfig.load 读）
            import dataclasses
            CFG_DUMP.write_text(json.dumps(dataclasses.asdict(self.cfg),
                                           ensure_ascii=False, indent=2),
                                encoding="utf-8")
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "maisubtitle.gpu_child", str(CFG_DUMP)],
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1)
        self._reader = threading.Thread(target=self._read_loop, daemon=True,
                                        name="gpu-child-reader")
        self._reader.start()
        self._start_vram_watch()

    def _kill_orphans(self):
        """清掉上次运行遗留的孤儿 gpu_child（父进程已死却还占着显存）。

        实锤场景（用户报告"跑久了显存越来越大"）：主进程原生崩溃/被杀时，子进程
        若正卡在 CUDA 调用里，读不到 stdin EOF → 变孤儿，显存永不释放；守护进程
        再拉起新实例 → 越积越多。这里在启动新子进程前把无主的旧子进程清掉。
        """
        try:
            import psutil
        except Exception:
            return
        me = os.getpid()
        for p in psutil.process_iter(["pid", "ppid", "cmdline"]):
            try:
                cl = p.info.get("cmdline") or []
                if not any("maisubtitle.gpu_child" in (c or "") for c in cl):
                    continue
                ppid = p.info.get("ppid") or 0
                if ppid == me or (ppid and psutil.pid_exists(ppid)):
                    continue                      # 父进程还在（可能是另一实例）：别动
                p.kill()
                self.on_state(f"清掉遗留的 GPU 子进程 pid={p.info.get('pid')}"
                              "（父进程已退出，显存已释放）")
            except Exception:
                continue

    def _start_vram_watch(self):
        """每 5 分钟记一行显存占用日志（排查"跑久了显存变大"的证据）。

        用 pynvml（不可用就静默跳过）；只记录不动手 —— 该值班日志留给下次排查。
        """
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")     # pynvml 的 deprecation 噪音
                import pynvml
            pynvml.nvmlInit()
        except Exception:
            return

        def _watch():
            while True:
                time.sleep(300.0)
                try:
                    h = pynvml.nvmlDeviceGetHandleByIndex(0)
                    info = pynvml.nvmlDeviceGetMemoryInfo(h)
                    self.on_state(f"vram: 全卡 {info.used // 1048576} MiB / "
                                  f"{info.total // 1048576} MiB")
                except Exception:
                    pass

        threading.Thread(target=_watch, daemon=True, name="vram-watch").start()

    def stop(self):
        try:
            if self._proc and self._proc.stdin:
                self._proc.stdin.close()        # 子进程 for line in stdin 结束 → 退出
        except Exception:
            pass
        try:
            if self._proc:
                self._proc.wait(timeout=5)
        except Exception:
            pass
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.kill()
        except Exception:
            pass

    def preload(self, timeout=PRELOAD_TIMEOUT):
        """加载模型（阻塞到就绪）。失败不致命：下一次请求还会重试。"""
        try:
            self.call("preload", timeout=timeout)
            return True
        except Exception as e:
            self.on_state(f"warn: GPU 子进程预热失败 {type(e).__name__}: {str(e)[:80]}")
            return False

    @property
    def alive(self) -> bool:
        return bool(self._proc and self._proc.poll() is None)

    # ---- 读线程：把子进程的每一行分发到对应请求的队列 ----
    def _read_loop(self):
        proc = self._proc
        while True:
            try:
                line = proc.stdout.readline()
            except Exception:
                break
            if not line:
                break                       # 子进程退出/管道关闭
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except Exception:
                continue
            if "state" in msg:
                try:
                    self.on_state(str(msg["state"]))
                except Exception:
                    pass
                continue
            rid = msg.get("id")
            with self._q_lock:
                q = self._q.get(rid)
            if q is None:                   # 迟到的响应（请求已超时判死）→ 丢
                continue
            q.put(msg)

    def _slot(self, rid: int) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._q_lock:
            self._q[rid] = q
            if len(self._q) > 64:           # 防御：清掉旧槽位
                for k in list(self._q)[:-32]:
                    self._q.pop(k, None)
        return q

    def _next(self) -> int:
        self._n += 1
        return self._n

    def _wait_up(self, timeout: float):
        """_down 置位 = 正在重启：等它清掉再发请求。

        Event 没有"等清零"，只能轮询 —— 以前写成 `self._down.wait(...)`，而进入
        这里时事件**已置位**、wait() 立即返回 True，整段逻辑从来没生效过。
        """
        if not self._down.is_set():
            return
        deadline = time.monotonic() + max(1.0, min(timeout, 120.0))
        while self._down.is_set():
            if time.monotonic() >= deadline:
                raise GpuDead("GPU 子进程正在重启（等待重启完成超时）")
            time.sleep(0.05)

    def _send(self, rid: int, method: str, args: dict):
        line = json.dumps({"id": rid, "method": method, "args": args},
                          ensure_ascii=True) + "\n"
        try:
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
        except Exception as e:
            raise GpuDead(f"GPU 子进程管道不可用: {str(e)[:60]}")

    # ---- 请求 ----
    def call(self, method: str, timeout: float = DEFAULT_TIMEOUT, **args):
        rid = self._next()
        if not self._lock.acquire(timeout=max(1.0, timeout)):
            # _restart 全程持锁（含模型重载，可达几十秒）：等锁超时多半是在等重启，
            # 报成"GPU 空闲超时"会误导排查方向（HANDOVER §七十六 的教训：报错要真）
            why = ("等待 GPU 子进程重启完成超时（模型重载中）" if self._down.is_set()
                   else "等待 GPU 空闲超时")
            raise GpuTimeout(f"{method} {why}")
        try:
            self._wait_up(timeout)
            if not self.alive:
                self._fail(method, "子进程已退出（原生崩溃？）")
                raise GpuDead("GPU 子进程已退出")
            q = self._slot(rid)
            self._send(rid, method, args)
            return self._await(q, rid, method, timeout)
        finally:
            self._lock.release()

    def stream(self, method: str, timeout: float = STREAM_TIMEOUT, **args):
        """流式方法（逐块 yield）。结束后收掉最终响应（出错在这里抛）。"""
        rid = self._next()
        if not self._lock.acquire(timeout=max(1.0, timeout)):
            # _restart 全程持锁（含模型重载，可达几十秒）：等锁超时多半是在等重启，
            # 报成"GPU 空闲超时"会误导排查方向（HANDOVER §七十六 的教训：报错要真）
            why = ("等待 GPU 子进程重启完成超时（模型重载中）" if self._down.is_set()
                   else "等待 GPU 空闲超时")
            raise GpuTimeout(f"{method} {why}")
        try:
            self._wait_up(timeout)
            if not self.alive:
                self._fail(method, "子进程已退出（原生崩溃？）")
                raise GpuDead("GPU 子进程已退出")
            q = self._slot(rid)
            self._send(rid, method, args)
            end = time.perf_counter() + timeout
            while True:
                left = end - time.perf_counter()
                if left <= 0:
                    self._fail(method, "流式无响应")
                    raise GpuTimeout(f"{method} 流式超时")
                try:
                    msg = q.get(timeout=left)
                except queue.Empty:
                    self._fail(method, "流式无响应")
                    raise GpuTimeout(f"{method} 流式超时")
                if msg.get("done"):
                    break
                if "chunk" in msg:
                    end = time.perf_counter() + timeout    # 每块都续期
                    yield msg.get("chunk") or ""
                    continue
                if not msg.get("ok"):
                    # 子进程报错（例如引擎不支持流式）必须**立刻**抛出：以前这里只认
                    # chunk/done，错误回包被当空气 → 白等 30s 超时，还会被当成"GPU 卡死"
                    # 去重启子进程，真因被彻底盖掉（2026-09-18 实锤：Hy-MT2 整句译文丢失）。
                    raise RuntimeError(str(msg.get("error") or "子进程报错"))
            self._await(q, rid, method, timeout)
        finally:
            self._lock.release()

    def _await(self, q: queue.Queue, rid: int, method: str, timeout: float):
        end = time.perf_counter() + timeout
        while True:
            left = end - time.perf_counter()
            if left <= 0:
                self._fail(method, "无响应（原生卡死？）")
                raise GpuTimeout(f"{method} 超时 {timeout:.0f}s")
            try:
                msg = q.get(timeout=left)
            except queue.Empty:
                self._fail(method, "无响应（原生卡死？）")
                raise GpuTimeout(f"{method} 超时 {timeout:.0f}s")
            if "chunk" in msg or "done" in msg:
                continue                     # 流式残块（非流式请求不该有）
            if not msg.get("ok"):
                raise RuntimeError(str(msg.get("error") or "子进程报错"))
            return msg.get("result")

    # ---- 卡死/崩溃处理 ----
    def _fail(self, method: str, why: str):
        if self._down.is_set():
            return
        self._down.set()
        self.on_state(f"warn: GPU 子进程 {method} {why} → 只重启它（模型重载，字幕稍后继续）")
        threading.Thread(target=self._restart, daemon=True, name="gpu-restart").start()

    def _restart(self):
        try:
            with self._lock:                 # 等当前请求退场
                try:
                    if self._proc and self._proc.poll() is None:
                        self._proc.kill()
                        self._proc.wait(timeout=5)
                except Exception:
                    pass
                with self._q_lock:
                    self._q.clear()
                self.start()
                if self._preload_on_restart:
                    self.preload()           # 重新加载模型
                self.on_state("GPU 子进程已重启，字幕恢复")
        except Exception as e:
            self.on_state(f"warn: GPU 子进程重启失败 {type(e).__name__}: {str(e)[:80]}")
        finally:
            self._down.clear()


class RemoteASR:
    """与本地识别后端同接口的代理（模型在子进程里）。"""

    def __init__(self, worker: GpuWorker, name: str = "gpu"):
        self._w = worker
        self.url = ""                            # live.py 用它打印后端地址
        self.name = name

    def transcribe(self, audio, language=None, prompt=None,
                   return_confidence=False, beam_size=5, without_timestamps=False,
                   no_fallback=False):
        # without_timestamps 必须透传（wots）：字幕时间轴来自 VAD，模型时间戳纯浪费
        # 解码步数 —— 以前收下即弃，asr.py 宣称的提速在子进程模式下从未生效（§八十）
        res = self._w.call("transcribe", timeout=DEFAULT_TIMEOUT,
                           audio=_pack(audio), language=language,
                           prompt=prompt, conf=return_confidence, beam=beam_size,
                           wots=bool(without_timestamps),
                           nfb=bool(no_fallback))
        return res[0], dict(res[1] or {})

    def detect_lang(self, audio):
        res = self._w.call("detect", timeout=DEFAULT_TIMEOUT, audio=_pack(audio))
        return res[0], float(res[1])


class RemoteMT:
    """与本地翻译引擎同接口的代理（语义见 translate.QwenCT2）。"""

    def __init__(self, worker: GpuWorker, engine: str = "gpu"):
        self._w = worker
        self.engine = engine
        self._terms: list[tuple[str, str]] = []
        self.load_s = 0.0

    def set_terms(self, terms):
        """术语随请求带上（子进程对术语是无状态的，避免两边状态不同步）。"""
        self._terms = list(terms or [])

    def translate(self, text: str, src_lang: str, context=None):
        out = self._w.call("translate", timeout=DEFAULT_TIMEOUT, text=text,
                           lang=src_lang, context=list(context or []),
                           terms=self._terms)
        # 子进程里的 `mt.translate()` 返回 (译文, meta) 元组，过 JSON 变成
        # **[译文, meta] 列表**；这里必须解包。
        # 悬案实锤（error.log 里的 "unhashable type: 'dict'"）：早期直接把整个列表
        # 当译文返回，调用方 `zh, _ = mt.translate(...)` 拿到的是列表，走到
        # `to_simplified(list)` 就抛错 —— 而且抛错点在"流式没出中文 → 整段重试"的
        # 兜底分支上，**整条字幕会被丢掉**（显示层只出原文的机会都没有）。
        if isinstance(out, (list, tuple)) and len(out) == 2 and isinstance(out[0], str):
            return out[0], dict(out[1] or {})
        if out is None:
            return "", {"engine": self.engine}
        return str(out), {"engine": self.engine}

    def translate_stream(self, text: str, src_lang: str, context=None,
                         max_tokens: int = 160):
        yield from self._w.stream("translate_stream", timeout=STREAM_TIMEOUT,
                                  text=text, lang=src_lang,
                                  context=list(context or []),
                                  terms=self._terms)

    def chat(self, messages):
        return self._w.call("chat", timeout=DEFAULT_TIMEOUT, messages=messages)