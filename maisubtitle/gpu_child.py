# -*- coding: utf-8 -*-
"""GPU 工作进程的入口（由 maisubtitle.gpu_proc 用 `python -m` 拉起）。

为什么不用 multiprocessing
--------------------------
Windows 上 multiprocessing 只有 spawn：子进程会**导入父进程的 __main__**。
本项目的实时入口 `scripts/live_demo.py` 是"脚本式"的（模块级就创建 Qt/管线、
没有 `if __name__ == "__main__"` 保护），spawn 子进程会把整个程序再跑一遍，
撞上单实例守卫后直接退出 —— 实测表现为"GPU 子进程秒退 + 请求全部超时"。
所以这里用一个**独立模块 + stdin/stdout 协议**代替 multiprocessing：
父进程 `python -m maisubtitle.gpu_child <cfg.json>`，之后全程走 JSON 行协议。

协议（每行一个 JSON，UTF-8，non-ASCII 转义）
-------------------------------------------
父 → 子：{"id": n, "method": "transcribe|detect|translate|translate_stream|chat|
          preload|load_s", "args": {...}}
子 → 父：{"id": n, "state": "..."}            进度/日志（可插在任意请求之间）
         {"id": n, "chunk": "..."}            流式翻译的中间结果
         {"id": n, "ok": true, "result": ...} 请求完成
         {"id": n, "ok": false, "error": "..."} 请求失败

音频用 base64 传（int16 原始字节 + shape），4s 音频约 170KB，对管道毫无压力。
"""
import base64
import json
import os
import sys
import threading
import time
import traceback


def _parent_alive(ppid: int) -> bool:
    """父进程是否还活着（Windows：拿父进程句柄看有没有收到退出信号）。"""
    import ctypes
    SYNCHRONIZE = 0x00100000
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(SYNCHRONIZE, False, ppid)
    if not h:
        return False
    try:
        return k32.WaitForSingleObject(h, 0) != 0     # 非 0 = 未 signal = 还活着
    finally:
        k32.CloseHandle(h)


def _start_parent_watchdog():
    """父进程一死就硬退（os._exit）——防"孤儿子进程占着显存不还"。

    实锤场景（用户报告"跑久了显存越来越大"）：主进程原生崩溃/被杀时，本进程若
    正卡在 CUDA 调用里，读不到 stdin EOF，就成了孤儿；守护进程再拉起新实例 →
    显存越积越多。os._exit 从任意线程都能立刻终止进程（不依赖 Python 清理）。
    """
    try:
        import ctypes
        if not hasattr(ctypes, "windll"):
            return
    except Exception:
        return
    ppid = os.getppid()
    if not ppid:
        return

    def _watch():
        while True:
            time.sleep(3.0)
            try:
                if not _parent_alive(ppid):
                    os._exit(3)
            except Exception:
                pass

    threading.Thread(target=_watch, daemon=True, name="parent-watchdog").start()


def _pack(audio):
    return {"b64": base64.b64encode(audio.tobytes()).decode("ascii"),
            "shape": list(audio.shape)}


def _unpack(d):
    import numpy as np
    buf = base64.b64decode(d["b64"])
    return np.frombuffer(buf, dtype=np.int16).reshape(tuple(d["shape"]))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("用法: python -m maisubtitle.gpu_child <config.json>", file=sys.stderr)
        return 2
    cfg_path = argv[0]

    # 先把协议通道抢下来：之后子进程自己的 print 全部改道 stderr，
    # 免得混进 JSON 行把父进程的解析搞坏。
    proto = sys.stdout
    sys.stdout = sys.stderr

    from pathlib import Path

    from .config import AppConfig, register_nvidia_dlls
    register_nvidia_dlls()          # ORT CUDA 会话前必须（DLL 句柄要持有）
    cfg = AppConfig.load(Path(cfg_path))     # load() 要 Path，不吃 str
    st = {"asr": None, "mt": None}
    _start_parent_watchdog()                 # 父进程死了就退出，别当孤儿占显存

    def send(msg):
        proto.write(json.dumps(msg) + "\n")
        proto.flush()

    def say(text):
        send({"id": 0, "state": str(text)})

    def ensure_asr():
        if st["asr"] is None:
            from .asr import make_asr
            t0 = time.perf_counter()
            st["asr"] = make_asr(cfg.asr_backend, cfg.asr_model, cfg.asr_device,
                                 cfg.asr_compute_type,
                                 str(getattr(cfg, "asr_http_url", "") or ""),
                                 str(getattr(cfg, "asr_http_model", "") or ""),
                                 qwen_dir=str(getattr(cfg, "asr_qwen_dir", "") or ""),
                                 ws=cfg.ws_opts())
            say(f"asr ready {time.perf_counter() - t0:.1f}s")
        return st["asr"]

    def ensure_mt():
        if st["mt"] is None:
            from .translate import NullMT, make_translator
            eng = str(cfg.engine or "qwen").strip().lower()
            try:
                mt = make_translator(eng)
            except Exception as e:
                # NLLB 已移除：没有兜底模型 → 只出原文（NullMT），并把原因报给主进程
                # [:200] 而不是 [:70]：make_translator 现在会把**每个候选**的原因都带上，
                # 截太短会把第一个（通常才是真因）剪掉。
                say(f"warn: 翻译引擎全部不可用({str(e)[:200]}) → 只出原文")
                mt = NullMT()
            st["mt"] = mt
            # 把"跑在 GPU 还是 CPU"写清楚：以前只有一条 warn 藏在中间，
            # 用户（和排查的人）根本看不出翻译到底用没用显卡
            _dev = getattr(mt, "device", None)
            if not _dev:
                try:
                    import torch as _t
                    _dev = "cuda" if _t.cuda.is_available() else "cpu"
                except Exception:
                    _dev = "?"
            say(f"{eng} 就绪 {type(mt).__name__} "
                f"{float(getattr(mt, 'load_s', 0) or 0):.1f}s | 设备 {_dev}")
        return st["mt"]

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue
        rid = req.get("id")
        method = req.get("method")
        a = req.get("args") or {}
        try:
            if method == "preload":
                ensure_asr()
                ensure_mt()
                send({"id": rid, "ok": True, "result": None})
            elif method == "transcribe":
                text, meta = ensure_asr().transcribe(
                    _unpack(a["audio"]), language=a.get("language"),
                    prompt=a.get("prompt"), return_confidence=bool(a.get("conf")),
                    beam_size=int(a.get("beam") or 5),
                    without_timestamps=bool(a.get("wots")),
                    no_fallback=bool(a.get("nfb")))
                send({"id": rid, "ok": True, "result": [text, meta]})
            elif method == "detect":
                lang, prob = ensure_asr().detect_lang(_unpack(a["audio"]))
                send({"id": rid, "ok": True, "result": [lang, prob]})
            elif method == "translate":
                mt = ensure_mt()
                mt.set_terms(a.get("terms") or [])
                try:
                    out = mt.translate(a["text"], a["lang"],
                                       context=list(a.get("context") or []))
                except TypeError:               # 不支持 context 的旧实现
                    out = mt.translate(a["text"], a["lang"])
                send({"id": rid, "ok": True, "result": out})
            elif method == "translate_stream":
                mt = ensure_mt()
                mt.set_terms(a.get("terms") or [])
                if not hasattr(mt, "translate_stream"):
                    # 引擎不支持逐 token 流式（HyMT2 走 PyTorch、NullMT 是占位）→ 退化成
                    # "整句一次产出"，仍按流式协议回一块 + done。
                    # ⚠ 不能直接调 mt.translate_stream：子进程抛 AttributeError 后，父进程
                    # 的流式循环只认 chunk/done，会一直等到 30s 超时 → **整句译文丢失**
                    #（2026-09-18 实锤：Hy-MT2 完全不出译文，日志只有 translate_stream 流式超时）。
                    try:
                        out = mt.translate(a["text"], a["lang"],
                                           context=list(a.get("context") or []))
                    except TypeError:               # 不支持 context 的旧实现
                        out = mt.translate(a["text"], a["lang"])
                    dst = out[0] if isinstance(out, (tuple, list)) else out
                    if dst:
                        send({"id": rid, "chunk": dst})
                else:
                    for chunk in mt.translate_stream(a["text"], a["lang"],
                                                     context=list(a.get("context") or [])):
                        send({"id": rid, "chunk": chunk})
                send({"id": rid, "done": True})
                send({"id": rid, "ok": True, "result": None})
            elif method == "chat":
                send({"id": rid, "ok": True,
                      "result": ensure_mt().chat(a.get("messages") or [])})
            elif method == "load_s":
                send({"id": rid, "ok": True,
                      "result": float(getattr(st["mt"], "load_s", 0) or 0)})
            else:
                send({"id": rid, "ok": False, "error": f"未知方法 {method}"})
        except Exception as e:
            send({"id": rid, "ok": False,
                  "error": f"{type(e).__name__}: {str(e)[:180]}",
                  "tb": traceback.format_exc()[-300:]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())