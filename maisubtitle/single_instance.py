# -*- coding: utf-8 -*-
"""单实例守卫：同一时刻只允许一份 MaiSubtitle（应用 / 守护进程各自一把锁）。

为什么需要（2026-09-16 日志实锤）：21:06:41 出现**两份守护进程 + 两个 live_demo**
（pid 10768/30872），日志里 `|-- vad`、`qwen3 就绪` 全部重复两遍 —— 两套 WASAPI
采集 + 两套模型去抢同一块 8GB 显存，表现就是"越来越卡、两个窗口、字幕乱跳"。

用 Windows 命名互斥体（Local\\ 命名空间 = 仅当前登录会话），而不是锁文件：
进程崩溃 / 被 taskkill 时内核自动释放，不会留下要人工清理的僵尸锁。
"""
import ctypes
from ctypes import wintypes

_ERROR_ALREADY_EXISTS = 183


def acquire(name: str) -> bool:
    """尝试独占名为 name 的互斥体。

    成功 → True（句柄保留在模块级列表里，持有到进程退出）；
    已被别的实例占用 → False。非 Windows 或创建失败一律返回 True（不拦），
    宁可多开也不能因为守卫本身把程序挡在门外。
    """
    try:
        k32 = ctypes.windll.kernel32
    except Exception:
        return True
    try:
        k32.CreateMutexW.restype = wintypes.HANDLE
        k32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL,
                                     wintypes.LPCWSTR]
        handle = k32.CreateMutexW(None, False, "Local\\" + name)
        if not handle:
            return True
        if k32.GetLastError() == _ERROR_ALREADY_EXISTS:
            k32.CloseHandle(handle)
            return False
        _HELD.append(handle)          # 保持引用：句柄一关锁就没了
        return True
    except Exception:
        return True


_HELD: list = []