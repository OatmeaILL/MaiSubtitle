# -*- coding: utf-8 -*-
"""开机自启（仅 Windows）：HKCU 的 CurrentVersion/Run 键。

只写当前用户的 Run 键（不碰服务/计划任务）；命令行 = 当前解释器 + live_demo.py
（PyInstaller 冻结后就是 exe 自身）。单实例守卫天然兜住"自启 + 手动双击"双开。
"""
import sys


def _cmd() -> str:
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}"'
    try:
        from pathlib import Path
        script = Path(sys.argv[0]).resolve()
        return f'"{sys.executable}" "{script}"'
    except Exception:
        return ""


def enabled() -> bool:
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                           r"Software\Microsoft\Windows\CurrentVersion\Run")
        try:
            winreg.QueryValueEx(k, "MaiSubtitle")
            return True
        finally:
            k.Close()
    except OSError:
        return False


def set_autostart(on: bool) -> tuple[bool, str]:
    try:
        import winreg
        cmd = _cmd()
        if not cmd:
            return False, "无法确定启动命令行（未知入口）"
        k = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                           r"Software\Microsoft\Windows\CurrentVersion\Run",
                           0, winreg.KEY_SET_VALUE)
        try:
            if on:
                winreg.SetValueEx(k, "MaiSubtitle", 0, winreg.REG_SZ, cmd)
            else:
                try:
                    winreg.DeleteValue(k, "MaiSubtitle")
                except FileNotFoundError:
                    pass
        finally:
            k.Close()
        return True, ""
    except OSError as e:
        return False, f"{type(e).__name__}: {e}"
