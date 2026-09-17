@echo off
rem ============================================================
rem  MaiSubtitle 控制台启动（排障用：实时看日志，关窗即退出）
rem  排障先看这里：任何"启动不了 / 没反应"都能在这个窗口看到原因。
rem  为什么需要守护：CTranslate2 的 CUDA 调用偶尔卡住不返回，Windows 会判定
rem        "未响应"（STATUS_APPLICATION_HANG），原生层无法从 Python 中断；
rem        子进程每秒写 logs\heartbeat.txt，卡死 25 秒即杀掉重启兜底。
rem  日志：logs\supervisor.log（守护层）| logs\crash.log（崩溃栈）
rem ============================================================
setlocal
cd /d "%~dp0"
title MaiSubtitle
set PYTHONIOENCODING=gbk:replace

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 找不到 .venv\Scripts\python.exe
    echo        新机器请先双击 安装_首次使用.bat
    pause
    exit /b 1
)

".venv\Scripts\python.exe" "scripts\preflight.py"
if errorlevel 1 (
    echo.
    echo 修法：双击 安装_首次使用.bat
    pause
    exit /b 1
)

".venv\Scripts\python.exe" "scripts\supervisor.py" %*
pause