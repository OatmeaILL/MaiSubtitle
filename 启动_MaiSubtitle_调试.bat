@echo off
rem ============================================================
rem  MaiSubtitle 调试启动（有控制台 + 卡死转储线程栈）
rem  用 python.exe（唯一带控制台的入口）+ --debug
rem  --debug 会打开 faulthandler：卡死 15 秒自动转储线程栈到
rem  logs\faulthandler.log，进程被冻住时也能取证。
rem  卡住时别关这个窗口：先看它，再看 logs\ 里的各类日志。
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
if not exist "scripts\live_demo.py" (
    echo [错误] 找不到 scripts\live_demo.py
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

echo 以调试模式启动（关闭本窗口即退出程序）。
".venv\Scripts\python.exe" "scripts\live_demo.py" --debug %*
pause