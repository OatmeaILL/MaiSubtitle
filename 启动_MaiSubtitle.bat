@echo off
rem ============================================================
rem  MaiSubtitle 日常启动（无控制台窗口 + 崩溃/卡死自动重启）
rem  识别/翻译模型都在 config.json 里选，不会被命令行参数强制覆盖。
rem  启动时可用参数透传给 live_demo.py，例如：
rem      启动_MaiSubtitle.bat --engine qwen3
rem  启动前会跑 scripts\preflight.py 体检：依赖/模型缺失就停下来说清楚
rem  （以前是"双击了没反应"：pythonw 无控制台 + 守护进程无限重启，见 HANDOVER 六十六）
rem ============================================================
setlocal
cd /d "%~dp0"
title MaiSubtitle
set PYTHONIOENCODING=gbk:replace

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 找不到 .venv\Scripts\python.exe
    echo        新机器请先双击 安装_首次使用.bat（建环境+装依赖+下模型，一步到位）
    pause
    exit /b 1
)
if not exist "scripts\live_demo.py" (
    echo [错误] 找不到 scripts\live_demo.py
    pause
    exit /b 1
)

".venv\Scripts\python.exe" "scripts\preflight.py" --quiet
if errorlevel 1 (
    echo.
    echo 修法：双击 安装_首次使用.bat
    pause
    exit /b 1
)

rem 守护进程：子进程崩溃(原生 access violation)/卡死(心跳停 25s)会自动重启
rem 日志：logs\supervisor.log 与 logs\crash.log；要退出请用托盘菜单的"退出"
start "" ".venv\Scripts\pythonw.exe" "scripts\supervisor.py" %*
exit /b 0