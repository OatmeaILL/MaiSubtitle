@echo off
rem ============================================================
rem  MaiSubtitle 首次使用安装（新机器先双击这个）
rem  四件事：建 .venv -> 装依赖（阿里云镜像）-> 下必需模型 -> 转换翻译模型
rem  参数会原样传给 scripts\setup_first_run.py，例如：
rem      安装_首次使用.bat --check        只看缺什么，不下载
rem      安装_首次使用.bat --skip-models  只装依赖，不下模型
rem ============================================================
setlocal
cd /d "%~dp0"
title MaiSubtitle 安装
rem Python 输出用控制台自己的编码（GBK）：中文正常显示，也不会因编码报错中断
set PYTHONIOENCODING=gbk:replace

if exist ".venv\Scripts\python.exe" goto hasvenv

echo [1/2] 建 Python 环境 .venv（Python 3.12）...
where uv >nul 2>nul
if not errorlevel 1 (
    rem --seed：顺手把 pip 装进 venv —— uv 建的 venv 默认不带 pip，后面装依赖要用
    uv venv --python 3.12 --seed .venv
    if not exist .venv\Scripts\python.exe uv venv --python 3.12 .venv
    goto checkvenv
)
where py >nul 2>nul
if not errorlevel 1 (
    py -3.12 -m venv .venv
    if not exist ".venv\Scripts\python.exe" py -m venv .venv
    goto checkvenv
)
where python >nul 2>nul
if not errorlevel 1 (
    python -m venv .venv
    goto checkvenv
)
echo.
echo [错误] 没找到 uv / py / python，没法建环境。装好其中一个再重跑本文件：
echo   1) uv（推荐，自带 Python 3.12）  https://docs.astral.sh/uv/
echo   2) Python 3.12                  https://www.python.org/downloads/
pause
exit /b 1

:checkvenv
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [错误] 建 .venv 失败（见上面的报错）。
    pause
    exit /b 1
)

:hasvenv
echo [2/2] 检查依赖与模型（缺什么补什么；已装的会跳过）...
echo.
".venv\Scripts\python.exe" "scripts\setup_first_run.py" %*
set RC=%ERRORLEVEL%
echo.
if "%RC%"=="0" (
    echo 装好了：双击 启动_MaiSubtitle.bat 开始使用。
) else (
    echo 还没装完（退出码 %RC%）：按上面的提示处理后，重跑本文件。
)
pause
exit /b %RC%