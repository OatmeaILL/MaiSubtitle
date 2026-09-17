@echo off
rem ============================================================
rem  MaiSubtitle 首次使用安装（新机器先双击这个）
rem  四件事：建 .venv -> 装依赖（先测速选最快的源）-> 下必需模型 -> 转换翻译模型
rem  参数会原样传给 scripts\setup_first_run.py，例如：
rem      安装_首次使用.bat --check        只看缺什么，不下载
rem      安装_首次使用.bat --skip-models  只装依赖，不下模型
rem      安装_首次使用.bat --mirror 清华  指定 PyPI 源（默认自动测速选最快）
rem ============================================================
setlocal
cd /d "%~dp0"
title MaiSubtitle 安装
rem Python 输出用控制台自己的编码（GBK）：中文正常显示，也不会因编码报错中断
set PYTHONIOENCODING=gbk:replace
rem 本机没有 uv 时的应急源（只用来装 uv 本身；装依赖的源由脚本测速选）
set PIPMIRROR=https://pypi.tuna.tsinghua.edu.cn/simple/

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
    rem 本机的 python 未必是 3.12（3.13/3.14 有些轮子没有）→ 先装上 uv，
    rem 让 uv 自己去取 Python 3.12（python -m uv 是官方支持的调用方式）
    echo   本机没有 uv，先用 python 装一个（pip install uv，走清华源）...
    python -m pip install --user uv -i %PIPMIRROR% --disable-pip-version-check
    python -m uv venv --python 3.12 --seed .venv
    if not exist .venv\Scripts\python.exe python -m venv .venv
    goto checkvenv
)
rem 连 python 都没有：自动下载 uv 的免安装版（走代理链），解压即用
echo.
echo   本机没有 Python —— 自动下载 uv（免安装，约 30MB）...
set UVDIR=%LOCALAPPDATA%\MaiSubtitle\uv
set UVZIP=%TEMP%\maisub-uv.zip
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='SilentlyContinue'; $u=@('https://ghfast.top/https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip','https://gh-proxy.com/https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip','https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip'); foreach($x in $u){ Invoke-WebRequest -Uri $x -OutFile $env:TEMP\maisub-uv.zip -TimeoutSec 180; if((Get-Item $env:TEMP\maisub-uv.zip).Length -gt 1048576){break} }; Expand-Archive -Path $env:TEMP\maisub-uv.zip -DestinationPath $env:LOCALAPPDATA\MaiSubtitle\uv -Force"
if not exist "%UVDIR%\uv.exe" goto nouv
"%UVDIR%\uv.exe" venv --python 3.12 --seed .venv
if not exist .venv\Scripts\python.exe "%UVDIR%\uv.exe" venv --python 3.12 .venv
goto checkvenv

:nouv
echo.
echo [错误] 自动下载 uv 失败（网络不通？）。手动装一个再重跑本文件：
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