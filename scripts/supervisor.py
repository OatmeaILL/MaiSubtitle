# -*- coding: utf-8 -*-
"""守护进程：启动 MaiSubtitle 并监控其主线程心跳，卡死或崩溃后自动重启。

背景（2026-09-14 实锤）：CTranslate2 的 CUDA 编码调用偶发**卡住不返回**
（faulthandler 连续 3 份转储都停在 faster_whisper encode，共 45s+）。此时
主线程被阻塞、Windows 判定"未响应"（退出码 0xCFFFFFFF = STATUS_APPLICATION_HANG），
用户只能手动关掉。原生调用无法从 Python 层中断，所以只能在**进程外**兜底：

  子进程每秒写 logs/heartbeat.txt；本守护进程发现心跳停止 > STALE 秒
  → taskkill 掉整个进程树 → 冷却后重新拉起。

用法：
    .venv\\Scripts\\python.exe scripts\\supervisor.py [传给 live_demo.py 的参数]

调试用环境变量：MAISUB_SUP_STALE（默认 25 秒）、MAISUB_SUP_GRACE（首次心跳等待，默认 180 秒）、
              MAISUB_SUP_PY / MAISUB_SUP_APP（换解释器 / 换脚本：排障自测用）、
              MAISUB_SUP_FAST_EXIT / MAISUB_SUP_FAST_MAX（秒退阈值与次数，见下）、
              MAISUB_NO_POPUP=1（不弹系统弹窗：自动化测试用）
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LOGS = ROOT / "logs"
HB = LOGS / "heartbeat.txt"
SUP_LOG = LOGS / "supervisor.log"
START_LOG = LOGS / "live_demo_startup.log"
PYW = Path(os.environ.get("MAISUB_SUP_PY",
                          str(ROOT / ".venv" / "Scripts" / "pythonw.exe")))
APP = Path(os.environ.get("MAISUB_SUP_APP", str(ROOT / "scripts" / "live_demo.py")))
STALE = float(os.environ.get("MAISUB_SUP_STALE", 25))
GRACE = float(os.environ.get("MAISUB_SUP_GRACE", 180))
# 连续"秒退"（活不到 FAST_EXIT_S 秒）FAST_MAX 次就**不再无限重启**：
# 实测（2026-09-17）依赖没装时，pythonw 无控制台 → 屏幕上什么都没有，
# 而守护进程每 2 秒重启一次、永远不停 —— 用户唯一的感受是"双击了没反应"。
FAST_EXIT_S = float(os.environ.get("MAISUB_SUP_FAST_EXIT", 5.0))
FAST_MAX = int(os.environ.get("MAISUB_SUP_FAST_MAX", 3))


def log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(SUP_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def heartbeat_age():
    """返回心跳文件的"年龄"（秒）；没有文件返回 None。"""
    try:
        return time.time() - float(HB.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _child_output():
    """子进程输出去哪：无控制台（pythonw 启动）→ 落 logs/live_demo_startup.log，
    因为那种情况下 stdout/stderr 是黑洞，起不来时"什么都没留下"；
    有控制台（python.exe / 控制台 bat）→ 照旧继承，用户能实时看日志。"""
    if sys.stdout is not None:
        return None
    try:
        return open(START_LOG, "w", encoding="utf-8", errors="replace")
    except Exception:
        return None


def _popup(title: str, text: str):
    """pythonw 没有控制台 → 用系统弹窗把话说给用户听（ctypes 是标准库）。
    MAISUB_NO_POPUP=1 关掉（自动化测试不能卡在等点击上）。"""
    if os.environ.get("MAISUB_NO_POPUP"):
        return
    try:
        import ctypes
        # MB_OK | MB_ICONERROR | MB_SETFOREGROUND
        ctypes.windll.user32.MessageBoxW(None, text, title, 0x00000000 | 0x10 | 0x10000)
    except Exception as e:
        log(f"（弹窗失败：{type(e).__name__}: {e}）")


def _bail(reason: str):
    """连续秒退 → 不再无限重启：写日志 + 弹窗，把"缺什么、下一步做什么"说清楚。"""
    tail = ""
    try:
        lines = START_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-8:]).strip()
    except Exception:
        pass
    log("放弃重启：" + reason)
    body = (f"MaiSubtitle 连续启动失败，已停止自动重启。\n\n{reason}\n\n"
            "怎么办：\n"
            "  1. 双击 安装_首次使用.bat —— 它会检查依赖与模型并自动补齐\n"
            "  2. 想直接看报错：双击 启动_MaiSubtitle_控制台.bat\n"
            "  3. 日志：logs\\live_demo_startup.log、logs\\supervisor.log\n")
    if tail:
        body += f"\n启动输出（最后几行）：\n{tail}\n"
    _popup("MaiSubtitle 启动失败", body)


def main():
    args = sys.argv[1:]
    # --no-ui（纯控制台）没有 UI 心跳定时器，不做卡死判定，只做崩溃重启
    watch_hb = "--no-ui" not in args
    # 单实例守卫：两个守护进程会各拉起一份 live_demo（日志实锤 2026-09-16
    # 21:06:41 双实例抢 GPU）→ 后来的守护进程直接退出，不启动子进程。
    # 注意用独立锁名：应用锁由子进程 live_demo 持有，两者不冲突。
    if watch_hb and os.environ.get("MAISUB_ALLOW_MULTI") != "1":
        from maisubtitle.single_instance import acquire
        if not acquire("MaiSubtitle.Supervisor"):
            log("已有守护进程在运行（单实例守卫）→ 本次退出，不启动 live_demo")
            return
    log(f"守护进程启动（卡死阈值 {STALE:.0f}s，心跳监控: {watch_hb}）："
        f"{' '.join(args) or '无额外参数'}")
    fast = 0                          # 连续"秒退"次数（成功后重置）
    while True:
        try:
            HB.unlink()
        except Exception:
            pass
        fo = _child_output()
        child = subprocess.Popen([str(PYW), str(APP)] + args, cwd=str(ROOT),
                                 env={**os.environ, "MAISUB_SUPERVISED": "1"},
                                 stdout=fo, stderr=(subprocess.STDOUT if fo else None))
        log(f"已启动 live_demo（pid={child.pid}）")
        started = time.time()
        seen_hb = False
        while True:
            time.sleep(2)
            if child.poll() is not None:
                code = child.returncode
                if code == 0:
                    log("进程正常退出（用户主动退出）→ 守护进程结束，不再重启")
                    return
                life = time.time() - started
                if life < FAST_EXIT_S:
                    fast += 1
                    log(f"进程秒退（活了 {life:.1f}s，code={code}）"
                        f"→ 第 {fast}/{FAST_MAX} 次")
                    if fast >= FAST_MAX:
                        _bail(f"连续 {fast} 次启动即退出（最后退出码 {code}）——"
                              f"多半是依赖没装或模型缺失")
                        return 1
                else:
                    fast = 0          # 活过阈值再挂 → 不是"环境不对"，照旧重启
                    log(f"进程异常退出（code={code}，0xCFFFFFFF=应用挂起）→ 2 秒后重启")
                break
            age = heartbeat_age()
            if not watch_hb:
                continue          # --no-ui 模式不看心跳
            if age is None:
                # 还没写出第一次心跳：模型加载期，给足宽限
                if not seen_hb and time.time() - started > GRACE:
                    log(f"超过 {GRACE:.0f}s 仍无心跳，判定启动异常 → 重启")
                    _kill(child)
                    break
                continue
            if not seen_hb:
                seen_hb = True
                log("已收到心跳，进入监控")
                fast = 0          # 真的起来了：清掉秒退计数
            if age > STALE:
                log(f"心跳停止 {age:.0f}s（> {STALE:.0f}s）→ 判定卡死，杀掉重启")
                _kill(child)
                break
        if fo is not None:
            try:
                fo.close()
            except Exception:
                pass
        time.sleep(2)          # 冷却，避免疯狂重启


def _kill(child):
    try:
        subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                       capture_output=True)
    except Exception:
        try:
            child.kill()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
