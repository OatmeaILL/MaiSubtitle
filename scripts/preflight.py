# -*- coding: utf-8 -*-
"""启动前体检（1~2 秒）：给启动器用 —— 依赖能不能 import、必需权重在不在。

为什么需要它：启动器用 pythonw（无控制台），环境不完整时 `import` 阶段就死掉，
而 `sys.excepthook` 要到 live_demo 更靠后才会装上 → **屏幕上什么都不会显示**，
守护进程还会每 2 秒无限重启（用户实测："双击了没反应"）。有了这道体检，
缺东西时启动器会**停下来并写清缺什么、敲哪条命令**。

退出码：0 = 可以启动；1 = 有硬伤。`--quiet` 只在失败时输出（启动器平时不刷屏）。
"""
import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
# 依赖清单与模型判据只有一份（在 setup_first_run 里），免得两处各写一遍、改一处漏一处
from setup_first_run import CORE_IMPORTS, MODEL_MARKS  # noqa: E402


def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _deep_ok() -> list:
    """真正 import 两个最要命的（find_spec 看不出"装坏了"：本机有过 pip 把包
    拆成空壳的先例）。这两个 import 很快，代价可接受。"""
    bad = []
    for mod in ("numpy", "PyQt6.QtWidgets"):
        try:
            __import__(mod)
        except Exception as e:
            bad.append(f"{mod}（{type(e).__name__}: {str(e)[:60]}）")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description="启动前体检")
    ap.add_argument("--quiet", action="store_true", help="只在失败时输出")
    a = ap.parse_args()

    problems = []
    miss = [m for m in CORE_IMPORTS if not _importable(m)]
    if miss:
        problems.append("缺依赖：" + " ".join(miss))
    bad = _deep_ok()
    if bad:
        problems.append("依赖装坏了：" + "、".join(bad))
    for name, rel in MODEL_MARKS:
        if not (ROOT / rel).exists():
            problems.append(f"缺模型：{name}（需 {rel}）")

    if not problems:
        if not a.quiet:
            print("[OK] 启动前检查通过")
        return 0

    print("[错误] 启动前检查没通过：")
    for p in problems:
        print("   · " + p)
    print("   → 双击 安装_首次使用.bat 修好后再启动"
          "（也可以：.venv\\Scripts\\python.exe scripts\\setup_first_run.py）")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())