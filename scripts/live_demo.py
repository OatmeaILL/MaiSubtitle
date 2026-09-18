"""MaiSubtitle 悬浮字幕应用（实时管线 + 托盘 + 悬浮窗 + 设置）。

    uv run python scripts/live_demo.py            # 默认配置启动
    uv run python scripts/live_demo.py --no-ui    # 纯控制台
    uv run python scripts/live_demo.py --engine qwen3 --glossary terms.csv

托盘：左键显隐，右键完整菜单（显隐/暂停/双语/导出/设置/术语库/退出）。
悬浮窗右键：同一份菜单（固定尺寸窗体，锚点稳定不裁剪）。
全局热键：F9 显隐 F10 穿透 F11 双语 F12 暂停 F5 源语言 F7 切屏 F8 术语库 F6 导出。

线程模型：热键回调（键盘钩子线程）与托盘/菜单（Qt 线程）都只改 targets 标志，
所有 Qt 操作统一定时器在主线程应用——不跨线程碰 UI，不卡未响应。
"""
import argparse
import os
import queue
import threading
import sys
import time as _time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# GIL 切换间隔：曾调到 0.0015 想"让 Qt 主线程更勤快"，但线程多（主/管线/worker/
# 看门狗…）时，间隔过小会让解释器大量时间花在 GIL 交接上，反而可能把主线程饿住
# 数秒（表现为"未响应"）。恢复 Python 默认 5ms。
sys.setswitchinterval(0.005)

from maisubtitle.config import (LOGS_DIR, AppConfig,  # noqa: E402
                                save_preserving_cli)

ap = argparse.ArgumentParser()
ap.add_argument("--no-ui", action="store_true")
ap.add_argument("--debug", action="store_true",
                help="调试：卡死 15s 自动把所有线程的调用栈转储到控制台与 logs/faulthandler.log")
ap.add_argument("--engine", default=None,
                choices=["hymt2", "qwen", "qwen3"])
ap.add_argument("--model", default=None, choices=["large-v3-turbo"])
ap.add_argument("--glossary", default=None, help="术语库 CSV/JSON（默认读 config.json）")
opts = ap.parse_args()

# ---- 单实例守卫：防"两份 live_demo 抢同一块 GPU" --------------------------
# 日志实锤（2026-09-16 21:06:41）：两个守护进程各拉起一份 live_demo（pid
# 10768/30872），日志里 vad/引擎就绪全部重复两遍，两套采集+模型抢 8GB 显存，
# 表现就是"越来越卡 + 两个悬浮窗"。这里直接让后来者退出（退出码 0：
# 守护进程会按"用户主动退出"处理，不再重启）。
# --no-ui（纯控制台/基准脚本）不设限；确需多开时设 MAISUB_ALLOW_MULTI=1。
if not opts.no_ui and os.environ.get("MAISUB_ALLOW_MULTI") != "1":
    from maisubtitle.single_instance import acquire  # noqa: E402
    if not acquire("MaiSubtitle.App"):
        _msg = ("已有 MaiSubtitle 实例在运行（单实例守卫）：本次启动已退出。"
                "如需强制多开，请设环境变量 MAISUB_ALLOW_MULTI=1 后重试。")
        print(_msg, flush=True)
        try:
            with open(LOGS_DIR / "live_demo.log", "a", encoding="utf-8") as _f:
                _f.write(f"[{datetime.now():%H:%M:%S}] |-- {_msg}\n")
        except Exception:
            pass
        sys.exit(0)

# ---- 原生崩溃取证：**始终开启**，把致命崩溃（access violation 等）的线程栈
# 写进 logs/crash.log。GPU/CTranslate2 的原生崩溃会让进程直接消失、Python 异常
# 机制完全来不及介入（error.log 里什么都没有），只有 faulthandler 抓得到。
# 它只在崩溃时写文件，正常运行零开销。
import faulthandler as _fh_mod
try:
    # 必须显式 utf-8：默认走系统 ANSI 代码页，中文时间戳头会写坏（第一条是英文所以看不出）
    _crash_fp = open(LOGS_DIR / "crash.log", "a", encoding="utf-8")
    _fh_mod.enable(file=_crash_fp, all_threads=True)
    # 写一条带时间戳的分隔头：崩栈本身没有时间，靠它和日志对齐
    _crash_fp.write(f"\n=== 启动 {datetime.now():%Y-%m-%d %H:%M:%S} pid={os.getpid()} ===\n")
    _crash_fp.flush()
except Exception:
    _crash_fp = None

# 调试模式额外开"定时转储"（15s 一份所有线程栈），用于排查卡死（非崩溃）
if opts.debug or os.environ.get("MAISUB_FAULTHANDLER") == "1":
    _fh2 = open(LOGS_DIR / "faulthandler.log", "a")
    _fh_mod.dump_traceback_later(15.0, repeat=True, file=_fh2)
    print(f"[debug] faulthandler 已开启：卡死 15s 会把所有线程栈写入 "
          f"{LOGS_DIR / 'faulthandler.log'}", flush=True)

cfg = AppConfig.load()
# CLI 透传只影响**本次运行**：记下磁盘原值与覆盖值，退出保存时还原。
# 否则"用某个 bat 跑过一次"就会把设置里的选择永久改掉
# （曾发生的坑：bat 传 --engine qwen，退出回写把 config.json 的 qwen3 覆盖成 qwen）。
cli_overrides: dict = {}                 # 字段 -> (磁盘原值, 本次覆盖值)
if opts.engine:
    cli_overrides["engine"] = (cfg.engine, opts.engine)
    cfg.engine = opts.engine
if opts.model:
    cli_overrides["asr_model"] = (cfg.asr_model, opts.model)
    cfg.asr_model = opts.model
if opts.glossary:
    cli_overrides["glossary_path"] = (cfg.glossary_path, opts.glossary)
    cfg.glossary_path = opts.glossary


def save_cfg_guarded():
    """退出时保存配置：CLI 透传字段还原为磁盘原值（透传不写回文件）。"""
    save_preserving_cli(cfg, cli_overrides)

log_path = LOGS_DIR / "live_demo.log"
last_state = ["启动"]                  # 看门狗用：记录管线最后所处阶段
ui_queue: queue.Queue = queue.Queue(maxsize=256)
LANG_TAG = {"en": "EN", "ja": "日本語", "ko": "한국어", "zh": "中文"}


def on_subtitle(cid, src, dst, lang, lat_ms):
    """控制台/日志留痕（cue id 只是用来区分同一句，日志里不打印）。"""
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] ({lang}) {src}" + (f"  →  {dst}" if dst else "") + f"  [{lat_ms}ms]"
    print(line, flush=True)
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def on_state(s):
    last_state[0] = s
    print(f"|-- {s}", flush=True)
    try:
        with open(LOGS_DIR / "live_demo.log", "a", encoding="utf-8") as f:
            import datetime as _dt
            f.write("[" + _dt.datetime.now().strftime("%H:%M:%S") + "] |-- " + s + "\n")
    except Exception:
        pass
    if s.startswith("notify: ") or s.startswith("error: "):
        try:
            with open(LOGS_DIR / "error.log", "a", encoding="utf-8") as f:
                f.write("\n[" + str(datetime.now()) + "] pipeline: " + s + "\n")
        except Exception:
            pass
    if s.startswith("notify: "):
        # 注意：on_state 跑在管线线程，托盘是 Qt 对象——必须经 ui_queue 回主线程，
        # 跨线程直接调 showMessage 属于 Qt 线程违规（会随机卡死/崩溃）。
        try:
            ui_queue.put_nowait(("__notify__", s[8:], ""))
        except Exception:
            pass
    if s.startswith(("warn: ", "error: ", "[错误]")):
        # 关键异常**不能只在后台/日志里**（GPU 子进程的 warn 以前只有日志，用户在界面上
        # 什么都看不到，只能自己去翻 logs/）。同样必须经 ui_queue 回主线程再碰 Qt。
        try:
            ui_queue.put_nowait(("__alert__", s, ""))
        except Exception:
            pass
    # 悬浮窗状态文本（经 ui_queue 回主线程应用）：加载阶段显示细粒度进度
    status = None
    if s in ("vad", "asr", "engine"):
        status = {"vad": "正在启动…",
                  "asr": "正在加载识别模型…（首次较久）",
                  "engine": "正在加载翻译模型…（首次较久）"}[s]
    elif " 就绪 " in s:
        status = "模型就绪，正在启动…"
    elif s == "running":
        status = "等待语音…"
    elif s == "stopped":
        status = "已停止"
    elif s.startswith("设备切换"):
        status = s
    if status:
        try:
            ui_queue.put_nowait(("__status__", status, ""))
        except Exception:
            pass


from maisubtitle.live import LivePipeline  # noqa: E402

pipe = LivePipeline(cfg, on_subtitle=on_subtitle, on_state=on_state)
pipe.start()


def shutdown_report():
    r = pipe.latency_report()
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"延迟统计: {r} 统计: {pipe.stats}\n")
    except Exception:
        pass
    print("延迟统计:", r)
    print("统计:", pipe.stats)


if opts.no_ui:
    import time
    try:
        while True:
            _time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop()
        shutdown_report()
    sys.exit(0)

from PyQt6.QtCore import QTimer, QPoint, Qt  # noqa: E402
from PyQt6.QtGui import QAction, QActionGroup, QCursor  # noqa: E402
from PyQt6.QtWidgets import QApplication, QDialog, QMenu, QMessageBox  # noqa: E402
from maisubtitle.overlay import SubtitleOverlay  # noqa: E402
from maisubtitle.tray import TrayController, make_icon  # noqa: E402

# Windows 的应用身份：不设的话系统通知/任务栏会把程序显示成 "Python"（进程名）。
try:
    import ctypes as _ct
    _ct.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
        "MaiSubtitle.RealtimeSubtitle")
except Exception:
    pass

app = QApplication([])
app.setApplicationName("MaiSubtitle")          # 托盘气泡/对话框标题等都用这个名字
app.setApplicationDisplayName("MaiSubtitle")
app.setWindowIcon(make_icon())
app.setQuitOnLastWindowClosed(False)  # 关字幕窗 ≠ 退出，走托盘退出
overlay = SubtitleOverlay(cfg.screen, cfg.font_size_src, cfg.font_size_dst,
                          cfg.opacity)
overlay.show()
overlay.set_display_mode(cfg.display_mode)
overlay.set_history(int(getattr(cfg, "history_lines", 1) or 1))

# 显示模式：双语 → 仅译文 → 仅原文 循环（热键 F11 / 菜单）
MODE_ORDER = ["bilingual", "target", "source"]
MODE_LABEL = {"bilingual": "双语", "target": "仅译文", "source": "仅原文"}

# 目标状态：任何输入源（热键/托盘/右键）只改这里，由主线程定时器统一应用
targets = {"visible": True, "paused": False, "through": cfg.click_through}
actions = {"settings": False, "editor": False, "export": False,
           "screen_cycle": False, "mode_cycle": False, "lang_cycle": False}


# 源语言：auto=自动检测；其余=强制按该语言识别（全部走这个语言，不做检测/切换）
LANG_CYCLE = ["auto", "en", "ja", "ko", "zh"]
LANG_NAME = {"auto": "自动检测", "en": "英语", "ja": "日语", "ko": "韩语", "zh": "中文"}


def apply_source_language(code: str):
    """设置源语言：写配置 + 立即生效（不用重启）+ 托盘提示。"""
    code = (code or "auto").strip().lower()
    if code not in LANG_CYCLE:
        code = "auto"
    cfg.source_language = code
    try:
        cfg.save()
    except Exception:
        pass
    try:
        pipe.set_source_language(code)      # 立即生效
    except Exception:
        pass
    print("源语言:", LANG_NAME[code], f"（{code}）")
    try:
        tray.notify(f"源语言：{LANG_NAME[code]}")
    except Exception:
        pass
    return code


def cycle_source_language():
    """F5：循环切换（菜单里是子菜单直接选，不循环）。"""
    cur = str(getattr(cfg, "source_language", "auto") or "auto").lower()
    nxt = LANG_CYCLE[(LANG_CYCLE.index(cur) + 1) % len(LANG_CYCLE)] \
        if cur in LANG_CYCLE else "auto"
    return apply_source_language(nxt)


def toggle(key: str):
    targets[key] = not targets[key]


def menu_builder() -> QMenu:
    """唯一菜单定义：托盘与悬浮窗右键共用。"""
    menu = QMenu()
    pairs = [
        ("隐藏字幕" if targets["visible"] else "显示字幕", lambda: toggle("visible")),
        ("暂停" if not targets["paused"] else "继续", lambda: toggle("paused")),
        (f"显示：{MODE_LABEL.get(overlay.display_mode, '双语')}（点击切换）",
         lambda: actions.__setitem__("mode_cycle", True)),
        ("关闭点击穿透" if targets["through"] else "开启点击穿透", lambda: toggle("through")),
    ]
    for text, fn in pairs:
        act = QAction(text, menu)
        act.triggered.connect(fn)
        menu.addAction(act)
    # 源语言：**子菜单直接选**（不再是"点一次切一次"），当前项打勾
    sub = menu.addMenu("源语言")
    grp = QActionGroup(sub)
    grp.setExclusive(True)
    cur_lang = str(getattr(cfg, "source_language", "auto") or "auto").lower()
    for code in LANG_CYCLE:
        act = QAction(f"{LANG_NAME[code]}", sub)
        act.setCheckable(True)
        act.setChecked(code == cur_lang)
        act.triggered.connect(lambda _=False, c=code: apply_source_language(c))
        grp.addAction(act)
        sub.addAction(act)
    menu.addSeparator()
    for text, key in [("导出本次会话 SRT", "export"), ("设置…", "settings"),
                      ("术语库…", "editor")]:
        act = QAction(text, menu)
        act.triggered.connect(lambda _=False, k=key: actions.__setitem__(k, True))
        menu.addAction(act)
    menu.addSeparator()
    act_quit = QAction("退出", menu)
    act_quit.triggered.connect(app.quit)
    menu.addAction(act_quit)
    return menu


tray = TrayController(left_click=lambda: toggle("visible"),
                      menu_builder=menu_builder)


def show_overlay_menu():
    """悬浮窗右键：与托盘同一份菜单，弹出位置钳制在屏幕内。"""
    menu = menu_builder()
    menu.adjustSize()
    pos = QCursor.pos()
    screen = QApplication.screenAt(pos) or app.primaryScreen()
    geo = screen.availableGeometry()
    x = min(pos.x(), geo.right() - menu.width())
    y = min(pos.y(), geo.bottom() - menu.height())
    menu.exec(QPoint(max(geo.left(), x), max(geo.top(), y)))


overlay.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
overlay.customContextMenuRequested.connect(lambda _pos: show_overlay_menu())


def apply_targets():
    if targets["visible"] != overlay.isVisible():
        overlay.setVisible(targets["visible"])
    if targets["paused"] != pipe.paused:
        (pipe.pause() if targets["paused"] else pipe.resume())
    if actions["mode_cycle"]:
        actions["mode_cycle"] = False
        cur = overlay.display_mode if overlay.display_mode in MODE_ORDER \
            else MODE_ORDER[0]
        nxt = MODE_ORDER[(MODE_ORDER.index(cur) + 1) % len(MODE_ORDER)]
        overlay.set_display_mode(nxt)
        cfg.display_mode = nxt
        cfg.bilingual = (nxt == "bilingual")
        try:
            cfg.save()             # 立即持久化，避免被强杀时丢失
        except Exception:
            pass
        print("显示模式切换为:", MODE_LABEL.get(nxt, nxt))
    if targets["through"] != overlay.click_through:
        overlay.set_click_through(targets["through"])
    if actions["lang_cycle"]:
        actions["lang_cycle"] = False
        cycle_source_language()
    if actions["screen_cycle"]:
        actions["screen_cycle"] = False
        print("屏幕:", overlay.cycle_screen())
    if actions["export"]:
        actions["export"] = False
        out = LOGS_DIR / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt"
        n = pipe.export_session(str(out), bilingual=cfg.bilingual)
        print(f"已导出本次会话字幕: {out}（{n} 条）")
        tray.notify(f"已导出 {n} 条字幕\n{out.name}")
    if actions["settings"]:
        actions["settings"] = False
        _open_settings()
    if actions["editor"]:
        actions["editor"] = False
        # 术语库路径：没设置过时默认编辑项目根的 glossary.csv，但**不写回配置** ——
        # "打开看一眼"不等于"启用了术语库"（2026-09-18 修：以前在这里就把
        # cfg.glossary_path 设上，按一次 F8 术语库就永久生效了）。只有用户在编辑器里
        # 真的点过保存，才算自己设置了这个术语库。
        was_default = not cfg.glossary_path
        gp = cfg.glossary_path or str(PROJECT_ROOT / "glossary.csv")
        was_paused = targets["paused"]
        targets["paused"] = True          # 编辑期间暂停出字幕
        apply_targets()
        # exec() 期间定时器停摆，drain 不会跑 → 这里显式把"已暂停"写上屏（否则看起来像卡死）
        overlay.set_status("已暂停（正在编辑术语库…）")
        from maisubtitle.glossary_ui import GlossaryEditor
        dlg = GlossaryEditor(gp, parent=None)
        dlg.exec()
        targets["paused"] = was_paused
        if getattr(dlg, "saved", False):
            if was_default:
                cfg.glossary_path = gp      # 首次启用：随退出保存写进 config.json
            print("术语库编辑完成，已实时生效:", gp)
            tray.notify(f"术语库已保存：{len(dlg.g.terms)} 条\n{Path(gp).name}"
                        + ("（重启后生效）" if was_default else "（已实时生效）"))
        else:
            print("术语库编辑已关闭（未保存）", gp)


_settings_dlg = None            # 持有引用：show() 非阻塞，不能被 Python GC 掉


def _open_settings():
    """打开设置窗口（**非阻塞 show()，而不是 exec()**）。

    实锤（离屏探针 2026-09-17）：`QDialog.exec()` 的嵌套模态循环期间 **Qt 定时器
    完全停摆**（同样 1.5s：exec 只跑了 13 个 60ms tick，show() 跑了 36 个）。
    而悬浮窗刷新（drain / overlay 心跳 / 动效）全靠定时器 → 用 exec() 打开设置时
    字幕会"冻住"（用户实报）。改用 show() + finished 信号：主事件循环照常跑，
    字幕继续刷新，还能边看边调；保存后走与原来完全一样的收尾。

    2026-09-17 又踩一次（用户实报"点不开设置"）：**PyQt6 的窗口枚举必须带 Scope**
    ——写成 `Qt.WindowStaysOnTopHint`（PyQt5 的短写法）会抛 AttributeError，窗口
    永远建不出来，而且**屏幕上一点提示都没有**（槽里抛的异常在 windowed 模式下
    直接进黑洞）。所以这里兜一层 try/except：写 error.log + 托盘通知，
    让"点不开"永远留下证据，而不是靠猜。
    """
    global _settings_dlg
    try:
        if _settings_dlg is not None:
            _settings_dlg.show()          # 曾被隐藏（非关闭）时也能再显示出来
            _settings_dlg.raise_()
            _settings_dlg.activateWindow()
            return
        from maisubtitle.settings_ui import SettingsDialog
        dlg = SettingsDialog(cfg, overlay, len(app.screens()),
                             cli_overrides=cli_overrides, on_exit_app=app.quit)
        _settings_dlg = dlg
        # **非模态**：模态会挡掉其它窗口的输入 → 设置开着时悬浮窗就拖不动了（用户实报）。
        # 置顶是为了不被"总在最前"的悬浮窗压住。
        dlg.setModal(False)
        dlg.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        dlg.finished.connect(lambda _r: _after_settings(dlg))
        dlg.show()
        dlg.raise_()                      # show() 不一定给焦点：显式抬到最前
        dlg.activateWindow()
    except Exception as e:
        _settings_dlg = None
        import traceback as _tb
        try:
            with open(LOGS_DIR / "error.log", "a", encoding="utf-8") as f:
                f.write(f"\n[{datetime.now()}] 打开设置失败: {type(e).__name__}: {e}\n")
                _tb.print_exception(type(e), e, e.__traceback__, file=f)
        except Exception:
            pass
        try:
            tray.notify(f"打开设置失败：{type(e).__name__}: {str(e)[:100]}\n详见 logs/error.log")
        except Exception:
            pass


def _after_settings(dlg):
    """设置窗口关闭后的收尾（保存→应用；取消/直接关闭→什么都不做）。"""
    global _settings_dlg
    _settings_dlg = None
    if dlg.result() != int(QDialog.DialogCode.Accepted):
        return
    from maisubtitle.settings_ui import RESTART_FIELDS
    overlay.move_to_screen(cfg.screen)
    overlay.set_opacity(cfg.opacity)
    # 源语言是运行期可切项：立即生效，不用重启
    try:
        pipe.set_source_language(getattr(cfg, "source_language", "auto"))
    except Exception:
        pass
    print("设置已保存并应用")
    # 改到"重启生效"的项 → 弹窗问是否立即重启（守护进程模式下同样适用）
    rf = list(getattr(dlg, "restart_fields", []) or [])
    if rf:
        names = "、".join(RESTART_FIELDS.get(k, k) for k in rf)
        ans = QMessageBox.question(
            None, "需要重启才生效",
            f"以下改动要重启程序才生效：\n\n{names}\n\n现在重启吗？\n"
            "（选「否」＝保存已生效，下次启动时应用）",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes)
        if ans == QMessageBox.StandardButton.Yes:
            _restart_app()
        else:
            tray.notify(f"改动已保存（{names}），下次启动生效")


def _restart_app():
    """立即重启（"设置改动需要重启"的确认弹窗用）。

    两种启动方式分别处理：
      * 守护进程模式（supervisor 注入 MAISUB_SUPERVISED=1）：以退出码 2 退出，
        交守护进程在 2s 冷却后按原命令行拉起（保住"卡死自动重启"的兜底）；
      * 独立启动：拉一个"睡 2s 再拉起同一命令行"的分离进程 —— 等本进程退出、
        单实例互斥体释放后再启动，否则新实例会被守卫直接拒绝。
    """
    import subprocess
    args = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    if os.environ.get("MAISUB_SUPERVISED") == "1":
        print("设置需要重启：交由守护进程重启")
        app.exit(2)
        return
    helper = ("import subprocess, sys, time\n"
              "time.sleep(2.0)\n"
              f"subprocess.Popen(sys.argv[1:], cwd={str(PROJECT_ROOT)!r})\n")
    flags = 0x00000008 | 0x00000200      # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    subprocess.Popen([sys.executable, "-c", helper, *args],
                     creationflags=flags, close_fds=True, cwd=str(PROJECT_ROOT))
    print("设置需要重启：已安排 2 秒后自动重启")
    app.exit(0)


# ---- 异常上屏：warn/error 一律走**右下角通知（托盘气泡）**，不再弹对话框 ----
# 为什么要上屏：GPU 子进程的 warn 以前只进日志，用户在界面上完全看不到
#（实测：新装环境"翻译引擎全部不可用 → 只出原文"，界面毫无提示，只能翻 logs/）。
# 2026-09-18 用户要求去掉弹窗：报错不再打断操作（以前每条 warn/error 弹一个非模态窗，
# 看视频时得手动关掉）。完整信息仍落 logs/error.log，气泡只显示前两行。
_alerted: set = set()          # 同一句话只提示一次（子进程重启循环会重复报同一条）

# 给最常见的两种问题配一句"人话 + 怎么办"，其余原样展示
ALERT_FRIENDLY = (
    ("翻译引擎全部不可用",
     "翻译用不了，这次只显示原文。\n\n"
     "最常见的原因：本机 torch 是 CPU 版（没吃上显卡）。"
     "可在 设置 → 翻译引擎 换成 qwen，或按 README 装 CUDA 版 torch；"
     "也可以在 设置 里指定一个已有的 CUDA torch 目录。"),
    ("缺模型", "有模型没下载齐。设置窗口底部有「下载缺失的模型」按钮，"
              "或跑 安装_首次使用.bat --check 看缺什么。"),
)


def notify_bg(text: str):
    """把一条消息送到右下角通知（**可从任意线程调**）：经 ui_queue 回主线程。

    主线程已死/Qt 不可用时退回直接 tray.notify，再失败就只留日志 —— 无论如何
    **不再弹系统对话框**（2026-09-18 用户要求：只保留右下角提醒）。
    """
    try:
        ui_queue.put_nowait(("__alert__", text, ""))
    except Exception:
        try:
            tray.notify(text.splitlines()[0][:200], 8000)
        except Exception:
            pass


def alert_user(text: str):
    """主线程里把一条 warn/error 推到前台：托盘气泡（右下角通知）。

    **只在主线程调用**（drain 里）。不再弹窗：`QMessageBox.exec()` 会冻住悬浮窗，
    而非模态窗也会挡视线/需手动关闭 —— 用户明确要求只看右下角提醒。
    """
    key = text.strip()[:110]
    if key in _alerted:
        return
    _alerted.add(key)
    body = text
    for pfx, friendly in ALERT_FRIENDLY:
        if pfx in text:
            body = friendly + "\n\n（原始信息：" + text.strip()[:160] + "）"
            break
    try:
        # 通知有字数上限：只取前两行有内容的行，否则会被截成半句
        lines = [ln for ln in body.splitlines() if ln.strip()][:2]
        tray.notify("\n".join(lines)[:240], 8000)
    except Exception:
        pass


def drain():
    # 字幕队列 → 悬浮窗
    # **同一句只应用最后一条**：流式翻译一句会推几十条中间态，
    # 逐条重排会把主线程拖住（进而队列积压、甚至丢消息）。合并后一句只排一次版。
    pending_cues = {}                # cid -> (src, dst, lang, t0)，dict 保序 = 时间顺序
    try:
        while True:
            item = ui_queue.get_nowait()
            if item[0] == "__status__":      # 生命周期状态文本
                overlay.set_status(item[1])
                continue
            if item[0] == "__notify__":      # 托盘气泡（从管线线程转发而来）
                tray.notify(item[1])
                continue
            if item[0] == "__alert__":       # warn/error → 托盘 + 关键项弹窗
                alert_user(item[1])
                continue
            if item[0] == "__activity__":    # 管线活动提示（识别中…/翻译中…）
                overlay.set_activity(item[1])
                continue
            if item[0] == "__drop__":        # 该句被门控丢弃/撤回 → 撤掉它的行
                _disp.remove(item[1])
                overlay.drop_cue(item[1])
                continue
            if targets["paused"]:
                # 暂停时仍然要给用户明确反馈，否则看起来就像"卡死了"
                _disp.reset()            # 暂停期间积压的句子不再补播
                overlay.set_status("已暂停（F12 或菜单继续）")
                if _time.perf_counter() - _pause_warn["t"] > 5.0:
                    _pause_warn["t"] = _time.perf_counter()
                    print("（暂停中：字幕未显示，F12 继续）", flush=True)
                continue
            cid = item[0]
            pending_cues[cid] = item[1:]     # 后到的覆盖先到的
    except queue.Empty:
        pass
    now = _time.perf_counter()
    for cid, (src, dst, lang, t0) in pending_cues.items():
        tag = LANG_TAG.get(lang, "")
        src_show = f"[{tag}] {src}" if tag else src
        if dst is None:
            dst_show = None              # 译文还没出来，先显示原文
        else:
            # 注意：中文语音"不翻译"只是**当前行**的显示占位，不能写进字幕数据，
            # 否则历史行会显示成这句提示而不是原来的中文句子（见 overlay._rows）。
            dst_show = dst or ""
        act = _disp.push(cid, src_show, dst_show, lang, now)
        if act:                          # 同一句的更新（流式译文等）：立即上屏
            _cid, _s, _d, _l = act[1]
            overlay.update_cue(_cid, _s, _d, _l)
        # 译文回退检测：同一条字幕的译文**变短**了（迟到的短更新盖住完整译文）。留证用。
        # 现在两端都挡住了它：显示层只前进（overlay._set_target 规则①），数据层也只前进
        # （overlay.update_cue 前缀型回退不写回）→ 残句既不会上屏，也不会被让位到历史行时
        # 露出来。这条日志保留，用来观察回退频率（多了说明管线侧值得再查）。
        if dst:
            prev = _ui_seen.get(cid, 0)
            if prev and len(dst) < prev - 3:
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"[{datetime.now():%H:%M:%S}] warn: 译文回退 "
                                f"cid={cid} {prev}→{len(dst)} 字（已被只前进规则挡住）\n")
                except Exception:
                    pass
            _ui_seen[cid] = max(prev, len(dst))
            if len(_ui_seen) > 300:
                for k in list(_ui_seen)[:100]:
                    _ui_seen.pop(k, None)
        # 记录"管线产出 → 界面应用"的滞后，出问题时能直接看出是队列积压还是主线程忙
        lag = now - t0
        if lag > 2.0 and _time.perf_counter() - _lag_warn["t"] > 5.0:
            _lag_warn["t"] = _time.perf_counter()
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.now():%H:%M:%S}] warn: UI 应用滞后 "
                            f"{lag:.1f}s（队列积压 {ui_queue.qsize()}）\n")
            except Exception:
                pass
    # 顺序上屏：新句必须等当前行停留够 subtitle_dwell_ms 才让位（一次只上一句）
    while True:
        item = _disp.ready(now)
        if not item:
            break
        cid, src_show, dst_show, lang = item
        overlay.update_cue(cid, src_show, dst_show, lang)
    overlay.tick()          # 拖动结束后把攒下的排版补上（release 丢失也能自愈）
    if getattr(overlay, "activity_timed_out", False):
        # 管线超过 10s 没有任何活动更新 → 大概率是卡死（GPU 原生调用不返回）
        overlay.activity_timed_out = False
        if _time.perf_counter() - _spin_warn["t"] > 30.0:
            _spin_warn["t"] = _time.perf_counter()
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.now():%H:%M:%S}] warn: 管线 10s 无活动，"
                            f"转圈已清除（若长时间不出字，多半是 GPU 调用卡死，"
                            f"守护进程会在 25s 后自动重启）\n")
            except Exception:
                pass
    apply_targets()
    tray.sync(targets, overlay.bilingual, overlay.click_through)


def _ui_put(item, kind: str):
    """把消息放进 UI 队列。

    队列满时**丢最旧的一条**再放新的——最旧的中间态没有价值，最新的才重要。
    早前是"满了就静默丢弃新消息"，会把"某句的最终译文"直接丢掉，
    表现就是"日志里翻译好了、屏幕上却一直停在上一条/流式中间态"。
    同时把丢弃事件写进日志（限频），下次再出问题就有据可查。
    """
    try:
        ui_queue.put_nowait(item)
        return True
    except Exception:
        pass
    try:
        ui_queue.get_nowait()          # 丢最旧
    except Exception:
        pass
    try:
        ui_queue.put_nowait(item)
        return True
    except Exception:
        pass
    now = _time.perf_counter()
    if now - _ui_put.last_warn > 5.0:
        _ui_put.last_warn = now
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now():%H:%M:%S}] warn: UI 队列异常，丢弃 {kind}\n")
        except Exception:
            pass
    return False


_ui_put.last_warn = 0.0
_lag_warn = {"t": 0.0}          # UI 滞后告警限频
_pause_warn = {"t": 0.0}        # 暂停提示限频
_ui_seen: dict = {}             # cid -> 已应用的最长译文长度（检测译文回退）
_spin_warn = {"t": 0.0}          # 转圈超时告警限频

# 字幕顺序上屏队列：管线是突发的（一段积压后会连出 2~3 句），一次性 apply 会让
# 观众只看到最后一句、前面的只在历史里一闪而过。这里保证"按顺序 + 每句停留够久"。
from maisubtitle.display_queue import DisplayQueue  # noqa: E402

_disp = DisplayQueue(dwell_ms=float(getattr(cfg, "subtitle_dwell_ms", 900) or 900))


def on_subtitle_ui(cid, src, dst, lang, lat_ms):
    on_subtitle(cid, src, dst, lang, lat_ms)  # UI 模式也要留痕
    _ui_put((cid, src, dst, lang, _time.perf_counter()), "final")


def on_partial_ui(cid, src, lang, lat_ms):
    """部分识别结果：dst=None 表示译文还没出来（悬浮窗先显示原文占位）。"""
    _ui_put((cid, src, None, lang, _time.perf_counter()), "partial")


def on_translation_update_ui(cid, src, dst, lang):
    """流式翻译中间结果：只更新悬浮窗（按 cid 更新对应那一行），不写日志。"""
    _ui_put((cid, src, dst, lang, _time.perf_counter()), "stream")


def on_activity_ui(text):
    """管线正在做什么（识别中…/翻译中…）：悬浮窗顶部小字提示，不写日志。"""
    _ui_put(("__activity__", text, ""), "activity")


def on_drop_ui(cid):
    """管线把这一句判为音乐/幻觉/复读丢弃（或撤回）：通知悬浮窗撤掉它的行。

    实时模式下部分识别可能已经把半句推上屏了，不撤回就会留一句永远等不到
    定稿的残句（"幽灵字幕"）。
    """
    _ui_put(("__drop__", cid, ""), "drop")


pipe.on_subtitle = on_subtitle_ui
pipe.on_partial = on_partial_ui
pipe.on_translation_update = on_translation_update_ui
pipe.on_activity = on_activity_ui
pipe.on_drop = on_drop_ui

# ---- UI 卡顿看门狗：主线程 >2s 没跑事件循环就记日志（"未响应"从此有据可查）----
from PyQt6.QtCore import QObject, pyqtSignal  # noqa: E402


class _UiPinger(QObject):
    pinged = pyqtSignal()


def _start_ui_watchdog(app):
    pinger = _UiPinger()
    last = {"t": _time.perf_counter()}
    hb = {"t": 0.0}

    def _on_ping():
        """主线程心跳：刷新 watchdog 用的时间戳，并每秒写一次心跳文件。

        心跳文件给守护进程（scripts/supervisor.py）用：主线程一旦被卡住的
        GPU 调用阻塞（连 Qt 事件都处理不了），心跳就停 → 守护进程杀掉重起。
        """
        now = _time.perf_counter()
        last["t"] = now
        if now - hb["t"] >= 1.0:
            hb["t"] = now
            try:
                with open(LOGS_DIR / "heartbeat.txt", "w", encoding="utf-8") as f:
                    f.write(f"{_time.time():.0f}")
            except Exception:
                pass

    pinger.pinged.connect(_on_ping)
    timer = QTimer()
    timer.timeout.connect(pinger.pinged.emit)
    timer.start(200)                     # 主线程活着就会不停刷新 last["t"]

    def _wd_write(line: str):
        try:
            with open(LOGS_DIR / "ui_watchdog.log", "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def _watch():
        stalled_since = None             # 卡顿起点；None = 当前没卡
        win_max = 0.0                    # 本分钟内的最大停顿
        win_t = _time.perf_counter()
        while True:
            _time.sleep(0.25)
            now = _time.perf_counter()
            idle = now - last["t"]
            win_max = max(win_max, idle)
            # 每分钟落一条"最大停顿"摘要：即使进程被强行结束，也留下线索
            if now - win_t >= 60.0:
                if win_max > 1.0:
                    _wd_write(f"[{datetime.now():%H:%M:%S}] 近 60s 主线程最大停顿 "
                              f"{win_max:.1f}s（管线阶段: {last_state[0]!r}）")
                win_max = 0.0
                win_t = now
            if idle > 2.0:
                if stalled_since is None:
                    stalled_since = last["t"]
                    _wd_write(f"[{datetime.now():%H:%M:%S}] UI 主线程卡住"
                              f"（管线阶段: {last_state[0]!r}）…")
            elif stalled_since is not None:
                # 恢复时记录本次卡顿的**总时长**（旧版只记首次检出的 ~2s，会低估）
                _wd_write(f"[{datetime.now():%H:%M:%S}] UI 卡顿结束：共 "
                          f"{now - stalled_since:.1f}s（管线阶段: {last_state[0]!r}）")
                stalled_since = None

    threading.Thread(target=_watch, daemon=True, name="ui-watchdog").start()
    # 明确记一条"看门狗已启动"：如果卡死时连这条都没有，说明连启动都没跑到；
    # 若这条在、之后却没有任何"卡住"记录，说明是**整进程被冻住**（C 调用持 GIL）。
    _wd_write(f"[{datetime.now():%H:%M:%S}] 看门狗已启动（主线程卡住 >2s 会记录）")
    return pinger, timer


_wd_pinger, _wd_timer = _start_ui_watchdog(app)
timer = QTimer()
timer.timeout.connect(drain)
timer.start(60)

# ---- 全局热键：只改 targets/flags，绝不跨线程碰 Qt ----
import keyboard  # noqa: E402

for k, fn in [("f9", lambda: toggle("visible")),
              ("f10", lambda: toggle("through")),
              ("f11", lambda: actions.__setitem__("mode_cycle", True)),
              ("f12", lambda: toggle("paused")),
              ("f5", lambda: actions.__setitem__("lang_cycle", True)),
              ("f7", lambda: actions.__setitem__("screen_cycle", True)),
              ("f8", lambda: actions.__setitem__("editor", True)),
              ("f6", lambda: actions.__setitem__("export", True))]:
    # 注意：Esc **完全不绑**——游戏/视频/网页里 Esc 用途太多，
    # 之前绑"退出"会让程序凭空消失，改成"隐藏"用户照样觉得是"关掉了"。
    # 显隐用 F9，退出用托盘菜单。
    try:
        keyboard.add_hotkey(k, fn)
    except Exception as e:
        print(f"热键 {k} 注册失败: {e}")

app.aboutToQuit.connect(pipe.stop)
app.aboutToQuit.connect(shutdown_report)
app.aboutToQuit.connect(save_cfg_guarded)   # 透传字段不写回文件（见 save_preserving_cli）


def _thread_excepthook(args):
    import traceback as _tb
    try:
        with open(LOGS_DIR / "error.log", "a", encoding="utf-8") as f:
            f.write("\n[" + str(datetime.now()) + "] thread " + str(args.thread.name) + ":\n")
            _tb.print_exception(args.exc_type, args.exc_value, args.exc_traceback, file=f)
    except Exception:
        pass
    # 线程里未捕获的异常以前只进日志 —— 说给用户听：右下角通知（不再弹系统框）
    try:
        _one = "".join(_tb.format_exception_only(args.exc_type, args.exc_value)).strip()
        notify_bg(f"后台线程出错（{args.thread.name}）：{_one}\n详情见 logs/error.log")
    except Exception:
        pass


threading.excepthook = _thread_excepthook


def _excepthook(tp, val, tb):
    # windowed 模式没有控制台，未捕获错误写入 logs/error.log
    import traceback as _tb
    with open(LOGS_DIR / "error.log", "a", encoding="utf-8") as f:
        f.write("\n[" + str(datetime.now()) + "]\n")
        _tb.print_exception(tp, val, tb, file=f)
    # 未捕获异常必须让用户看见（以前只有日志 → 表现为"莫名其妙闪退/没反应"）：
    # 走右下角通知；Qt 已不可用时就只剩日志了（2026-09-18 起不再弹系统框）
    try:
        _one = "".join(_tb.format_exception_only(tp, val)).strip()
        notify_bg(f"MaiSubtitle 出错了：{_one}\n详情见 logs/error.log")
    except Exception:
        pass


sys.excepthook = _excepthook
# ---- 启动自检：缺模型时给"人话 + 可执行命令"（不阻断启动，见 maisubtitle/selfcheck.py）----
from maisubtitle import selfcheck  # noqa: E402

_model_issues = selfcheck.check(cfg)
for _m in _model_issues:
    print("|-- 模型自检: " + _m)
if _model_issues:
    # 缺模型 → 右下角通知（说清缺什么；设置窗口底部有「下载缺失的模型」按钮）
    alert_user("模型自检发现问题：\n" + "\n".join(_model_issues[:3]) +
               ("\n…还有 %d 条" % (len(_model_issues) - 3) if len(_model_issues) > 3 else ""))
else:
    print("|-- 模型自检: 齐全（" + "；".join(selfcheck.summary(cfg)) + "）")
tray.notify("已启动：播放英/日/韩语音即可出字幕\n右键字幕窗或托盘图标打开设置/退出")

print(f"悬浮窗已显示（屏 {cfg.screen}），托盘图标已就绪。F9 显隐，退出请用托盘菜单。")
sys.exit(app.exec())
