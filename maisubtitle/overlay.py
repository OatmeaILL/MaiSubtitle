"""悬浮字幕窗。

设计要点（2026-09-13 重构）：
- **宽度固定、高度自适应**：宽度沿用 min(1075, 屏宽×0.875)，高度按内容算；
  窗口**底边对齐**（保持底部位置不变，向上生长），这样右键锚点不会漂移。
- **多行**：可显示"若干条历史句 + 当前句"。每条字幕有独立 id（cue_id），
  上一句还在翻译时新句进来也不会被覆盖——译文好了会补回到它自己的那一行。
- **排版用 QFontMetrics 真实度量**（旧版是估算公式，长文本会显示不全）：
  逐档降字号直到所有行都能装进可用高度。
- **显示模式**：bilingual（双语）/ target（仅译文）/ source（仅原文）。
"""
import ctypes

from PyQt6.QtCore import (Qt, QPoint, QTimer, QRectF, QPropertyAnimation,
                          QEasingCurve)
import time
from PyQt6.QtGui import (QFont, QColor, QFontMetrics, QGuiApplication,
                         QPainter, QPen)
from PyQt6.QtWidgets import (QWidget, QApplication, QLabel, QVBoxLayout,
                             QGraphicsOpacityEffect)

GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x20
WS_EX_LAYERED = 0x80000
FAMILY = "Microsoft YaHei"

MODE_BILINGUAL = "bilingual"
MODE_TARGET = "target"
MODE_SOURCE = "source"


class _Spinner(QWidget):
    """右下角的"正在干活"转圈（替代原来占一整行的"识别中…"文字）。

    自绘一段圆弧 + 定时器推进角度，视觉上就是转圈；半透明、不抢鼠标事件。
    用自绘而不是字体图标：不依赖字体是否含 braille/emoji 字形。
    """

    def __init__(self, parent):
        super().__init__(parent)
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self.setFixedSize(18, 18)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setWindowOpacity(0.9)
        self.hide()

    def _tick(self):
        self._angle = (self._angle + 30) % 360
        self.update()

    def start(self, tip: str = ""):
        if tip:
            self.setToolTip(tip)
        if not self._timer.isActive():
            self._timer.start(80)          # 每 80ms 转 30° → 约 1 秒一圈
        self.show()
        self.raise_()

    def stop(self):
        self._timer.stop()
        self.hide()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(3.0, 3.0, float(self.width() - 6), float(self.height() - 6))
        pen = QPen(QColor(150, 200, 255, 235), 2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        # -300° 的一段弧（75%）；整段随时间旋转 → 转圈效果
        p.drawArc(rect, int(self._angle * 16), int(-300 * 16))


def _is_latin(ch: str) -> bool:
    """ASCII 字母/数字（判定"半个英文单词"用；中文不算，见 _tick_reveal）。"""
    return ch.isascii() and ch.isalnum()


def _teardown_ghost(g):
    """真正收掉残影控件（由 _drop_ghost 延后一拍调到，见那里的说明）。"""
    try:
        g.hide()
        g.deleteLater()
    except Exception:
        pass


class SubtitleOverlay(QWidget):
    def __init__(self, screen_index: int = 0,
                 font_size_src: int = 12, font_size_dst: int = 17,
                 opacity: float = 0.92):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint
                            | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.Tool
                            # 点击不激活（不抢焦点）：既不会打断用户正在用的程序，
                            # 也避开"点击取焦点"引发的窗口激活路径（曾疑似卡死诱因）。
                            # 不影响鼠标事件与拖动。
                            | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._click_through = False
        self._mode = MODE_BILINGUAL
        self._history = 1                 # 显示的历史句数
        self._dragging = False            # 用户按住窗口拖动中
        self._dirty = False               # 拖动期间被推迟的重排版
        self._activity = ""               # 实时活动提示（识别中…/翻译中…）
        self._activity_t = 0.0            # 它是什么时候设置的（超时自愈用）
        self.activity_timed_out = False   # 被超时清掉过（外面据此写一条日志）
        self._cur_h = 0                   # 当前窗口高度（避免无谓的 setFixedHeight）
        self._src_size, self._dst_size = font_size_src, font_size_dst
        self._min_src, self._min_dst = 8, 9
        self._spinner = _Spinner(self)     # 右下角"正在干活"转圈
        self._cues: list[dict] = []       # [{cid, src, dst, lang}]（旧的在前）
        # 已经上过屏的 cid：被挤出 _cues 的**旧句**若再来迟到更新，绝不能重新 append
        # —— 那会让它"复活"成当前行，把真正最新的那句挤进历史行
        #（实锤：cid=1 迟到定稿 → 当前行变成第 1 句、第 4 句掉进历史行）。
        # 量级：cid 是短字符串/int，连看几小时也就几千条，可忽略；clear() 会清空。
        self._shown_cids: set = set()
        self._status: str | None = None   # 非 None 时显示状态文案（隐藏字幕内容）
        self._padding = 10

        self._box = QVBoxLayout(self)
        self._box.setContentsMargins(12, 8, 12, 8)
        self._box.setSpacing(2)
        # 顶部弹性占位：多出来的高度全给上方 → **文字始终贴着底边**。
        # 配合 _reserved_h() 的固定预留高度，新来一句字幕时整块文字不再往上跳
        # （窗口底边对齐 + 高度随内容变 = 每来一句全屏字幕就"整体位移"一次，
        #   看久了非常累；这是"字幕变化的方式令人难受"的主要原因之一）。
        self._box.addStretch(1)
        self._labels: list[QLabel] = []   # 行池（按需显示/隐藏）
        # 动效（2026-09-17）：逐字显露的"细帧"定时器 + 换行交叉淡出的残影。
        # 显露走 overlay 自己的 16ms 心跳，因为外面 drain 是 60ms 一次——
        # 按 60ms 一格地蹦字，用户看到的正是"机械"。只有还有字要显露时它才开着。
        self._reveal_timer = QTimer(self)
        self._reveal_timer.setInterval(self._REVEAL_MS)
        self._reveal_timer.timeout.connect(self._advance)
        self._ghosts: list[QLabel] = []   # 绝对定位的残影（不进布局，不影响排版）

        screens = QGuiApplication.screens()
        self._screen_index = min(screen_index, len(screens) - 1)
        self.resize_to_screen(screens[self._screen_index])
        self.setWindowOpacity(opacity)
        self._drag_pos: QPoint | None = None

    # ---------------- 几何 ----------------
    @property
    def _max_width(self) -> int:
        screens = QGuiApplication.screens()
        geo = screens[min(self._screen_index, len(screens) - 1)].geometry()
        return min(1075, int(geo.width() * 0.875))

    @property
    def _max_content_h(self) -> int:
        """内容"目标"高度：排版缩字号时按它来算（不超过屏高 34%）。"""
        screens = QGuiApplication.screens()
        geo = screens[min(self._screen_index, len(screens) - 1)].geometry()
        return max(90, int(geo.height() * 0.34))

    @property
    def _hard_content_h(self) -> int:
        """内容硬上限：**当前行永远不能被裁**，必要时允许窗口长到屏高 50%。"""
        screens = QGuiApplication.screens()
        geo = screens[min(self._screen_index, len(screens) - 1)].geometry()
        return max(120, int(geo.height() * 0.50))

    def _content_width(self) -> int:
        return self._max_width - self._box.contentsMargins().left() \
            - self._box.contentsMargins().right() - 4

    def resize_to_screen(self, screen):
        geo = screen.geometry()
        self._width = self._max_width
        self.setFixedWidth(self._width)
        self._bottom = geo.y() + geo.height() - 140
        self._relayout()

    def move_to_screen(self, index: int):
        screens = QGuiApplication.screens()
        self._screen_index = min(index, len(screens) - 1)
        self.resize_to_screen(screens[self._screen_index])

    def cycle_screen(self):
        screens = QGuiApplication.screens()
        self._screen_index = (self._screen_index + 1) % len(screens)
        self.resize_to_screen(screens[self._screen_index])
        return self._screen_index

    # ---------------- 内容 ----------------
    def set_history(self, n: int):
        self._history = max(0, min(3, int(n)))
        self._relayout()

    def set_display_mode(self, mode: str):
        self._mode = mode if mode in (MODE_BILINGUAL, MODE_TARGET, MODE_SOURCE) \
            else MODE_BILINGUAL
        self._relayout()

    @property
    def display_mode(self) -> str:
        return self._mode

    @property
    def bilingual(self) -> bool:
        return self._mode == MODE_BILINGUAL

    def set_font_sizes(self, src: int, dst: int):
        self._src_size, self._dst_size = src, dst
        self._relayout()

    def set_opacity(self, opacity: float):
        self.setWindowOpacity(opacity)

    def set_status(self, text: str):
        """显示状态文案（启动中/等待语音…），隐藏字幕内容。"""
        self._status = text
        self._relayout()

    def update_cue(self, cid, src: str, dst: str | None, lang: str = ""):
        """新增或更新一条字幕（dst=None 表示译文还没出来）。

        译文**只前进**：迟到的短更新（流式重译的中间态、回退）如果只是已展示译文的
        一个前缀，就不写回 —— 屏上本来也不会显示它（见 _set_target 的规则①），
        但数据要是被它改短，这句被让位到历史行时读的就是这个残句
        （用户反馈："还没来得及在当前句显示，就被下一句挤到历史行了"）。
        非前缀的变短是真更正（改译/术语修正），照常写入。
        """
        for c in self._cues:
            if c["cid"] == cid:
                if src:
                    c["src"] = src
                if dst is not None:
                    old = c.get("dst")
                    if not (dst and isinstance(old, str) and old
                            and len(dst) < len(old) and old.startswith(dst)):
                        c["dst"] = dst
                if lang:
                    c["lang"] = lang
                break
        else:
            if cid in self._shown_cids:
                # 已经上过屏、又被挤出 _cues 的**旧句**：这只是迟到更新。绝不能
                # 重新 append —— 那会让它"复活"成当前行，把真正最新的那句挤进历史行
                #（实锤：cid=1 的迟到定稿让当前行变回第 1 句、第 4 句掉进历史行）。
                # 它那一行早已不在可见区（_cues 只留 history+2 条），丢掉这次更新
                # 对屏幕没有任何影响。
                return
            self._cues.append({"cid": cid, "src": src or "",
                               "dst": dst, "lang": lang})   # None = 译文还没出来
            self._shown_cids.add(cid)
            # 只保留最近若干条（历史 + 当前）
            keep = self._history + 2
            if len(self._cues) > keep:
                self._cues = self._cues[-keep:]
        self._status = None                 # 有字幕了就清掉状态文案
        self._relayout()

    def set_activity(self, text: str):
        """实时活动提示（"识别中…/翻译中…"）：**右下角小转圈**，不再占一行文字。

        与 set_status 的区别：不会被"新字幕到来"清掉，只能显式置空——
        它表达的是"现在正在做什么"。空串 = 停止转圈（隐藏）。
        """
        text = (text or "").strip()
        if text == self._activity:
            # 文案没变也要**刷新"最后活动时刻"**：管线连续报"识别中…"时若只按
            # 文案变化计时，10s 后就会被当成卡死清掉转圈（logs 里那批
            # "管线 10s 无活动，多半是 GPU 调用卡死"就是这么来的误报）。
            if text:
                self._activity_t = time.perf_counter()
            return
        self._activity = text
        self._activity_t = time.perf_counter()
        if text:
            self._spinner.start(text)
        else:
            self._spinner.stop()
        self._relayout()

    def drop_cue(self, cid):
        """撤回一条字幕（管线事后判定它是音乐/幻觉，不该上屏）。

        实时模式下"部分识别"可能已经把半句推上来，随后定稿被门控丢弃；
        不撤回就会留下一句永远等不到翻译的残句（"幽灵字幕"）。
        """
        n = len(self._cues)
        self._cues = [c for c in self._cues if c["cid"] != cid]
        if len(self._cues) != n:
            self._relayout()

    def clear(self):
        self._cues = []
        # 顺带忘掉"已上过屏"：清屏（暂停/切模式）之后重新开始，cid 可能从头计数，
        # 不清会把新句误判成"已退休的旧句"而丢掉。
        self._shown_cids.clear()
        self._status = None
        self._relayout()

    # ---------------- 排版 ----------------
    def _rows(self) -> list[tuple]:
        """要渲染的行：[(text, kind, size_base, min_size)]，kind ∈ src/dst/history。"""
        if self._status is not None:
            return [(self._status, "history", self._src_size, self._min_src)]
        if not self._cues:
            return []
        cur = self._cues[-1]
        rows = []
        # 历史行（较旧的若干条）：按当前显示模式挑该给的文本
        for c in self._cues[-(self._history + 1):-1]:
            if self._mode == MODE_SOURCE:
                text = c.get("src") or c.get("dst") or ""
            else:
                text = c.get("dst") or c.get("src") or ""
            if text:
                rows.append((text, "history", self._src_size, self._min_src))
        # 当前行
        src, dst = cur.get("src", ""), cur.get("dst")
        if self._mode == MODE_BILINGUAL:
            if src:
                rows.append((src, "src", self._src_size, self._min_src))
            if dst is None:
                rows.append(("…", "dst", self._dst_size, self._min_dst))
            elif dst:
                rows.append((dst, "dst", self._dst_size, self._min_dst))
            elif cur.get("lang") == "zh":
                # 中文语音不翻译：提示**只画在当前行**，不进字幕数据、不进历史行
                rows.append(("（中文语音，不翻译）", "dst", self._dst_size,
                             self._min_dst))
            # 其余情况（英文等翻译失败）：不显示译文行，宁可只留原文
        elif self._mode == MODE_SOURCE:
            rows.append((src or "…", "src", self._src_size, self._min_src))
        else:
            if dst:
                rows.append((dst, "dst", self._dst_size, self._min_dst))
            elif cur.get("lang") == "zh":
                rows.append(("（中文语音，不翻译）", "dst", self._dst_size,
                             self._min_dst))
            else:
                rows.append(("…", "dst", self._dst_size, self._min_dst))
        return rows

    def _label(self, i: int) -> QLabel:
        while len(self._labels) <= i:
            lbl = QLabel("")
            lbl.setWordWrap(True)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl._shown = lbl._target = ""
            self._box.addWidget(lbl)
            self._labels.append(lbl)
        return self._labels[i]

    def _style(self, lbl: QLabel, kind: str):
        if kind == "history":
            lbl.setStyleSheet("color: rgba(190,190,190,200);"
                              "background: rgba(10,10,14,130);"
                              "border-radius: 6px; padding: 1px 10px;")
        elif kind == "src":
            lbl.setStyleSheet("color: rgba(210,210,210,240);"
                              "background: rgba(10,10,14,150);"
                              "border-radius: 6px; padding: 4px 14px;")
        else:
            lbl.setStyleSheet("color: rgba(255,255,255,245);"
                              "background: rgba(10,10,14,150);"
                              "border-radius: 6px; padding: 4px 14px;")

    def _height_of(self, text: str, size: int, width: int) -> int:
        """一行内容所需高度：字体度量 + 标签自带的 CSS 上下内边距。

        内边距必须算进去——否则每行少算 8px，行数一多就会把最后一行裁掉
        （表现为"翻译只显示开头"）。
        """
        fm = QFontMetrics(QFont(FAMILY, size))
        return fm.boundingRect(0, 0, width, 2000,
                               Qt.TextFlag.TextWordWrap, text).height() + 8

    def _fit(self, rows, width, max_h):
        """逐档降字号，直到所有行都能装进可用高度（真实字体度量，不是估算）。"""
        step = 0
        while True:
            sizes = [max(mn, base - step) for _t, _k, base, mn in rows]
            total = 0
            for (text, _k, _b, _m), size in zip(rows, sizes):
                total += self._height_of(text, size, width) + 2
            if total <= max_h or step >= 20:
                return sizes, total
            step += 1

    def _reserved_h(self) -> int:
        """按"配置的槽位"预留高度：历史行 + 当前原文行 + 当前译文行，各按一行算。

        这样窗口高度基本恒定（只有长句换行才会变高），新句出现/旧句消失都不再
        改变高度 → 整块文字不会上下跳。多余的空白是透明的，不影响观感；
        文字靠 _box 顶部的 stretch 压在底边，位置也不动。
        """
        width = self._content_width()
        src_h = self._height_of("Ag", self._src_size, width) + 2
        dst_h = self._height_of("Ag", self._dst_size, width) + 2
        h = self._history * src_h
        if self._mode in (MODE_BILINGUAL, MODE_SOURCE):
            h += src_h
        if self._mode in (MODE_BILINGUAL, MODE_TARGET):
            h += dst_h
        m = self._box.contentsMargins()
        return h + m.top() + m.bottom() + self._padding

    # ---------------- 逐字显露（丝滑过渡） ----------------
    # 数据是**突发**来的：英文部分识别约 800ms 一批、译文按 token 一批、续接纠正一次
    # 换掉整句。直接 setText 就是用户说的"突然吐出一大堆英文、再吐出一大堆中文"。
    # 这里给每行加一层"只前进的显露"：屏上显示到哪由 `_shown` 记着，每帧往前补一小段。
    #   ① **绝不回退**：新目标比屏上短（流式中间态回退）→ 一个字都不动，等它长回来
    #      （"字往回缩"是用户明确说过很难受的）。
    #   ② **只前进**：目标把屏上内容当开头 → 逐字补，按词边界，不露半个英文单词。
    #   ③ **分叉就切**：与屏上没共同开头（换句）→ 直接切，不做打字机（那会很怪）；
    #      切的同时外面挂一次**交叉淡入淡出**（旧文案留残影淡出），见 _spawn_ghosts。
    _REVEAL_MS = 16      # 显露帧间隔（≈60fps）：外面 drain 是 60ms 一帧，太粗、一格一格跳
    _REVEAL_MIN = 1      # 每帧至少补这么多字（细到一个字一个字落地）
    _REVEAL_MAX = 3      # 每帧最多补这么多字：突发一大段也绝不"一帧蹦出一堆"
    _REVEAL_FRAC = 0.3   # 每帧补"剩余"的这个比例：先快后慢，尾巴自然收敛
    _XFADE_MS = 150      # 换行交叉淡入淡出时长
    _XFADE_LIFT = 10     # 残影没有行接手时向上飘的像素（"整块往上走"的观感）
    _MAX_GHOSTS = 4      # 同时在飞的残影上限（字幕来得比淡出快时的兜底）

    def _set_target(self, lbl, text: str) -> bool:
        """登记某一行"最终要显示"的文本（真正的显露在 tick 里逐帧做）。

        返回 True = 这行内容被**整段换掉**了（不是接在后面长出来的）：外面据此
        做一次交叉淡入淡出——直接把新文案硬切上去正是用户说的"很机械"。
        """
        if getattr(lbl, "_target", None) == text:
            return False
        shown = getattr(lbl, "_shown", None)
        if shown is None or shown == "":
            # 新行/清空后的行：直接给全（外面有 120ms 淡入），不做打字机
            lbl._target = lbl._shown = text
            if lbl.text() != text:
                lbl.setText(text)
            return False
        lbl._target = text
        if text.startswith(shown) or shown.startswith(text):
            return False                 # ① 延长 / 流式回退：都不动屏上已有的字
        n = 0
        for a, b in zip(shown, text):
            if a != b:
                break
            n += 1
        # ② 大部分相同（续接纠正后重译出来的译文）→ 从公共前缀接着补，看着像"译文变长了"；
        #    差得多（换句）→ 直接切。
        #    注意两个条件都要满足：短句（如 6 字的"这是译文行。"）若只看"差 ≤6 字"，
        #    任何分叉都会被判成"大部分相同"→ 把屏上的字整段抹掉重显（实测踩过）。
        if n >= len(shown) * 0.6 and (len(shown) - n) <= 6:
            lbl._shown = shown[:n]
            if lbl.text() != lbl._shown:
                lbl.setText(lbl._shown)
            return False
        lbl._shown = text                # ③ 换句：整段换掉，交给外面做交叉淡入淡出
        lbl.setText(text)
        return True

    def _tick_reveal(self, lbl) -> bool:
        """把一个还在"显露中"的行往前补一小段；有变化返回 True。"""
        tgt = getattr(lbl, "_target", "")
        shown = getattr(lbl, "_shown", "")
        if not tgt or shown == tgt or not tgt.startswith(shown):
            return False
        # 步长按"剩余量"指数收敛：一大段文字头几帧走得快、尾巴一个字一个字地落；
        # 上限 _REVEAL_MAX 是硬保证——不管一次补进来多少字，都不会一帧蹦出一堆。
        rem = len(tgt) - len(shown)
        step = min(self._REVEAL_MAX,
                   max(self._REVEAL_MIN, int(rem * self._REVEAL_FRAC) + 1))
        cut = min(len(tgt), len(shown) + step)
        # 别露出半个英文单词。⚠ 必须限定 ASCII：中日韩字符 .isalnum() 也是 True，
        # 用宽条件会一路吃到句尾（中文变成"一次跳完"，实测踩过）。
        while cut < len(tgt) and _is_latin(tgt[cut - 1]) and _is_latin(tgt[cut]):
            cut += 1
        lbl._shown = tgt[:cut]
        lbl.setText(lbl._shown)
        return True

    def _advance(self):
        """推进所有行的"逐字显露"（细帧心跳：只在还有字要显露时才开着）。"""
        busy = False
        for lbl in self._labels:
            if not lbl.isVisible():
                continue
            if getattr(lbl, "_shown", None) == getattr(lbl, "_target", None):
                continue
            try:
                self._tick_reveal(lbl)
            except Exception:
                pass          # 动效绝不允许影响字幕刷新
            if getattr(lbl, "_shown", None) != getattr(lbl, "_target", None):
                busy = True       # 还没补完：心跳继续
        if busy:
            if not self._reveal_timer.isActive():
                self._reveal_timer.start()
        elif self._reveal_timer.isActive():
            self._reveal_timer.stop()   # 静止即停，不白烧 CPU

    def _apply(self, rows, sizes, resize: bool):
        """渲染文本；resize=False 时只更新文字，不碰窗口尺寸/位置。"""
        win_before = self.pos()
        # 换行前的快照：被整段换掉的旧文案要留个"残影"做交叉淡入淡出
        snap = {}
        for lbl in self._labels:
            if lbl.isVisible() and lbl.text():
                snap[lbl] = (lbl.text(), lbl.geometry(), lbl.font(),
                             lbl.styleSheet())
        replaced: list[tuple] = []
        for i, ((text, kind, base, mn), size) in enumerate(zip(rows, sizes)):
            lbl = self._label(i)
            was_hidden = not lbl.isVisible()
            lbl.setFont(QFont(FAMILY, size, QFont.Weight.Bold if kind == "dst"
                              else QFont.Weight.Normal))
            self._style(lbl, kind)
            forked = self._set_target(lbl, text)
            lbl.setVisible(True)
            if was_hidden:
                # 新出现的行淡入（120ms）：硬切的出现方式很刺眼。
                # 行内文字更新（流式译文）不淡入 —— 那会变成一直闪。
                self._fade_in(lbl)
            elif forked and lbl in snap:
                # 整行被换掉：旧文案留残影淡出、新文案淡入，两者交叉 150ms
                replaced.append(snap[lbl])
                self._fade_in(lbl, self._XFADE_MS)
        for j in range(len(rows), len(self._labels)):
            lbl = self._labels[j]
            if lbl in snap:
                replaced.append(snap[lbl])   # 被丢掉的历史行也淡出，而不是"啪"地消失
            lbl.setVisible(False)
            lbl.setText("")
            lbl._shown = lbl._target = ""
        if not resize:
            self._spawn_ghosts(replaced, win_before)
            self._kick_reveal()
            return
        width = self._content_width()
        h = sum(self._height_of(t, s, width) + 2 for (t, _k, _b, _m), s
                in zip(rows, sizes)) if rows else 60
        h += self._box.contentsMargins().top() + self._box.contentsMargins().bottom()
        # 高度取"内容需要"与"槽位预留"的较大者：预留让新句出现时高度不变
        # → 配合顶部 stretch，文字原地换内容而不是整块上下位移。
        want_h = max(60, min(int(self._hard_content_h),
                             max(int(h) + self._padding, self._reserved_h())))
        # 只在高度真的变了才动窗口（流式每秒几十次更新，无谓的 setFixedHeight/move
        # 在 Windows 上会引发大量重合成）
        if abs(want_h - self._cur_h) > 2:
            self._cur_h = want_h
            self.setFixedHeight(want_h)
        y = self._bottom - self.height()
        if self.y() != y:
            self.move(self.x(), y)
        self._place_spinner()
        self._spawn_ghosts(replaced, win_before)
        self._kick_reveal()

    def _kick_reveal(self):
        """有行还没补完就立刻起"细帧心跳"——别等外面 60ms 一次的 tick（那是延迟）。"""
        if self._reveal_timer.isActive():
            return
        for lbl in self._labels:
            if lbl.isVisible() and getattr(lbl, "_shown", None) != getattr(lbl, "_target", None):
                self._reveal_timer.start()
                return

    def _fade_in(self, lbl, ms: int = 120):
        """给一行加一次淡入（新行 120ms / 换行交叉淡入 150ms）。

        动画引用挂在 label 上，否则被 GC 掉不播。
        """
        try:
            eff = lbl.graphicsEffect()
            if not isinstance(eff, QGraphicsOpacityEffect):
                eff = QGraphicsOpacityEffect(lbl)
                lbl.setGraphicsEffect(eff)
            anim = QPropertyAnimation(eff, b"opacity", lbl)
            anim.setDuration(ms)
            anim.setStartValue(0.0)
            anim.setEndValue(1.0)
            anim.start()
            lbl._fade = anim
        except Exception:
            pass          # 动效绝不能影响字幕刷新

    # ---------------- 换行交叉淡入淡出 / 上移感 ----------------
    def _owner_row(self, text: str):
        """新排版里哪一行"接手"了这段旧文案？（残影据此上移；认不出就 None）

        只在**真有包含关系**（前缀关系）时才算接手，且至少 4 个字：
        「上一句译文」从当前行挪进历史行，就是这种情况——残影顺势滑上去，
        看着就是"整块内容往上走了一格"，而不是原地闪一下换掉。
        """
        if len(text) < 4:
            return None
        best, best_n = None, 0
        for j, row in enumerate(self._rows()):
            t = row[0]
            if not t or not (t.startswith(text) or text.startswith(t)):
                continue
            n = min(len(text), len(t))
            if n > best_n:
                best, best_n = j, n
        return best

    def _spawn_ghosts(self, replaced, win_before):
        """给"被换掉"的旧文案做残影：原地淡出；有行接手就滑过去。"""
        if not replaced:
            return
        # 先让新排版立刻生效：残影要知道自己该滑到哪儿（只在换行那几帧做一次）
        try:
            self._box.activate()
        except Exception:
            pass
        dx, dy = win_before.x() - self.x(), win_before.y() - self.y()
        for old_text, geom, font, style in replaced[-self._MAX_GHOSTS:]:
            target = None
            j = self._owner_row(old_text)
            if j is not None and j < len(self._labels) \
                    and self._labels[j].isVisible():
                target = self._labels[j].geometry().topLeft()
            self._slide_out(old_text, geom.translated(dx, dy), font, style, target)

    def _slide_out(self, text, geom, font, style, target):
        """残影：旧文案留在原位淡出；有行接手就滑向它（"整块往上走"的观感）。

        残影是**绝对定位**的独立子控件，不进布局——所以它怎么飘都不会动到
        真正的字幕排版（不裁字、不跳行）。
        """
        try:
            while len(self._ghosts) >= self._MAX_GHOSTS:
                self._drop_ghost(self._ghosts[0])
            g = QLabel(text, self)
            g.setWordWrap(True)
            g.setAlignment(Qt.AlignmentFlag.AlignCenter)
            g.setFont(font)
            g.setStyleSheet(style)
            g.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            g.setGeometry(geom)
            eff = QGraphicsOpacityEffect(g)
            g.setGraphicsEffect(eff)
            fade = QPropertyAnimation(eff, b"opacity", g)
            fade.setDuration(self._XFADE_MS)
            fade.setStartValue(1.0)
            fade.setEndValue(0.0)
            if target is None:
                target = geom.topLeft() - QPoint(0, self._XFADE_LIFT)
            if target != geom.topLeft():
                move = QPropertyAnimation(g, b"pos", g)
                move.setDuration(self._XFADE_MS)
                move.setStartValue(geom.topLeft())
                move.setEndValue(target)
                move.setEasingCurve(QEasingCurve.Type.OutCubic)
                move.start()
                g._move = move
            fade.finished.connect(lambda: self._drop_ghost(g))
            g._fade = fade
            self._ghosts.append(g)
            g.show()
            g.raise_()
            fade.start()
        except Exception:
            pass          # 动效绝不能影响字幕刷新

    def _drop_ghost(self, g):
        """收掉一个残影（从淡出动画的 finished 里调）。

        ⚠ 隐藏/销毁**必须延后一拍**：直接在 finished 信号里 hide()+deleteLater()
        会踩到 Qt 动画内部（离屏跑回归实测整进程 ACCESS_VIOLATION）。
        """
        try:
            self._ghosts.remove(g)
        except ValueError:
            pass
        QTimer.singleShot(0, lambda: _teardown_ghost(g))

    def _place_spinner(self):
        """把转圈固定在内容区右下角（绝对定位，不参与布局，不影响排版高度）。"""
        if not self._spinner.isVisible():
            return
        m = self._box.contentsMargins()
        self._spinner.move(self.width() - m.right() - self._spinner.width(),
                           self.height() - m.bottom() - self._spinner.height())

    def _relayout(self):
        rows = self._rows()
        width = self._content_width()
        # 顺序很重要：**先丢最旧的历史行，再考虑缩字号**。
        # 旧顺序（先缩字号）会让行数一变字号就跳——屏幕上的字忽大忽小，比丢一行历史
        # 难看得多。缩字号只作兜底（长句确实装不下时）。
        sizes = [base for _t, _k, base, _mn in rows]
        total = sum(self._height_of(t, s, width) + 2 for (t, _k, _b, _m), s
                    in zip(rows, sizes)) if rows else 0
        dropped = 0
        while total > self._max_content_h and dropped < self._history and len(rows) > 2:
            rows = rows[1:]                     # 当前行永不丢
            dropped += 1
            sizes = [base for _t, _k, base, _mn in rows]
            total = sum(self._height_of(t, s, width) + 2 for (t, _k, _b, _m), s
                        in zip(rows, sizes))
        if total > self._max_content_h:
            sizes, total = self._fit(rows, width, self._max_content_h)   # 兜底
        # 用户正按着鼠标（拖动中）→ **只冻结窗口尺寸/位置**，文字照常刷新；
        # 判断用真实按键状态，绝不用可能"卡住"的自有标志位。
        if QApplication.mouseButtons() & Qt.MouseButton.LeftButton:
            self._dirty = True
            try:
                self._apply(rows, sizes, resize=False)
            except Exception:
                pass
            return
        self._dirty = False
        try:
            self._apply(rows, sizes, resize=True)
        except Exception:
            pass          # 排版异常绝不允许穿透到 Qt 事件循环

    def tick(self):
        """由 UI 定时器每次调用：推进"逐字显露"+ 补排版 + 活动提示超时自愈。"""
        # 逐字显露：把突发来的一整段文字摊到多帧里显现（见 _set_target 与 _REVEAL_*）。
        # 这里推进一次；更细的帧由 overlay 自己的 _reveal_timer（16ms）补足。
        self._advance()
        # 转圈超时自愈：管线卡死（原生调用不返回）时，activity 永远清不掉，
        # 用户看到的就是"右下角一直转圈、也不再出字"。超过 10s 没更新就清掉，
        # 免得让人误以为还在干活（真正的原因由看门狗/守护进程的日志说明）。
        if self._activity and (time.perf_counter() - self._activity_t) > 10.0:
            self._activity = ""
            self.activity_timed_out = True
            self._spinner.stop()
            self._relayout()
        if self._dirty:
            self._relayout()

    # ---------------- 交互 ----------------
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self._dragging = True

    def mouseMoveEvent(self, e):
        if self._drag_pos is not None and e.buttons() & Qt.MouseButton.LeftButton:
            self.move(e.globalPosition().toPoint() - self._drag_pos)
            self._bottom = self.y() + self.height()

    def mouseReleaseEvent(self, e):
        self._drag_pos = None
        if self._dragging:
            self._dragging = False
            self._relayout()          # 应用拖动期间攒下的排版变化

    def set_click_through(self, on: bool):
        hwnd = int(self.winId())
        user32 = ctypes.windll.user32
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        if on:
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE,
                                  style | WS_EX_LAYERED | WS_EX_TRANSPARENT)
        else:
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style & ~WS_EX_TRANSPARENT)
        self._click_through = on

    @property
    def click_through(self) -> bool:
        return self._click_through
