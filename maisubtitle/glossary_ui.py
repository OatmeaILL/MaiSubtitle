"""术语库管理界面（阶段 4）：表格编辑/增删/导入导出/命中日志/冲突提示。"""
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QFileDialog, QMessageBox, QTabWidget, QTextEdit, QHeaderView,
    QWidget, QLabel, QComboBox,
)

from .glossary import Glossary, TermEntry, _norm

COLS = ["语言", "源词", "别名(|分隔)", "中文译名", "级别", "优先级", "备注",
        "作用域"]
USE_TIP = ("词条作用域（TermEntry.use）：both=识别+翻译（默认）／"
           "asr=只进识别提示词与纠错／mt=只作翻译强制译名。"
           "以前编辑器没有这一列，任何一次保存都会把 asr/mt 词条静默重置成 both")


class GlossaryEditor(QDialog):
    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"术语库管理 — {path}")
        self.resize(860, 560)
        self.g = Glossary(path)
        # 用户**真的点过保存**置 True：调用方据此把"这个术语库"写进配置（= 自己设置过）。
        # 只打开看一眼就关掉不算 —— 否则按一次 F8 就等于启用了术语库（2026-09-18 修）。
        self.saved = False
        self._scope_filter = None         # None=全部 / "asr" / "mt" / "both"
        self._row_term_idx: list[int] = []  # 表格行 -> self.g.terms 下标（筛选下的映射）
        self._build_ui()
        self._reload_table()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        tabs = QTabWidget()
        lay.addWidget(tabs)

        # ---- 词条表 ----
        page = QWidget()
        v = QVBoxLayout(page)
        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        v.addWidget(self.table)
        # 作用域筛选（§八十四）：识别/翻译两个库的分离在此可见——
        # asr=只进识别提示词与近音纠错；mt=只作翻译强制译名；both=两边都用。
        # 筛选只影响显示，保存**绝不丢**被隐藏的词条。
        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("作用域筛选"))
        self.cmb_scope = QComboBox()
        self.cmb_scope.addItems(["全部", "识别专用（asr）", "翻译专用（mt）", "通用（both）"])
        self.cmb_scope.currentIndexChanged.connect(self._on_scope_filter)
        scope_row.addWidget(self.cmb_scope)
        scope_row.addStretch(1)
        _tip = QLabel("两个库的分离开关：asr=只进识别侧；mt=只进翻译侧；both=两边都用。\n"
                      "筛选只影响显示——保存不会丢被隐藏的词条。")
        _tip.setWordWrap(True)
        _tip.setStyleSheet("color: #888;")
        scope_row.addWidget(_tip)
        v.addLayout(scope_row)

        btns = QHBoxLayout()
        for text, fn in (("新增", self._add_row), ("删除选中", self._del_row),
                         ("保存并生效", self._save)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            btns.addWidget(b)
        btns.addStretch(1)
        for text, fn in (("导入 CSV", lambda: self._import("csv")),
                         ("导入 JSON", lambda: self._import("json")),
                         ("导出 CSV", lambda: self._export("csv")),
                         ("导出 JSON", lambda: self._export("json"))):
            b = QPushButton(text)
            b.clicked.connect(fn)
            btns.addWidget(b)
        v.addLayout(btns)
        tabs.addTab(page, "词条")

        # ---- 命中日志 ----
        page2 = QWidget()
        v2 = QVBoxLayout(page2)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        v2.addWidget(self.log_view)
        refresh = QPushButton("刷新命中日志")
        refresh.clicked.connect(self._refresh_log)
        v2.addWidget(refresh)
        tabs.addTab(page2, "命中日志")

    def _reload_table(self):
        self.g.load()
        self._fill_table()

    def _on_scope_filter(self, idx: int):
        self._scope_filter = (None, "asr", "mt", "both")[idx] if idx else None
        self._fill_table()

    def _fill_table(self):
        """按 self.g.terms 重建表格（只显示作用域筛选命中的行，行↔词条映射记进
        _row_term_idx —— 保存时被隐藏的词条原样保留，绝不丢数据）。"""
        t = self.table
        shown = [(i, term) for i, term in enumerate(self.g.terms)
                 if self._scope_filter is None or term.use == self._scope_filter]
        self._row_term_idx = [i for i, _ in shown]
        t.setRowCount(len(shown))
        # 冲突检测：同一归一化变体多目标
        seen = {}
        conflicts = set()
        for _i, term in shown:
            for variant in [term.source] + term.aliases:
                k = _norm(variant)
                if k in seen and seen[k] != term.target_zh:
                    conflicts.add(id(term))
                seen.setdefault(k, term.target_zh)
        for r, (_i, term) in enumerate(shown):
            values = [term.src_lang, term.source, "|".join(term.aliases),
                      term.target_zh, term.force, str(term.priority), term.note,
                      term.use]
            for c, val in enumerate(values):
                item = QTableWidgetItem(val)
                if id(term) in conflicts:
                    item.setBackground(QColor(120, 60, 30))
                    item.setToolTip("存在同源词多译名冲突，请调整优先级或别名")
                if COLS[c] == "作用域":
                    item.setToolTip(USE_TIP)
                t.setItem(r, c, item)

    def _from_table(self) -> list[TermEntry]:
        """解析表格为**完整**词条列表：被作用域筛选隐藏的词条原样保留。"""
        parsed: dict[int, TermEntry] = {}
        for r, tidx in enumerate(self._row_term_idx):
            def cell(c, _r=r):
                it = self.table.item(_r, c)
                return it.text().strip() if it else ""
            if not cell(1):
                continue
            try:
                priority = int(cell(5) or 2)
            except ValueError:
                # 以前直接抛 ValueError：用户在优先级列敲了非数字，整个保存/导出
                # 静默失败（只进 error.log），界面毫无指向性提示
                priority = 2
            use = (cell(7) or "both").strip().lower()
            if use not in ("both", "asr", "mt"):
                use = "both"
            parsed[tidx] = TermEntry(
                src_lang=cell(0) or "common", source=cell(1),
                aliases=[a for a in cell(2).split("|") if a],
                target_zh=cell(3), force=cell(4) or "建议",
                priority=priority, note=cell(6), use=use)
        return [parsed.get(i, orig) for i, orig in enumerate(self.g.terms)]

    def _add_row(self):
        # 筛选状态下新增的行自动带上当前作用域（"识别专用"页里新增就是 asr）
        self.g.terms.append(TermEntry(src_lang="common", source="", aliases=[],
                                      target_zh="", use=self._scope_filter or "both"))
        self._fill_table()
        self.table.setCurrentCell(self.table.rowCount() - 1, 1)

    def _del_row(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        idxs = sorted({self._row_term_idx[r] for r in rows
                       if 0 <= r < len(self._row_term_idx)}, reverse=True)
        for i in idxs:
            del self.g.terms[i]
        self._fill_table()

    def _save(self):
        self.g.terms = self._from_table()
        self.g.save()          # 写文件 → 管线靠 mtime 热更新自动生效
        self.saved = True
        self._reload_table()
        QMessageBox.information(self, "已保存",
                                f"{len(self.g.terms)} 条术语已写入，识别和翻译立即生效。")

    def _import(self, fmt: str):
        path, _ = QFileDialog.getOpenFileName(
            self, "导入术语库", "", f"{fmt.upper()} (*.{fmt})")
        if not path:
            return
        # 导入 = 把外部文件的词条**并入当前库**（保存仍写到当前库的路径）：
        # 以前直接改 self.g.path，"保存并生效"写的是导入的文件，而配置与热更新
        # 盯的还是原路径 —— 看似保存了、实际不生效，通知里显示的还是旧文件名
        try:
            imported = Glossary(path)      # 构造即 load；畸形文件在这里抛
        except Exception as e:
            QMessageBox.warning(self, "导入失败",
                                f"读不了该文件：{type(e).__name__}: {str(e)[:120]}")
            return
        self.g.terms = imported.terms
        self._fill_table()

    def _export(self, fmt: str):
        import pathlib
        path, _ = QFileDialog.getSaveFileName(
            self, "导出术语库", f"terms.{fmt}", f"{fmt.upper()} (*.{fmt})")
        if not path:
            return
        self.g.terms = self._from_table()
        old, self.g.path = self.g.path, pathlib.Path(path)
        self.g.save()
        self.g.path = old

    def _refresh_log(self):
        lines = []
        for h in reversed(self.g.hit_log[-200:]):
            if h.get("event") == "conflict":
                lines.append(f"[冲突] {h['key']}: 采用 {h['winner']}（优先级高于 {h['loser']}）")
            else:
                import time as _t
                ts = _t.strftime("%H:%M:%S", _t.localtime(h.get("time", 0)))
                lines.append(f"[{ts}] 命中 {h['input']} → {h['target']}（{h['how']}）")
        self.log_view.setPlainText("\n".join(lines) or "暂无命中记录")
