"""术语库管理界面（阶段 4）：表格编辑/增删/导入导出/命中日志/冲突提示。"""
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QFileDialog, QMessageBox, QTabWidget, QTextEdit, QHeaderView,
    QWidget,
)

from .glossary import Glossary, TermEntry

COLS = ["语言", "源词", "别名(|分隔)", "中文译名", "级别", "优先级", "备注"]


class GlossaryEditor(QDialog):
    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"术语库管理 — {path}")
        self.resize(860, 560)
        self.g = Glossary(path)
        # 用户**真的点过保存**置 True：调用方据此把"这个术语库"写进配置（= 自己设置过）。
        # 只打开看一眼就关掉不算 —— 否则按一次 F8 就等于启用了术语库（2026-09-18 修）。
        self.saved = False
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
        t = self.table
        t.setRowCount(len(self.g.terms))
        # 冲突检测：同一归一化变体多目标
        seen = {}
        conflicts = set()
        for term in self.g.terms:
            for variant in [term.source] + term.aliases:
                from .glossary import _norm
                k = _norm(variant)
                if k in seen and seen[k] != term.target_zh:
                    conflicts.add(id(term))
                seen.setdefault(k, term.target_zh)
        for r, term in enumerate(self.g.terms):
            values = [term.src_lang, term.source, "|".join(term.aliases),
                      term.target_zh, term.force, str(term.priority), term.note]
            for c, val in enumerate(values):
                item = QTableWidgetItem(val)
                if id(term) in conflicts:
                    item.setBackground(QColor(120, 60, 30))
                    item.setToolTip("存在同源词多译名冲突，请调整优先级或别名")
                t.setItem(r, c, item)

    def _from_table(self) -> list[TermEntry]:
        terms = []
        for r in range(self.table.rowCount()):
            def cell(c):
                it = self.table.item(r, c)
                return it.text().strip() if it else ""
            if not cell(1):
                continue
            terms.append(TermEntry(
                src_lang=cell(0) or "common", source=cell(1),
                aliases=[a for a in cell(2).split("|") if a],
                target_zh=cell(3), force=cell(4) or "建议",
                priority=int(cell(5) or 2), note=cell(6)))
        return terms

    def _add_row(self):
        self.table.insertRow(self.table.rowCount())

    def _del_row(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def _save(self):
        self.g.terms = self._from_table()
        self.g.save()          # 写文件 → 管线靠 mtime 热更新自动生效
        self.saved = True
        self._reload_table()
        QMessageBox.information(self, "已保存",
                                f"{len(self.g.terms)} 条术语已写入，识别/翻译管线实时生效。")

    def _import(self, fmt: str):
        import pathlib
        path, _ = QFileDialog.getOpenFileName(
            self, "导入术语库", "", f"{fmt.upper()} (*.{fmt})")
        if not path:
            return
        self.g.path = pathlib.Path(path)
        self.g.load()
        self._reload_table()

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
