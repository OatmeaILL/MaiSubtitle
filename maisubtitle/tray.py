"""系统托盘：图标 + 左键显隐 + 右键弹出与悬浮窗同源的完整菜单（定位防裁剪）。"""
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont, QCursor
from PyQt6.QtWidgets import QSystemTrayIcon, QApplication


def make_icon() -> QIcon:
    """程序化生成托盘图标：深色圆角块 + 白色"字"。"""
    pm = QPixmap(64, 64)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor(24, 26, 34, 235))
    p.setPen(QColor(90, 200, 250))
    p.drawRoundedRect(4, 4, 56, 56, 14, 14)
    p.setPen(QColor(255, 255, 255))
    p.setFont(QFont("Microsoft YaHei", 26, QFont.Weight.Bold))
    p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, "字")
    p.end()
    return QIcon(pm)


class TrayController:
    """left_click: 左键动作；menu_builder: 每次右键现场构建 QMenu（与悬浮窗同源）。"""

    def __init__(self, left_click, menu_builder):
        self.left_click = left_click
        self.menu_builder = menu_builder
        self.tray = QSystemTrayIcon(make_icon())
        self.tray.setToolTip("MaiSubtitle 实时字幕（左键显隐，右键菜单）")
        self.tray.activated.connect(self._activated)
        self.tray.show()

    def _activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.left_click()
        elif reason in (QSystemTrayIcon.ActivationReason.Context,
                        QSystemTrayIcon.ActivationReason.DoubleClick):
            self.popup_menu()

    def popup_menu(self):
        """在光标处弹出菜单，整菜单钳制在屏幕可用区内，不会被任务栏裁剪。"""
        menu = self.menu_builder()
        menu.adjustSize()
        pos = QCursor.pos()
        screen = QApplication.screenAt(pos) or QApplication.primaryScreen()
        geo = screen.availableGeometry()
        x = pos.x() - menu.width() // 2
        y = min(pos.y(), geo.bottom() - menu.height())   # 底边不越界（任务栏之上）
        x = max(geo.left(), min(x, geo.right() - menu.width()))
        y = max(geo.top(), y)
        menu.exec(QPoint(x, y))

    def sync(self, targets: dict, bilingual: bool, through: bool):
        """菜单每次现场构建即反映最新状态，无需同步。保留接口占位。"""

    def notify(self, msg: str, msecs: int = 2500):
        self.tray.showMessage("MaiSubtitle", msg,
                              QSystemTrayIcon.MessageIcon.Information, msecs)
