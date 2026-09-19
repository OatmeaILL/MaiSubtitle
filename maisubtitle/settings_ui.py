"""设置对话框：分页组织（识别与翻译 / 分句 / 显示 / 术语），按选择联动启用。

设计要点
  - **分 4 个页签**：以前是一长条 form，几十行挤在一起很难找。
  - **识别引擎合并成一个下拉**（2026-09-17）：whisper 只剩 large-v3-turbo 一个权重，
    再单列"Whisper 模型"没有意义；Qwen3-ASR 的模型目录改为**自动侦测**（不再手填）。
  - **按后端联动**：选 http 时才亮出地址/服务端模型名。
  - **VAD 参数联动**：FireRed 与 Silero/FSMN 的参数互斥显示（改对侧的参数不生效）。
  - **重启提示**：改到需要重启的项，保存后由主程序弹窗询问是否立即重启。
"""
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QComboBox, QSpinBox, QSlider,
    QCheckBox, QLineEdit, QPushButton, QHBoxLayout, QLabel, QFileDialog,
    QDialogButtonBox, QTabWidget, QWidget, QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QGuiApplication

from .asr import WS_BACKENDS, find_qwen_dir
from .autostart import enabled as autostart_enabled
from .config import PROJECT_ROOT

# 识别引擎：一个下拉同时表达 backend 与（whisper 唯一的）模型 ——
# 以前"选 whisper 再选 whisper 模型"两层让人迷惑，且 whisper 仅剩 large-v3-turbo。
# userData 存后端键（写配置用），显示文本给人看。
ASR_CHOICES = [
    ("Whisper large-v3-turbo（本地 GPU，支持语种自动检测）", "whisper",
     "本地 faster-whisper large-v3-turbo：默认、最稳（small/medium 已弃用并清除）"),
    ("Qwen3-ASR-0.6B（本地 ONNX；不做语种检测）", "qwen3-onnx",
     "本地 Qwen3-ASR-0.6B INT4（ONNX，免 torch）。不做语种检测 → 建议固定源语言。"),
    ("HTTP 服务（远程识别）", "http",
     "HTTP 服务：vllm serve / qwen-asr-serve / FunASR（需实现 /v1/audio/transcriptions）"),
    ("火山引擎流式识别（WebSocket，云端）", "ws",
     "火山引擎「大模型流式语音识别」：双向流式 + 二遍识别，标点/ITN 由云端给，"
     "不占本地 GPU。需要控制台的 API Key；语音会上传云端。"),
]

# 这些字段改动后要重启程序才生效（保存后主程序弹窗询问是否立即重启）
RESTART_FIELDS = {
    "engine": "翻译引擎", "asr_backend": "识别引擎", "asr_model": "识别模型",
    "asr_compute_type": "识别计算精度",
    "vad_engine": "VAD 引擎", "punct_engine": "补标点引擎",
    # 这三个都在管线启动时读一次（不重启不生效）：以前只登记了 punct_engine，
    # 改「接缝标点/每行补标点/前卷」不提示重启 —— 用户会以为改了没用
    "seam_punct": "接缝标点判定", "punct_final": "定稿补标点", "pre_roll": "前卷",
    # 分句参数全部在管线启动时读一次（_run 开头快照），不登记就会"改了没效果
    # 也没提示"—— 与"保存后立即生效"的预期相反
    "firered_min_silence_ms": "FireRed 句尾静音", "firered_max_speech_s": "FireRed 单句上限",
    "end_silence_ms": "句尾静音（Silero/FSMN）", "max_sentence_s": "单句最长（Silero/FSMN）",
    "merge_short_ms": "碎片合并（Silero/FSMN）", "min_speech_ratio": "音乐过滤门",
    "min_rms": "近静音能量门",
    "partial_interval_ms": "部分识别节奏", "idle_flush_ms": "断流收尾等待",
    "stream_mode": "流式模式", "beam_size": "识别束宽",
    "context_sentences": "翻译上下文句数", "gpu_subprocess": "GPU 子进程开关",
    "torch_external_dir": "外部 torch 目录",
    "asr_http_url": "HTTP 地址", "asr_http_model": "服务端模型名",
    "asr_ws_url": "火山流式地址", "asr_ws_model_name": "火山模型名称",
    "asr_ws_api_key": "火山 API Key",
    "asr_ws_app_key": "火山 App Key", "asr_ws_access_key": "火山 Access Key",
    "asr_ws_resource_id": "火山资源 ID",
}
ENGINES = [
    ("hymt2", "混元 Hy-MT2：专用翻译模型，术语/语境最稳（约 1.3s/句）。默认引擎，"
              "权重由 安装_首次使用.bat 下载"),
    ("qwen", "Qwen2.5-1.5B-CT2：最快（约 0.22s/句）"),
    ("qwen3", "Qwen3-1.7B-CT2：质量略优、稍慢（约 0.37s/句）"),
]
VADS = [
    ("firered", "FireRedVAD（推荐：多语种 SOTA，切句整；实测 8 段/平均 3.06s）"),
    ("fsmn", "FSMN-VAD（备选；连续语音下句子偏长）"),
    ("silero", "Silero（切得最碎：14 段/平均 1.78s，但延迟最低、零额外依赖）"),
]


def _fill_combo(combo, options, current, tips=None):
    """填充下拉并选中 current；current 不在列表里就插到最前（防保存时被悄悄改掉）。"""
    items = list(options)
    if current and current not in items:
        items.insert(0, current)
    combo.addItems(items)
    if current:
        combo.setCurrentText(current)
    if tips:
        for i, opt in enumerate(options):
            if opt in tips:
                combo.setItemData(i, tips[opt], Qt.ItemDataRole.ToolTipRole)
    return combo


def _row(form, text, widget):
    """加一行并返回 (label, widget)，便于按条件灰掉整行。"""
    label = QLabel(text)
    form.addRow(label, widget)
    return label, widget


# ---- 「下载缺失的模型」：弹独立 cmd 窗口跑（与 安装_首次使用.bat 同款）----
# 模块级：设置窗口关掉再打开也要能看到"已经在下载"，避免起第二个进程抢同一个文件。
_DL_PROC = None
_BENCH_PROC = None        # 部署跑分窗口的进程句柄（与下载窗口互不干扰，各留一个槽）


def _dl_running() -> bool:
    """是否有下载窗口正在跑。"""
    return _DL_PROC is not None and _DL_PROC.poll() is None


def _spawn_console_window(title: str, script: str, args: list,
                          done_text: str, fail_text: str,
                          tag: str) -> "subprocess.Popen | None":
    """新开一个命令窗口跑 `scripts/<script> <args…>`（看得见进度与报错）。

    为什么不用管道捕获输出（2026-09-18 用户实报后改）：tqdm 的进度条是 `\\r` 刷新的
    整行，捕获后只能显示截断的一小截；而且设置窗口一关就完全没界面了。改成新控制台后，
    进度、镜像回落、报错都摆在用户面前 —— 与 安装_首次使用.bat 的体验一致。
    bat 按项目硬约束落成 **GBK + CRLF**，并带 `PYTHONIOENCODING=gbk:replace`
    （见 HANDOVER §五：脚本往控制台打印不能用 ✔/⚠ 这类 GBK 编不出的字符）。
    进程句柄由调用方存模块级（下载 _DL_PROC / 跑分 _BENCH_PROC）：设置窗口关掉再
    打开仍能看到"正在跑"，也不会起第二个抢同一批文件。
    """
    py = Path(sys.executable)
    if py.name.lower() == "pythonw.exe":        # pythonw 没有控制台，换 python.exe
        py = py.with_name("python.exe")
    if not py.exists():
        py = Path(sys.executable)
    bat = Path(os.environ.get("TEMP", ".")) / f"maisub_{tag}_{os.getpid()}.bat"
    lines = [
        "@echo off",
        "chcp 936 >nul",
        f'cd /d "{PROJECT_ROOT}"',
        f"title {title}",
        "set PYTHONIOENCODING=gbk:replace",
        f'"{py}" "scripts\\{script}" {" ".join(args)}',
        "echo.",
        "if errorlevel 1 goto fail",
        f"echo {done_text}",
        "goto end",
        ":fail",
        f"echo {fail_text}",
        ":end",
        "echo.",
        "pause",
        # 用户按键后自删（标准技巧：goto 使 cmd 放开文件句柄再 del）——
        # 以前每次下载都在 %TEMP% 残留一个 maisub_*.bat，跨进程越积越多
        '(goto) 2>nul & del "%~f0"',
    ]
    try:
        bat.write_bytes(("\r\n".join(lines) + "\r\n").encode("gbk", "replace"))
        flags = 0x00000010 if os.name == "nt" else 0     # CREATE_NEW_CONSOLE
        return subprocess.Popen(["cmd.exe", "/c", str(bat)],
                                cwd=str(PROJECT_ROOT), creationflags=flags)
    except Exception as e:
        try:
            with open(PROJECT_ROOT / "logs" / "error.log", "a", encoding="utf-8") as f:
                f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 启动{title}失败: "
                        f"{type(e).__name__}: {e}\n")
        except Exception:
            pass
        return None


def _spawn_download_window(names: list) -> "subprocess.Popen | None":
    """新开命令窗口跑 `model_manager.py download <名字…>`（槽 = _DL_PROC）。"""
    global _DL_PROC
    if _dl_running():
        return None
    _DL_PROC = _spawn_console_window(
        "MaiSubtitle 模型下载", "model_manager.py", ["download", *names],
        "[完成] 模型已下好：回到 MaiSubtitle 重启一次即生效。",
        "[未完成] 上面就是原因；直接重跑一次即可（支持断点续传）。", "dl")
    return _DL_PROC


def _bench_running() -> bool:
    """是否有跑分窗口正在跑。"""
    return _BENCH_PROC is not None and _BENCH_PROC.poll() is None


def _spawn_bench_window(langs: str) -> "subprocess.Popen | None":
    """新开命令窗口跑部署跑分 bench_deploy.py（槽 = _BENCH_PROC）。

    结果三处可看：命令窗口（排名与推荐）、logs/bench_deploy.json（完整）、
    docs/bench_dev.json（精简对照，随 git 发布给其他用户参考）。
    """
    global _BENCH_PROC
    if _bench_running():
        return None
    args = [] if langs == "en,ja,ko,zh" else ["--langs", langs]
    _BENCH_PROC = _spawn_console_window(
        "MaiSubtitle 部署跑分", "bench_deploy.py", args,
        "[完成] 跑分结束：排名与推荐见上。重新双击 启动_MaiSubtitle.bat 启动程序。",
        "[未完成] 上面就是原因（多半是显存不够或模型缺失）；"
        "可用 --langs en,ja --mts qwen 缩小范围重跑。", "bench")
    return _BENCH_PROC


class SettingsDialog(QDialog):
    def __init__(self, cfg, overlay, screen_count: int, parent=None,
                 cli_overrides: dict | None = None, on_exit_app=None):
        super().__init__(parent)
        self.cfg = cfg
        self.overlay = overlay
        # 「部署跑分」用：跑分要独占 GPU，确认后先退出主程序再跑（live_demo 传 app.quit）
        self.on_exit_app = on_exit_app
        # {字段: (磁盘原值, 本次运行的 CLI 覆盖值)}——用户没改动这些字段时保存不写
        # 覆盖值，避免"透传启动一次"把配置里的选择改掉（见 config.save_preserving_cli）
        self._cli = cli_overrides or {}
        # 保存时填充：本次改动里"需要重启才生效"的字段（主程序据此弹窗询问重启）
        self.restart_fields: list[str] = []
        self.setWindowTitle("MaiSubtitle 设置")
        self.resize(520, 560)

        lay = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.addTab(self._tab_asr(cfg), "识别与翻译")
        tabs.addTab(self._tab_split(cfg), "分句（VAD）")
        tabs.addTab(self._tab_display(cfg, screen_count), "显示")
        tabs.addTab(self._tab_glossary(cfg), "术语与质量")
        lay.addWidget(tabs)

        # 模型状态（发布体验：一眼看出缺什么、**回落到了哪个模型**；缺了也能跑）
        self.lb_models = QLabel("")
        self.lb_models.setWordWrap(True)
        self.lb_models.setToolTip("按界面上当前的选择实时刷新；缺模型时启动日志与托盘也会提示；"
                                  "补救：scripts/model_manager.py list")
        lay.addWidget(self.lb_models)
        # 「下载缺失模型」：选到本机没有的模型时出现，点一下补下（2026-09-18）。
        # 下载**弹一个独立的 cmd 窗口**跑（与 安装_首次使用.bat 同款：新控制台 + GBK
        # 输出 + 跑完 pause），进度/报错用户都看得见、窗口关掉也不影响下载。
        # 曾经的做法是管道捕获子进程输出再截一行显示 —— 进度条是 \r 刷新的整行，
        # 截出来等于没有；设置窗口一关就完全没界面（用户实报，已改）。
        self.btn_dl = QPushButton("下载缺失的模型")
        self.btn_dl.setToolTip("把「模型状态」里标 ✘ 的那几项下齐（走 ModelScope/HF 镜像，"
                               "可断点续传；FireRedVAD / Qwen 会自动做导出/转换）\n"
                               "会弹出一个命令窗口显示实时进度")
        self.btn_dl.clicked.connect(self._download_missing)
        self.btn_dl.hide()
        self.lb_dl = QLabel("")
        self.lb_dl.setWordWrap(True)
        self.lb_dl.hide()
        row_dl = QHBoxLayout()
        row_dl.addWidget(self.btn_dl)
        row_dl.addWidget(self.lb_dl, 1)
        lay.addLayout(row_dl)
        self._dl_timer = QTimer(self)
        self._dl_timer.setInterval(1000)
        self._dl_timer.timeout.connect(self._dl_poll)
        self._dl_active = False          # 下载窗口在跑的标志（_dl_poll 据此收尾）
        # 「部署跑分」入口（2026-09-18 用户要求）：弹独立 cmd 窗口跑全组合跑分，
        # 结束后窗口里直接给排名与推荐 —— 用户在自己电脑上部署时照着选搭配。
        self.btn_bench = QPushButton("部署跑分")
        self.btn_bench.setToolTip(
            "对 3 种分句 × 2 种识别 × 3 种翻译做全组合实测（4 语种切片）。\n"
            "耗时约 15~25 分钟、GPU 会跑满；只跑本机已下载的模型。\n"
            "结束在弹出的命令窗口里看排名与推荐（程序算法，模型自评仅供参考）")
        self.btn_bench.clicked.connect(self._run_bench)
        self.lb_bench = QLabel("")
        self.lb_bench.setWordWrap(True)
        self.lb_bench.hide()
        row_bench = QHBoxLayout()
        row_bench.addWidget(self.btn_bench)
        row_bench.addWidget(self.lb_bench, 1)
        lay.addLayout(row_bench)
        # 换引擎 / 换 VAD → 状态行立刻跟着变（不用先保存）
        self.cmb_engine.currentIndexChanged.connect(self._refresh_models)
        self.cmb_vad.currentIndexChanged.connect(self._refresh_models)
        self._refresh_models()

        self.lbl_hint = QLabel("")
        self.lbl_hint.setStyleSheet("color: #d08a2a;")
        self.lbl_hint.setWordWrap(True)
        lay.addWidget(self.lbl_hint)

        note = QLabel("标「重启生效」的项：保存后会问你是否立即重启；其余项立即生效。\n"
                      "识别/翻译模型需要 models/ 下已有对应权重（Qwen3-ASR 目录自动侦测）。")
        note.setStyleSheet("color: #888;")
        lay.addWidget(note)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._save)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        self._sync_backend()
        self._sync_vad()
        self._backend_ready = True   # 之后的下拉切换才算"用户主动选择"（才弹注意事项）

    # ---------------- 页签 1：识别与翻译 ----------------
    def _tab_asr(self, cfg):
        page = QWidget()
        form = QFormLayout(page)
        self._form_asr = form          # _sync_backend 靠 labelForField 连标签一起显隐
        form.addRow(_desc("识别引擎与翻译引擎；标「重启生效」的项改完会问你是否立即重启。"))

        self.cmb_backend = QComboBox()
        for label, key, tip in ASR_CHOICES:
            self.cmb_backend.addItem(label, key)
            self.cmb_backend.setItemData(self.cmb_backend.count() - 1, tip,
                                         Qt.ItemDataRole.ToolTipRole)
        cur = str(getattr(cfg, "asr_backend", "whisper") or "whisper")
        idx = self.cmb_backend.findData(cur)
        if idx < 0:                  # 未知值：插到最前（老规矩，防保存时被悄悄改掉）
            self.cmb_backend.insertItem(0, f"{cur}（未知后端）", cur)
            idx = 0
        self.cmb_backend.setCurrentIndex(idx)
        self.cmb_backend.currentIndexChanged.connect(self._on_backend_changed)
        form.addRow("识别引擎（重启生效）", self.cmb_backend)

        # 识别计算精度：速度/质量取舍旋钮（2026-09-19 实测见 HANDOVER §八十一）
        self.cmb_ct = _fill_combo(
            QComboBox(), ["int8_float16", "float16", "int8"],
            str(getattr(cfg, "asr_compute_type", "int8_float16") or "int8_float16"),
            {"int8_float16": "推荐（默认）：权重 8 位存储、混 16 位计算。实测比 float16 "
                             "快 7~8%、最慢的那几句也更快，识别结果与 float16 一致",
             "float16": "半精度。如果觉得识别在嘈杂环境变差，换回这个试试",
             "int8": "纯 8 位。速度和 int8_float16 差不多，一般用不到"})
        form.addRow("识别计算精度（重启生效）", self.cmb_ct)

        # Qwen3-ASR 的模型目录：自动侦测（放在 models/ 下即可，不再手填路径）
        self.lb_qwen_state = QLabel("")
        self.lb_qwen_state.setWordWrap(True)
        self.lb_qwen, _ = _row(form, "  └ Qwen3-ASR 模型", self.lb_qwen_state)

        self.ed_http = QLineEdit(str(getattr(cfg, "asr_http_url", "") or ""))
        self.ed_http.setPlaceholderText("http://127.0.0.1:8000/v1/audio/transcriptions")
        self.lb_http, _ = _row(form, "  └ HTTP 地址", self.ed_http)

        self.ed_http_model = QLineEdit(str(getattr(cfg, "asr_http_model", "") or ""))
        self.ed_http_model.setPlaceholderText("服务端模型名，如 Qwen/Qwen3-ASR-0.6B（可空）")
        self.lb_http_model, _ = _row(form, "  └ 服务端模型名", self.ed_http_model)

        # 火山引擎流式识别（WebSocket）：密钥 + 资源 ID。
        # 新版控制台只要 X-Api-Key；旧版要 App Key + Access Key（两套填一套即可）。
        self.ed_ws_key = QLineEdit(str(getattr(cfg, "asr_ws_api_key", "") or ""))
        self.ed_ws_key.setPlaceholderText("控制台 → API Key 管理 → 生成的 API Key")
        self.lb_ws_key, _ = _row(form, "   ① 火山 API Key（必填）", self.ed_ws_key)

        self.ed_ws_model = QLineEdit(str(getattr(cfg, "asr_ws_model_name", "")
                                         or "bigmodel"))
        self.ed_ws_model.setPlaceholderText("bigmodel")
        self.ed_ws_model.setToolTip("请求体里的 model_name：按文档目前只有 bigmodel，"
                                    "一般不用改")
        self.lb_ws_model, _ = _row(form, "   ② 模型名称", self.ed_ws_model)

        self.ed_ws_app = QLineEdit(str(getattr(cfg, "asr_ws_app_key", "") or ""))
        self.ed_ws_app.setPlaceholderText("旧版控制台：App Key / APP ID（可空）")
        self.lb_ws_app, _ = _row(form, "  └ （旧版）App Key", self.ed_ws_app)

        self.ed_ws_acc = QLineEdit(str(getattr(cfg, "asr_ws_access_key", "") or ""))
        self.ed_ws_acc.setPlaceholderText("旧版控制台：Access Token（可空）")
        self.lb_ws_acc, _ = _row(form, "  └ （旧版）Access Key", self.ed_ws_acc)

        self.ed_ws_res = QLineEdit(str(getattr(cfg, "asr_ws_resource_id", "") or ""))
        self.ed_ws_res.setPlaceholderText("volc.bigasr.sauc.duration（1.0 小时版）")
        self.ed_ws_res.setToolTip(
            "资源 ID 要和控制台开通的服务对上：\n"
            "豆包流式识别 1.0：volc.bigasr.sauc.duration（小时版）/\n"
            "                  volc.bigasr.sauc.concurrent（并发版）\n"
            "豆包流式识别 2.0：volc.seedasr.sauc.duration / .concurrent")
        self.lb_ws_res, _ = _row(form, "   ③ 资源 ID", self.ed_ws_res)

        self.ed_ws_url = QLineEdit(str(getattr(cfg, "asr_ws_url", "") or ""))
        self.ed_ws_url.setPlaceholderText(
            "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async")
        self.ed_ws_url.setToolTip(
            "端点（一般不用改）：\n"
            "bigmodel_async = 双向流式优化版（推荐，支持二遍识别）\n"
            "bigmodel        = 双向流式\n"
            "bigmodel_nostream = 流式输入（更准更慢，15s 后才回结果）")
        self.lb_ws_url, _ = _row(form, "  └ 火山端点", self.ed_ws_url)

        self.lb_ws_note = QLabel("")
        self.lb_ws_note.setWordWrap(True)
        self.lb_ws_note.setStyleSheet("color: #8a7a3a;")
        form.addRow("", self.lb_ws_note)

        # 「验证连接」：按上面的值真连一次云端（本机有 testdata/_clip_zh.wav 就发
        # 3 秒真语音、顺带验证识别；没有素材就发静音，只验证鉴权与协议）。
        self.btn_ws_test = QPushButton("验证连接（真连云端一次）")
        self.btn_ws_test.setToolTip(
            "用上面的地址 / 密钥 / 模型名 / 资源 ID 真连一次云端：\n"
            "· 有 testdata/_clip_zh.wav → 发 3 秒真语音，能看到识别出的文字\n"
            "· 没有素材 → 发 1.2 秒静音，只验证鉴权与协议（结果里没有文字）\n"
            "失败会带上 logid —— 火山控制台查问题只认这个。")
        self.btn_ws_test.clicked.connect(self._ws_test)
        self.lb_ws_test_row, _ = _row(form, "   验证连接", self.btn_ws_test)
        self.lb_ws_test = QLabel("未验证")
        self.lb_ws_test.setWordWrap(True)
        form.addRow("", self.lb_ws_test)

        # 验证结果由工作线程写进 dict、主线程 200ms 轮询取回 —— 不用跨线程发信号：
        # 信号发到已销毁的对话框上会崩（这个对话框随时可能被关掉）。
        self._ws_probe = {"running": False, "done": False, "t0": 0.0}
        self._ws_timer = QTimer(self)
        self._ws_timer.setInterval(200)
        self._ws_timer.timeout.connect(self._ws_poll)
        for _w in (self.ed_ws_key, self.ed_ws_app, self.ed_ws_acc):
            # editingFinished（焦点离开/回车）而不是 textChanged：以前每敲一个字符
            # 就对 15+ 个 widget 重跑一遍显隐 + 重写提示样式，纯浪费；"填了没"的
            # 状态在离开输入框时刷新语义不变
            _w.editingFinished.connect(self._sync_backend)

        self.cmb_engine = QComboBox()
        _fill_combo(self.cmb_engine, [o for o, _ in ENGINES], cfg.engine, dict(ENGINES))
        form.addRow("翻译引擎（重启生效）", self.cmb_engine)

        self.cmb_srclang = QComboBox()
        _fill_combo(self.cmb_srclang, ["auto", "en", "ja", "ko", "zh"],
                    str(getattr(cfg, "source_language", "auto") or "auto"),
                    {"auto": "自动检测（默认）",
                     "en": "强制英语", "ja": "强制日语", "ko": "强制韩语",
                     "zh": "强制中文（不翻译）"})
        form.addRow("源语言（立即生效，F5 切换）", self.cmb_srclang)

        self.sp_beam = QSpinBox()
        self.sp_beam.setRange(1, 10)
        self.sp_beam.setValue(int(getattr(cfg, "beam_size", 5)))
        self.sp_beam.setToolTip("解码束宽：5 对中文/嘈杂更稳，实测不增延迟（重启生效）")
        form.addRow("识别束宽 beam_size（重启生效）", self.sp_beam)

        self.sp_ctx = QSpinBox()
        self.sp_ctx.setRange(0, 10)
        self.sp_ctx.setValue(int(getattr(cfg, "context_sentences", 5)))
        self.sp_ctx.setToolTip("翻译时带上前几句译文做上下文（Qwen 系引擎有效），0=不带")
        form.addRow("翻译上下文句数（重启生效）", self.sp_ctx)
        return page

    def _on_backend_changed(self, _idx: int = 0):
        """用户手动切换识别引擎：切到 Qwen3-ASR 时弹一次注意事项。"""
        self._sync_backend()
        self._refresh_models()
        if not getattr(self, "_backend_ready", False):
            return                      # 打开对话框时的初始化填充，不算"用户选择"
        if (self.cmb_backend.currentData() or "") == "qwen3-onnx":
            QMessageBox.information(
                self, "Qwen3-ASR 注意事项",
                "Qwen3-ASR-0.6B（本地 ONNX）注意事项：\n\n"
                "• 不做语种检测 → 建议把「源语言」固定为 en / ja / ko / zh（F5 也能切）\n"
                "• 首次加载较慢（ORT CUDA 会话），之后每段识别约 0.3~1.2s\n"
                "• 与 whisper 各有胜负：韩语长段比 whisper 慢，英文/中文接近\n"
                "• 模型目录自动侦测：需含 encoder.int4.onnx / decoder_init.int4.onnx / "
                "decoder_step.int4.onnx")
        elif (self.cmb_backend.currentData() or "") in WS_BACKENDS:
            QMessageBox.information(
                self, "火山流式识别注意事项",
                "火山引擎「大模型流式语音识别」（WebSocket 双向流式）注意事项：\n\n"
                "• **音频会上传云端**（按量计费），离线/私密场景请用本地后端\n"
                "• 必须填控制台的密钥：新版控制台只要 API Key；旧版要 App Key + "
                "Access Key（填一套即可）\n"
                "• 「资源 ID」要和已开通的服务对上（1.0 = volc.bigasr.sauc.duration；"
                "2.0 = volc.seedasr.sauc.duration）\n"
                "• 不做语种检测 → 建议把「源语言」固定为 en / ja / ko / zh（F5 也能切）；"
                "云端模型主要认中文/英文\n"
                "• 术语表会作为「热词」随请求上传（≤50 条，提示性，不保证 100% 生效）\n"
                "• 填完点下面的「验证连接」先试一次（真连云端，会告诉你成没成、识别到什么）\n"
                "• 改完要重启程序才生效")

    # ---------------- 页签 2：分句（VAD） ----------------
    def _tab_split(self, cfg):
        page = QWidget()
        form = QFormLayout(page)
        form.addRow(_desc("决定「一句」有多长：体感延迟 ≈ 段长 + 0.7s，段长是最大的旋钮；"
                          "拿不准就点上面的「低延迟预设」。"))

        btn_lowlat = QPushButton("低延迟预设（实时渐进 + 4s 单句上限 + 500ms 断句）")
        btn_lowlat.setToolTip(
            "把影响延迟的几项一次调好：\n"
            "  流式模式 = realtime、渐进显示 = 开\n"
            "  FireRed 单句上限 = 4s、断句静音 = 500ms\n"
            "体感延迟 ≈ 段长 + 0.7s；关掉渐进显示/放宽单句上限会让延迟成倍上升。")
        btn_lowlat.clicked.connect(self._preset_low_latency)
        btn_balanced = QPushButton("均衡（推荐默认）")
        btn_balanced.setToolTip(
            "恢复出厂推荐：实时渐进 + FireRed 单句上限 7s + 断句静音 500ms + 每句停留 2000ms\n"
            "（= README 默认值表）")
        btn_balanced.clicked.connect(self._preset_balanced)
        btn_quality = QPushButton("高质量（完整句优先）")
        btn_quality.setToolTip(
            "句子更完整、翻译上下文更好，但延迟明显上升：\n"
            "  FireRed 单句上限 = 10s、渐进显示 = 关（整句出）")
        btn_quality.clicked.connect(self._preset_quality)
        _row3 = QHBoxLayout()
        _row3.addWidget(btn_lowlat)
        _row3.addWidget(btn_balanced)
        _row3.addWidget(btn_quality)
        form.addRow("", _wrap(_row3))

        self.sp_partial = QSpinBox()
        self.sp_partial.setRange(300, 3000)
        self.sp_partial.setSingleStep(100)
        self.sp_partial.setSuffix(" ms")
        self.sp_partial.setValue(int(getattr(cfg, "partial_interval_ms", 800)))
        self.sp_partial.setToolTip(
            "部分识别节奏：说话期间每这么久把已说的内容识别上屏一次。\n"
            "调小=字幕长得快但 GPU 占用高（超过 3s 的长句会自动 ×2 降频）；\n"
            "调大=省 GPU 但字幕生长慢。重启生效")
        self.lb_partial, _ = _row(form, "部分识别节奏", self.sp_partial)

        self.sp_idle = QSpinBox()
        self.sp_idle.setRange(1000, 6000)
        self.sp_idle.setSingleStep(250)
        self.sp_idle.setSuffix(" ms")
        self.sp_idle.setValue(int(float(getattr(cfg, "idle_flush_ms", 2500.0))))
        self.sp_idle.setToolTip(
            "音频断流（视频暂停/过场）多久后把最后一句收尾上屏。\n"
            "实测 900ms 会频繁误切出半句，默认 2500 —— 改小前想清楚。重启生效")
        self.lb_idle, _ = _row(form, "断流收尾等待", self.sp_idle)

        self.cmb_vad = QComboBox()
        _fill_combo(self.cmb_vad, [o for o, _ in VADS], cfg.vad_engine, dict(VADS))
        self.cmb_vad.currentIndexChanged.connect(self._sync_vad)
        form.addRow("VAD 引擎（重启生效）", self.cmb_vad)

        self.sp_fr_sil = QSpinBox()
        self.sp_fr_sil.setRange(200, 2000)
        self.sp_fr_sil.setSingleStep(50)
        self.sp_fr_sil.setSuffix(" ms")
        self.sp_fr_sil.setValue(int(float(getattr(cfg, "firered_min_silence_ms", 700.0))))
        self.sp_fr_sil.setToolTip("FireRedVAD：句间静音超过这么久才断句（实测 700ms 最优）")
        self.lb_fr_sil, _ = _row(form, "  └ FireRed 断句静音", self.sp_fr_sil)

        self.sp_fr_max = QSpinBox()
        self.sp_fr_max.setRange(4, 20)
        self.sp_fr_max.setSuffix(" s")
        self.sp_fr_max.setValue(int(float(getattr(cfg, "firered_max_speech_s", 7.0))))
        self.sp_fr_max.setToolTip("FireRedVAD 单句上限：连续说话没有停顿时强制切。\n"
                                 "实测设 20s 时韩语解说出过一行等 18.7s 的字；设 7s 最长等 7.0s")
        self.lb_fr_max, _ = _row(form, "  └ FireRed 单句上限", self.sp_fr_max)

        self.sp_end_sil = QSpinBox()
        self.sp_end_sil.setRange(150, 900)
        self.sp_end_sil.setSuffix(" ms")
        self.sp_end_sil.setValue(int(getattr(cfg, "end_silence_ms", 400)))
        self.sp_end_sil.setToolTip("（仅 Silero/FSMN）句尾静音多久算说完：调大=句子更长；\n"
                                   "FireRed 引擎请用上面的「FireRed 断句静音」")
        self.lb_end_sil, _ = _row(form, "句尾静音（Silero/FSMN）", self.sp_end_sil)

        self.sp_merge = QSpinBox()
        self.sp_merge.setRange(0, 3000)
        self.sp_merge.setSingleStep(100)
        self.sp_merge.setSuffix(" ms")
        self.sp_merge.setValue(int(getattr(cfg, "merge_short_ms", 1200) or 0))
        self.sp_merge.setToolTip("（仅 Silero/FSMN）碎片合并：说得很短先不断句，"
                                 "等下一截接上再合并；0=关闭")
        self.lb_merge, _ = _row(form, "碎片合并（Silero/FSMN）", self.sp_merge)

        self.sp_max_sent = QSpinBox()
        self.sp_max_sent.setRange(4, 20)
        self.sp_max_sent.setSuffix(" s")
        self.sp_max_sent.setValue(int(getattr(cfg, "max_sentence_s", 10)))
        self.sp_max_sent.setToolTip("（仅 Silero/FSMN）单句硬上限：连续说话到这么长就强制切；\n"
                                    "FireRed 引擎请用上面的「FireRed 单句上限」")
        self.lb_max_sent, _ = _row(form, "单句最长（Silero/FSMN）", self.sp_max_sent)

        self.sp_ratio = QSpinBox()
        self.sp_ratio.setRange(0, 100)
        self.sp_ratio.setSuffix(" %")
        self.sp_ratio.setValue(int(float(getattr(cfg, "min_speech_ratio", 0.35)) * 100))
        self.sp_ratio.setToolTip(
            "音乐过滤强度：低于这个占比的段先当作可能是音乐/伴奏——仍会正常识别，\n"
            "识别后再按置信度筛掉确实是音乐的部分，所以听歌/游戏 BGM 不容易漏句。\n"
            "调高＝过滤更严；设 0＝不过滤。听歌时漏字就调低，字幕出垃圾就调高。")
        form.addRow("音乐过滤强度", self.sp_ratio)

        self.sp_rms = QSpinBox()
        self.sp_rms.setRange(0, 400)
        self.sp_rms.setValue(int(getattr(cfg, "min_rms", 60)))
        self.sp_rms.setToolTip("能量门：低于此音量视为没人说话（挡小声时的幻觉字幕）")
        form.addRow("静音门槛（RMS）", self.sp_rms)

        self.chk_seam = QCheckBox("接缝标点判定：续接段先补标点，由句读决定分/合")
        self.chk_seam.setChecked(bool(getattr(cfg, "seam_punct", True)))
        self.chk_seam.setToolTip(
            "连续说话时会撞上「单句上限」被强切，后半截紧跟着就来。\n"
            "开着：合并前让本地模型给「两截拼起来」的文本补标点——接缝处\n"
            "有句末标点说明上句已经说完，就分成两行；没有就并回一行。\n"
            "（whisper 在 4s 强切的片段上几乎不给句读，只能这样把句读补出来）\n"
            "关掉：回到老行为——只要撞上限就无条件拼成一行。\n"
            "代价：这类段多一次 Qwen 调用（实测 0.2~0.4s）。")
        form.addRow("", self.chk_seam)

        self.cmb_punct = QComboBox()
        _fill_combo(self.cmb_punct, ["auto", "cpu", "qwen"],
                    str(getattr(cfg, "punct_engine", "auto") or "auto"),
                    {"auto": "自动：中文用 CPU 小模型（1~4ms），英文等用 Qwen（0.3~0.5s）",
                     "cpu": "只用 CPU 标点模型（FunASR CT-Transformer）：1~4ms、不占 GPU，"
                            "但实测**只对中文可靠**，英文会乱插句号",
                     "qwen": "只用本地 Qwen：0.3~0.5s、占 GPU，中英日韩都能用（英文较准）"})
        self.cmb_punct.setToolTip(
            "决定「接缝标点判定」用谁来补标点。当前只喂接缝窗口（上一行尾 12 词 +\n"
            "本段前 12 词），比整句喂省一半以上。CPU 模型需要 models/punc-ct-transformer-zh-en-onnx。")
        form.addRow("  └ 补标点引擎（重启生效）", self.cmb_punct)
        self.cmb_punct.currentIndexChanged.connect(self._refresh_models)   # cpu 缺模型时提示下载

        self.chk_final = QCheckBox("每行定稿补标点：识别出的半句补上句读再上屏")
        self.chk_final.setChecked(bool(getattr(cfg, "punct_final", True)))
        self.chk_final.setToolTip(
            "whisper 在「撞上限强切」的片段上几乎不给句读（实测 4s 片段只有 1 个标点），\n"
            "屏幕上就是一串没头没尾的半句。开着：每行定稿前补一次标点。\n"
            "  · 中文走 CPU 小模型（1~4ms，不占 GPU）\n"
            "  · 英/日/韩走本地 Qwen（每句 +0.3~0.5s，占 GPU）\n"
            "补出来的结果要过「不许改词」校验（去掉标点后必须与原文一字不差），\n"
            "校验不过就用原文 —— 宁可没标点，也不能改字。\n"
            "Qwen3-ASR / 云端识别自己会给标点，用那些后端时可以关掉省时间。")
        form.addRow("", self.chk_final)

        self.chk_preroll = QCheckBox("前卷：撞上限的片段补 0.25s 前文再识别（防丢词）")
        self.chk_preroll.setChecked(bool(getattr(cfg, "pre_roll", True)))
        self.chk_preroll.setToolTip(
            "4s 上限强切会把一个词劈成两半，而 whisper 解本段时会把开头那半个\n"
            "词整个丢掉（实测「…show rover | around We've got…」里的 around 没了，\n"
            "字幕直接少一个词）。补 0.25s 前文音频再识别就能找回来，重复的部分\n"
            "按词自动去掉。关掉回到老行为。")
        form.addRow("", self.chk_preroll)

        self.cmb_stream = QComboBox()
        _fill_combo(self.cmb_stream, ["realtime", "sentence"],
                    getattr(cfg, "stream_mode", "realtime"),
                    {"realtime": "实时：边说边出识别进度，译文流式生成（重启生效）",
                     "sentence": "按句：整句说完才识别一次，更稳（重启生效）"})
        form.addRow("流式模式（重启生效）", self.cmb_stream)
        return page

    # ---------------- 页签 3：显示 ----------------
    def _tab_display(self, cfg, screen_count):
        page = QWidget()
        form = QFormLayout(page)
        form.addRow(_desc("悬浮窗外观与排版；这一页的调整保存后立即生效（不需要重启）。"))

        self.cmb_mode = QComboBox()
        _fill_combo(self.cmb_mode, ["bilingual", "target", "source"],
                    getattr(cfg, "display_mode", "bilingual"),
                    {"bilingual": "双语：原文 + 译文（默认）",
                     "target": "仅译文", "source": "仅原文（不翻译，最省资源）"})
        form.addRow("显示模式", self.cmb_mode)

        self.sp_dwell = QSpinBox()
        self.sp_dwell.setRange(0, 3000)
        self.sp_dwell.setSingleStep(100)
        self.sp_dwell.setSuffix(" ms")
        self.sp_dwell.setValue(int(float(getattr(cfg, "subtitle_dwell_ms", 900) or 0)))
        self.sp_dwell.setToolTip(
            "每句字幕在当前行至少停留多久才让位给下一句。\n"
            "管线偶发突发（连出 2~3 句）时，没有它就只能看到最后一句、\n"
            "前面的在历史行一闪而过。0 = 立即切换（旧行为）")
        form.addRow("每句最少停留", self.sp_dwell)

        self.sp_hist = QSpinBox()
        self.sp_hist.setRange(0, 5)      # 3→5（overlay 已支持，§八十二）
        self.sp_hist.setValue(int(getattr(cfg, "history_lines", 1)))
        self.sp_hist.setToolTip("悬浮窗保留的历史句行数（上一句还在翻译时也留在屏幕上）")
        form.addRow("历史行数", self.sp_hist)

        self.sp_src_font = QSpinBox()
        self.sp_src_font.setRange(8, 40)
        self.sp_src_font.setValue(cfg.font_size_src)
        form.addRow("原文字号", self.sp_src_font)

        self.sp_dst_font = QSpinBox()
        self.sp_dst_font.setRange(10, 56)
        self.sp_dst_font.setValue(cfg.font_size_dst)
        form.addRow("译文字号", self.sp_dst_font)

        self.sld_opacity = QSlider(Qt.Orientation.Horizontal)
        self.sld_opacity.setRange(30, 100)
        self.sld_opacity.setValue(int(cfg.opacity * 100))
        form.addRow("窗口不透明度 %", self.sld_opacity)

        self.sp_offset = QSpinBox()
        self.sp_offset.setRange(-5000, 5000)
        self.sp_offset.setSuffix(" ms")
        self.sp_offset.setValue(cfg.display_offset_ms)
        self.sp_offset.setToolTip("字幕延迟补偿（蓝牙耳机等音频链路延迟）")
        form.addRow("延迟补偿", self.sp_offset)

        self.sp_screen = QSpinBox()
        self.sp_screen.setRange(0, max(screen_count - 1, 0))
        self.sp_screen.setValue(cfg.screen)
        form.addRow("显示器编号", self.sp_screen)

        self.chk_progressive = QCheckBox("渐进显示：识别先出原文，译文好了再补")
        self.chk_progressive.setChecked(bool(getattr(cfg, "progressive_display", True)))
        form.addRow("", self.chk_progressive)

        self.chk_through = QCheckBox("鼠标点击穿透")
        self.chk_through.setChecked(cfg.click_through)
        form.addRow("", self.chk_through)

        self.chk_stream_tr = QCheckBox("流式翻译：译文逐字长出（关=整句一次出现）")
        self.chk_stream_tr.setChecked(bool(getattr(cfg, "stream_translation", True)))
        self.chk_stream_tr.setToolTip(
            "译文没正常出现时可以关掉它试试；改完立即生效")
        form.addRow("", self.chk_stream_tr)

        self.sp_width = QSpinBox()
        self.sp_width.setRange(50, 95)
        self.sp_width.setSuffix(" %")
        self.sp_width.setValue(int(float(getattr(cfg, "width_pct", 0.875)) * 100))
        self.sp_width.setToolTip("字幕窗宽度占屏比：越大单行能放的字越多。立即生效")
        form.addRow("字幕宽度", self.sp_width)

        self.chk_idle = QCheckBox("静默期显示『监听中 mm:ss』角标（证明程序活着）")
        self.chk_idle.setChecked(bool(getattr(cfg, "idle_notice", True)))
        form.addRow("", self.chk_idle)

        btn_resetpos = QPushButton("恢复默认位置")
        btn_resetpos.setToolTip("清除记住的位置，回到屏幕底部默认摆放（不压任务栏）")
        btn_resetpos.clicked.connect(self._reset_overlay_pos)
        form.addRow("", btn_resetpos)

        self.sp_split = QSpinBox()
        self.sp_split.setRange(0, 200)
        self.sp_split.setValue(int(getattr(cfg, "split_long_chars", 60) or 0))
        self.sp_split.setToolTip(
            "超过这么长的句按句末标点切成多行（时间按比例分摊；0=不切）。立即生效")
        form.addRow("单行字数上限", self.sp_split)
        return page

    # ---------------- 页签 4：术语与质量 ----------------
    def _tab_glossary(self, cfg):
        page = QWidget()
        form = QFormLayout(page)
        form.addRow(_desc("术语库**默认关闭**：填了路径才启用（术语会进识别提示词与翻译术语保护）。"
                          "F8 可直接打开编辑器，保存过就自动启用。"))

        self.ed_glossary = QLineEdit(cfg.glossary_path or "")
        self.ed_glossary.setPlaceholderText("留空 = 不启用术语库")
        self.ed_glossary.setToolTip(
            "术语库是 CSV/JSON 文件（源码仓里有 glossary.example.csv 样例）。\n"
            "留空 = 完全不用术语；填了才读。相对路径按项目根目录解析。")
        btn = QPushButton("浏览…")
        btn.clicked.connect(self._browse_glossary)
        row = QHBoxLayout()
        row.addWidget(self.ed_glossary)
        row.addWidget(btn)
        form.addRow("术语库文件", _wrap(row))

        self.chk_halluc = QCheckBox("丢弃经典幻觉句（如「ご視聴ありがとうございました」）")
        self.chk_halluc.setChecked(bool(getattr(cfg, "drop_hallucinations", True)))
        self.chk_halluc.setToolTip("Whisper 在音乐/静音片段爱吐这类套话；误伤真实对白时可关")
        form.addRow("", self.chk_halluc)

        self.chk_gpuproc = QCheckBox("识别/翻译放进子进程（GPU 卡死只重启它，不拖死界面）")
        self.chk_gpuproc.setChecked(bool(getattr(cfg, "gpu_subprocess", True)))
        self.chk_gpuproc.setToolTip(
            "推荐开启：模型活在工作进程里，原生卡死（access violation / CUDA 不返回）\n"
            "时只重启子进程并重载模型，悬浮窗与采集不受影响。\n"
            "关掉后回到老行为：卡死要靠守护进程杀掉整个程序。重启生效。")
        form.addRow("", self.chk_gpuproc)

        # 外部 CUDA torch（进阶）：机器上已经有别的带 CUDA 的 torch 环境时，指一下就不必
        # 再下 2.5GB。只影响走 PyTorch 的 hymt2 引擎。**重启生效**（挂载在 import torch 之前）。
        self.ed_torch_dir = QLineEdit(str(getattr(cfg, "torch_external_dir", "") or ""))
        self.ed_torch_dir.setPlaceholderText("留空 = 用本环境的 torch；可填 site-packages 或 venv 根目录")
        _btn_torch = QPushButton("浏览…")
        _btn_torch.clicked.connect(self._browse_torch_dir)
        _row_torch = QHBoxLayout()
        _row_torch.addWidget(self.ed_torch_dir)
        _row_torch.addWidget(_btn_torch)
        form.addRow("外部 torch 目录（重启生效）", _wrap(_row_torch))

        form.addRow(_desc("—— 系统 ——"))
        self.chk_autostart = QCheckBox("开机自动启动 MaiSubtitle")
        self.chk_autostart.setChecked(autostart_enabled())
        self.chk_autostart.setToolTip("写入当前用户的注册表 Run 键；勾选即生效（不用保存）。"
                                      "单实例守卫保证与手动启动不冲突")
        self.chk_autostart.toggled.connect(self._on_autostart_toggle)
        form.addRow("", self.chk_autostart)
        return page

    def _browse_glossary(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择术语库", "", "术语库 (*.csv *.json)")
        if path:
            self.ed_glossary.setText(path)

    def _browse_torch_dir(self):
        """选外部 torch 目录：**允许选 venv 根目录**（config.external_torch_site 会自动补
        Lib/site-packages），所以这里只提示"选 site-packages 或 venv 根目录都行"。"""
        path = QFileDialog.getExistingDirectory(
            self, "选择带 CUDA 版 torch 的环境目录（site-packages 或 venv 根目录）")
        if path:
            self.ed_torch_dir.setText(path)

    # ---------------- 联动 ----------------
    def _sync_backend(self, *_a):
        """按识别引擎**隐藏**无关项（用户要求：没选中对应后端就不显示那几行），
        并给出该后端的注意事项。

        ⚠ 只是把不相关字段藏起来，值仍然随 save 一起写回（隐藏 ≠ 清空）——
        用户切回原后端时之前填的密钥/地址还在。
        """
        b = self.cmb_backend.currentData() or self.cmb_backend.currentText()
        is_qwen, is_http = b == "qwen3-onnx", b == "http"
        is_ws = b in WS_BACKENDS
        form = self._form_asr

        def _show(widgets, on: bool):
            for w in widgets:
                w.setVisible(on)
                # addRow("", w) 这种无标签行要连空 label 一起藏，否则留一行空白
                lb = form.labelForField(w)
                if lb is not None:
                    lb.setVisible(on)

        _show((self.lb_qwen, self.lb_qwen_state), is_qwen)
        _show((self.lb_http, self.ed_http, self.lb_http_model, self.ed_http_model), is_http)
        _show((self.lb_ws_key, self.ed_ws_key, self.lb_ws_model, self.ed_ws_model,
               self.lb_ws_app, self.ed_ws_app,
               self.lb_ws_acc, self.ed_ws_acc, self.lb_ws_res, self.ed_ws_res,
               self.lb_ws_url, self.ed_ws_url,
               self.lb_ws_test_row, self.btn_ws_test, self.lb_ws_test,
               self.lb_ws_note), is_ws)
        self.btn_ws_test.setEnabled(is_ws and not self._ws_probe.get("running"))
        if is_qwen:
            found = find_qwen_dir(str(getattr(self.cfg, "asr_qwen_dir", "") or ""))
            if found:
                self.lb_qwen_state.setText(f"已检测到：models/{found.name}")
                self.lb_qwen_state.setStyleSheet("color: #4a8a4a;")
            else:
                self.lb_qwen_state.setText("未检测到（把 Qwen3-ASR 模型目录放进 models/ 即可）")
                self.lb_qwen_state.setStyleSheet("color: #c05a2a;")
            self.lbl_hint.setText("Qwen3-ASR 后端：不做语种检测 → 建议把「源语言」固定成"
                                  "en/ja/ko/zh（F5 也能切）")
        elif is_http:
            self.lbl_hint.setText("HTTP 后端：不做语种检测 → 建议固定「源语言」；"
                                  "服务需实现 /v1/audio/transcriptions 接口")
        elif is_ws:
            has = bool(self.ed_ws_key.text().strip()) or (
                bool(self.ed_ws_app.text().strip())
                and bool(self.ed_ws_acc.text().strip()))
            self.lb_ws_note.setText(
                "密钥已填 ✔ —— 建议点下面的「验证连接」真连一次（新版控制台只要 API Key）"
                if has else
                "还没填密钥：① 填「API Key」（新版控制台）或「App Key + Access Key」（旧版）"
                "　② 模型名称保持 bigmodel　③ 资源 ID 对齐已开通的服务"
                "　—— 填完点「验证连接」")
            self.lb_ws_note.setStyleSheet("color: #4a8a4a;" if has else "color: #c05a2a;")
            self.lbl_hint.setText("火山流式后端：不做语种检测 → 建议固定「源语言」；"
                                  "音频会上传云端（按量计费），术语表会作为热词随请求上传")
        else:
            self.lbl_hint.setText("")

    # ---------------- 火山流式识别：一键验证连接 ----------------
    def _ws_test(self):
        """「验证连接」：按界面上的值真连一次云端（网络调用放线程里，界面不冻）。

        用**界面上当前填的值**而不是磁盘配置：这样"改完先验证、再保存"的顺序才对，
        也不必为了试一次就重启程序。
        """
        if self._ws_probe.get("running"):
            return
        from .asr_ws import probe
        args = dict(url=self.ed_ws_url.text().strip(),
                    api_key=self.ed_ws_key.text().strip(),
                    app_key=self.ed_ws_app.text().strip(),
                    access_key=self.ed_ws_acc.text().strip(),
                    resource_id=self.ed_ws_res.text().strip(),
                    model_name=self.ed_ws_model.text().strip())
        self._ws_probe.update(running=True, done=False, ok=False, msg="",
                              t0=time.perf_counter())
        self.btn_ws_test.setEnabled(False)
        self.lb_ws_test.setText("验证中…（真连云端，通常几秒）")
        self.lb_ws_test.setStyleSheet("color: #8a7a3a;")

        def _work():
            try:
                res = probe(**args)
            except Exception as e:          # probe 自己不抛异常；这里只是兜底
                res = {"ok": False, "msg": f"{type(e).__name__}: {str(e)[:200]}",
                       "logid": "", "text": "", "audio": ""}
            self._ws_probe.update(res)
            self._ws_probe["running"] = False
            self._ws_probe["done"] = True

        threading.Thread(target=_work, daemon=True).start()
        self._ws_timer.start()

    def _ws_poll(self):
        """主线程轮询验证结果（见 _ws_test 的说明：不走跨线程信号）。"""
        st = self._ws_probe
        if st.get("running") and (time.perf_counter() - float(st.get("t0") or 0)) > 45:
            st.update(running=False, done=True, ok=False,
                      msg="验证超时（45s 没回结果）—— 检查网络，或看日志里的 logid")
        if not st.get("done"):
            return
        self._ws_timer.stop()
        ok = bool(st.get("ok"))
        parts = [("✔ 通过：" if ok else "✘ 失败：") + str(st.get("msg") or "")]
        if st.get("text"):
            parts.append(f"识别到：{st['text']}")
        if st.get("audio"):
            parts.append(f"音频：{st['audio']}")
        logid = str(st.get("logid") or "")
        if logid and logid != "-":
            parts.append(f"logid={logid}（报错给火山时用）")
        self.lb_ws_test.setText("　".join(parts))
        self.lb_ws_test.setStyleSheet("color: #4a8a4a;" if ok else "color: #c05a2a;")
        self.btn_ws_test.setEnabled(True)

    def _preset_low_latency(self):
        """一键低延迟：把影响体感延迟的几项一次调好。

        体感延迟 ≈ 段长 + 0.7s（实测），所以真正要压的是"段长"和"是否边听边出字"。
        """
        def _set(sp, v):
            sp.setValue(int(max(sp.minimum(), min(v, sp.maximum()))))

        self.cmb_stream.setCurrentText("realtime")       # 边说边出字
        self.chk_progressive.setChecked(True)            # 识别完先上原文
        _set(self.sp_fr_max, 4)
        _set(self.sp_fr_sil, 500)
        _set(self.sp_end_sil, 350)
        _set(self.sp_merge, 600)
        self.sp_dwell.setValue(900)
        self.lbl_hint.setText("已套用低延迟预设：实时渐进显示 + FireRed 单句上限 4s + "
                              "断句静音 500ms + 每句停留 900ms（保存后重启生效）")

    def _preset_balanced(self):
        """推荐默认 = 出厂默认值表（README 同一份口径）。"""
        def _set(sp, v):
            sp.setValue(int(max(sp.minimum(), min(v, sp.maximum()))))

        self.cmb_stream.setCurrentText("realtime")
        self.chk_progressive.setChecked(True)
        _set(self.sp_fr_max, 7)
        _set(self.sp_fr_sil, 500)
        _set(self.sp_end_sil, 400)
        _set(self.sp_merge, 1200)
        self.sp_dwell.setValue(2000)
        self.lbl_hint.setText("已套用均衡预设：实时渐进 + FireRed 单句上限 7s + "
                              "断句静音 500ms + 每句停留 2000ms（保存后重启生效）")

    def _preset_quality(self):
        """完整句优先：延迟换完整性（解说/课程类内容合适）。"""
        def _set(sp, v):
            sp.setValue(int(max(sp.minimum(), min(v, sp.maximum()))))

        self.cmb_stream.setCurrentText("realtime")
        self.chk_progressive.setChecked(False)
        _set(self.sp_fr_max, 10)
        _set(self.sp_fr_sil, 600)
        _set(self.sp_end_sil, 500)
        _set(self.sp_merge, 1200)
        self.sp_dwell.setValue(2000)
        self.lbl_hint.setText("已套用高质量预设：FireRed 单句上限 10s + 渐进显示关 "
                              "（整句出）—— 延迟会明显上升（保存后重启生效）")

    def _model_kw(self) -> dict:
        """界面上"当前选着"的引擎/后端/VAD（还没保存也算）——状态行与下载都按它算。"""
        return dict(engine=self.cmb_engine.currentText(),
                    backend=self.cmb_backend.currentData() or self.cmb_backend.currentText(),
                    vad=self.cmb_vad.currentText())

    def _refresh_models(self, *_a):
        """按**界面上当前选的值**刷新模型状态行（不必先保存；换引擎/换 VAD 会触发）。"""
        if not hasattr(self, "_dl_timer"):
            return                       # 构造过程中信号先到、状态行/下载行还没建好
        from . import selfcheck
        kw = self._model_kw()
        self.lb_models.setText("模型状态：" + "　".join(selfcheck.summary(self.cfg, **kw)))
        missing = selfcheck.missing_downloads(self.cfg, **kw)
        self.lb_models.setStyleSheet(
            "color: #4a8a4a;" if not missing else "color: #c05a2a;")
        if _dl_running():
            # 下载中（可能是别的设置窗口启动的）：禁用按钮 + 说清在哪儿看进度，
            # 否则用户会再点一次 → 两个进程抢同一个文件
            self.btn_dl.setVisible(True)
            self.btn_dl.setEnabled(False)
            self.btn_dl.setText("正在下载…")
            self.lb_dl.setVisible(True)
            self.lb_dl.setText("模型正在独立的命令窗口里下载，进度与报错都在那个窗口里看。\n"
                               "下完后关掉那个窗口，这里会自动刷新；重启程序即生效。")
            self.lb_dl.setStyleSheet("color: #d08a2a;")
            if not self._dl_timer.isActive():
                self._dl_active = True       # 不置位的话 _dl_poll 每秒空转、永远等不到收尾
                self._dl_timer.start()       # 关掉再打开的窗口也能盯到结束
            return
        self.btn_dl.setEnabled(True)
        self.btn_dl.setVisible(bool(missing))
        if missing:
            self.btn_dl.setText(f"下载缺失的模型（{len(missing)} 项）")
            self.btn_dl.setToolTip("会弹出一个命令窗口显示实时进度：\n"
                                   + "\n".join("· " + d for _, d in missing))

    def _download_missing(self):
        """弹独立命令窗口补下缺失模型（与 安装_首次使用.bat 同一套做法）。"""
        from . import selfcheck
        missing = selfcheck.missing_downloads(self.cfg, **self._model_kw())
        if not missing:
            return
        if _dl_running():
            return                       # 已有一个在跑，不重复启动
        desc = "\n".join("· " + d for _, d in missing)
        ans = QMessageBox.question(
            self, "下载缺失的模型",
            f"本机还没有这些模型：\n\n{desc}\n\n"
            "现在下载吗？会弹出一个命令窗口显示实时进度（走 ModelScope / HF 镜像，"
            "可断点续传；FireRedVAD 与 Qwen 会在下完后自动导出 / 转换）。\n"
            "下载期间程序照常用；下完重启一次即生效。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes)
        if ans != QMessageBox.StandardButton.Yes:
            return
        if _spawn_download_window([n for n, _ in missing]) is None:
            self.lb_dl.setVisible(True)
            self.lb_dl.setText("启动下载窗口失败：详见 logs/error.log；"
                               "也可以手动跑 安装_首次使用.bat --check 看缺什么。")
            self.lb_dl.setStyleSheet("color: #c05a2a;")
            return
        self._refresh_models()
        self._dl_active = True
        self._dl_timer.start()

    def _dl_poll(self):
        """轮询下载窗口是否结束（进度本身在命令窗口里看）。结束后刷新模型状态。"""
        if not self._dl_active or _dl_running():
            return
        self._dl_active = False
        self._dl_timer.stop()
        self._refresh_models()
        from . import selfcheck
        left = selfcheck.missing_downloads(self.cfg, **self._model_kw())
        self.lb_dl.setVisible(True)
        if not left:
            self.lb_dl.setText("✔ 模型已下载齐 —— 重启程序后生效")
            self.lb_dl.setStyleSheet("color: #4a8a4a;")
        else:
            self.lb_dl.setText(f"下载窗口已结束，但仍缺 {len(left)} 项："
                               "看那个窗口里的报错；重跑一次可断点续传")
            self.lb_dl.setStyleSheet("color: #c05a2a;")

    def _run_bench(self):
        """「部署跑分」入口：跑分要**独占 GPU** —— 主程序驻留的识别/翻译引擎不释放，
        跑分数据会严重失真（显存挤兑，实测识别慢 30 倍）。所以确认后：
        ① 弹独立 cmd 窗口（跑分启动本身有模型加载，不依赖本进程存活）；
        ② 关掉设置窗并退出主程序（退出码 0，守护进程按"主动退出"处理不会重启）；
        ③ 跑完用户重新启动即可。bench_deploy 里还有 GPU 占用兜底检测（等/询问）。"""
        if _bench_running():
            return
        ret = QMessageBox.question(
            self, "开始部署跑分？",
            "跑分要独占 GPU，主程序会先退出（配置自动保存）：\n\n"
            "· 点「是」后弹出跑分命令窗口，本程序退出\n"
            "· 跑分约 15~25 分钟，GPU 跑满；只测本机已下载的模型\n"
            "· 跑完后重新双击 启动_MaiSubtitle.bat 即可\n"
            "· 结果：窗口里直接给排名与推荐；完整数据 logs/bench_deploy.json\n\n"
            "继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if ret != QMessageBox.StandardButton.Yes:
            return
        if _spawn_bench_window("en,ja,ko,zh") is None:
            self.lb_bench.setVisible(True)
            self.lb_bench.setText("启动跑分窗口失败：详见 logs/error.log；也可以手动跑 "
                                  "python scripts/bench_deploy.py")
            self.lb_bench.setStyleSheet("color: #c05a2a;")
            return
        # 先关掉设置窗（finished 回调对 Rejected 不做保存），再退主程序
        self.close()
        if callable(self.on_exit_app):
            self.on_exit_app()

    def _sync_vad(self):
        fr = self.cmb_vad.currentText() == "firered"
        for w in (self.lb_fr_sil, self.sp_fr_sil, self.lb_fr_max, self.sp_fr_max):
            w.setEnabled(fr)
        # 这三个是 Silero/FSMN 分段器的参数：FireRed 有自己的静音/上限（且是智能切点），
        # 亮着会让人以为改了它就对 FireRed 生效 —— 灰掉（2026-09-17）
        for w in (self.lb_end_sil, self.sp_end_sil, self.lb_merge, self.sp_merge,
                  self.lb_max_sent, self.sp_max_sent):
            w.setEnabled(not fr)

    def _on_autostart_toggle(self, on: bool):
        from maisubtitle.autostart import set_autostart
        ok, msg = set_autostart(on)
        if not ok:
            self.chk_autostart.blockSignals(True)
            self.chk_autostart.setChecked(not on)
            self.chk_autostart.blockSignals(False)
            QMessageBox.warning(self, "开机自启设置失败", msg)

    def _reset_overlay_pos(self):
        self.cfg.overlay_pos = ""
        self.overlay.restore_default_position()
        QMessageBox.information(self, "已恢复", "字幕已回到默认位置（底部居左、不压任务栏）。")

    # ---------------- 保存 ----------------
    def _save(self):
        cfg, overlay = self.cfg, self.overlay
        # 需要重启才生效的字段：对比基准用**磁盘配置**而不是内存 cfg ——
        # CLI 透传启动时（如 --engine qwen，磁盘是 qwen3），用户没动下拉、保存时
        # _resolve 写回磁盘原值；拿内存覆盖值当基准就会凭空 diff 出"改了引擎"，
        # 误弹"需要重启才生效"（什么都没改也弹）。
        from .config import AppConfig
        disk_cfg = AppConfig.load()
        restart_before = {k: getattr(disk_cfg, k, None) for k in RESTART_FIELDS}

        def _resolve(field: str, new_value: str) -> str:
            """CLI 透传字段：用户没改选择 → 写磁盘原值；改了 → 写用户的新值。"""
            applied = self._cli.get(field)
            if applied and new_value == applied[1]:
                return applied[0]
            return new_value

        cfg.asr_backend = self.cmb_backend.currentData() or self.cmb_backend.currentText()
        cfg.asr_compute_type = self.cmb_ct.currentText()
        cfg.asr_model = "large-v3-turbo"      # 唯一保留的 whisper 权重（界面不再可选）
        cfg.asr_http_url = self.ed_http.text().strip()
        cfg.asr_http_model = self.ed_http_model.text().strip()
        cfg.asr_ws_url = self.ed_ws_url.text().strip()
        cfg.asr_ws_model_name = self.ed_ws_model.text().strip() or "bigmodel"
        cfg.asr_ws_api_key = self.ed_ws_key.text().strip()
        cfg.asr_ws_app_key = self.ed_ws_app.text().strip()
        cfg.asr_ws_access_key = self.ed_ws_acc.text().strip()
        cfg.asr_ws_resource_id = self.ed_ws_res.text().strip()
        cfg.vad_engine = self.cmb_vad.currentText()
        cfg.firered_min_silence_ms = float(self.sp_fr_sil.value())
        cfg.firered_max_speech_s = float(self.sp_fr_max.value())
        cfg.engine = _resolve("engine", self.cmb_engine.currentText())
        cfg.display_mode = self.cmb_mode.currentText()
        cfg.history_lines = self.sp_hist.value()
        cfg.subtitle_dwell_ms = float(self.sp_dwell.value())
        cfg.drop_hallucinations = self.chk_halluc.isChecked()
        cfg.gpu_subprocess = self.chk_gpuproc.isChecked()
        cfg.min_rms = float(self.sp_rms.value())
        cfg.source_language = self.cmb_srclang.currentText()
        cfg.stream_mode = self.cmb_stream.currentText()
        cfg.end_silence_ms = float(self.sp_end_sil.value())
        cfg.max_sentence_s = float(self.sp_max_sent.value())
        cfg.merge_short_ms = float(self.sp_merge.value())
        cfg.seam_punct = self.chk_seam.isChecked()
        cfg.punct_final = self.chk_final.isChecked()
        cfg.punct_engine = self.cmb_punct.currentText()
        cfg.pre_roll = self.chk_preroll.isChecked()
        cfg.min_speech_ratio = self.sp_ratio.value() / 100.0
        cfg.progressive_display = self.chk_progressive.isChecked()
        cfg.stream_translation = self.chk_stream_tr.isChecked()
        cfg.width_pct = self.sp_width.value() / 100.0
        cfg.idle_notice = self.chk_idle.isChecked()
        cfg.split_long_chars = self.sp_split.value()
        cfg.partial_interval_ms = float(self.sp_partial.value())
        cfg.idle_flush_ms = float(self.sp_idle.value())
        cfg.context_sentences = self.sp_ctx.value()
        cfg.beam_size = self.sp_beam.value()
        cfg.font_size_src = self.sp_src_font.value()
        cfg.font_size_dst = self.sp_dst_font.value()
        cfg.opacity = self.sld_opacity.value() / 100
        cfg.display_offset_ms = self.sp_offset.value()
        cfg.screen = self.sp_screen.value()
        # 双语/仅译文/仅原文**只认"显示模式"这一个来源**：以前这里还有个"双语显示"
        # 复选框，与下拉各写一个字段，结果取消勾选后屏幕照旧显示双语（因为悬浮窗
        # 看的是 display_mode），导出却按勾选框走——两边能互相矛盾。现在统一由
        # display_mode 推导，导出（export_session 的 bilingual）也跟着屏幕走。
        cfg.bilingual = (cfg.display_mode == "bilingual")
        cfg.click_through = self.chk_through.isChecked()
        cfg.glossary_path = _resolve("glossary_path", self.ed_glossary.text().strip())
        # 外部 torch 目录：保持用户填的原样（绝对路径），交给 config.mount_external_torch 解析
        cfg.torch_external_dir = self.ed_torch_dir.text().strip()
        cfg.save()
        # 立即生效项
        overlay.set_font_sizes(cfg.font_size_src, cfg.font_size_dst)
        overlay.setWindowOpacity(cfg.opacity)
        overlay.set_display_mode(cfg.display_mode)
        overlay.set_history(cfg.history_lines)
        overlay.set_click_through(cfg.click_through)
        screens = QGuiApplication.screens()
        scr = screens[min(cfg.screen, len(screens) - 1)]
        geo = scr.geometry()
        overlay.move(geo.x() + (geo.width() - overlay.width()) // 2,
                     geo.y() + geo.height() - 130)
        self.restart_fields = [k for k in RESTART_FIELDS
                               if getattr(cfg, k, None) != restart_before[k]]
        self.accept()


def _wrap(layout) -> QWidget:
    """把 QHBoxLayout 包一层 QWidget，便于当作 form 的一列。"""
    w = QWidget()
    w.setLayout(layout)
    return w


def _desc(text: str) -> QLabel:
    """页签顶部的一句说明（灰色小字，整行）。"""
    lb = QLabel(text)
    lb.setStyleSheet("color: #777;")
    lb.setWordWrap(True)
    return lb
