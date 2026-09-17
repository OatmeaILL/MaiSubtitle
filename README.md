# MaiSubtitle — 本地实时字幕翻译（Windows）

把系统正在播放的声音（视频 / 直播 / 游戏 / 会议）实时变成**悬浮字幕**：
环回采集 → 自动分句 → 识别（英/日/韩/中）→ 译成简体中文。
**全本地推理，音频与文本不出本机。**

---

## 一、环境要求

| 项 | 要求 | 说明 |
|---|---|---|
| 系统 | Windows 10 / 11（64 位） | 采集用的是 WASAPI 环回 |
| 显卡 | **NVIDIA 显卡 + 驱动**，建议 ≥ 6 GB 显存 | 识别/翻译都在 GPU 上跑；没 N 卡可把 `onnxruntime-gpu` 换成 `onnxruntime`，但会明显变慢 |
| 磁盘 | **约 4 GB**（必需：依赖 ~1 GB + 模型 ~3 GB）；全装可选件约 9 GB | 转换用的 HF 权重转完可删 |
| Python | **3.12**（装 uv 的话它会自动带一个） | 不需要 torch，除非要自己转换模型 |
| 网络 | 首次要下模型（脚本已走国内镜像回落） | 之后完全离线运行 |

## 二、快速开始

**新机器：双击 `安装_首次使用.bat`**（建环境 → 装依赖 → 下必需模型 → 转换翻译模型 → 自检；
缺什么补什么，重复运行不会重复下载）。装好后**双击 `启动_MaiSubtitle.bat`** 就能用。
想先看缺什么、不下载：`安装_首次使用.bat --check`。

<details>
<summary>手动三步（等价命令，供不想用 bat 的人）</summary>

```bash
cd MaiSubtitle
uv venv --python 3.12 .venv            # 没有 uv 就用：python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
.venv\Scripts\python.exe scripts/download_models.py --only whisper_turbo silero_vad qwen_1_5b_hf
.venv\Scripts\python.exe scripts/model_manager.py convert qwen_1_5b    # 转换要 torch：pip install torch --find-links https://mirrors.aliyun.com/pytorch-wheels/cpu/
```
</details>

| 启动器 | 用途 |
|---|---|
| `安装_首次使用.bat` | **新机器先跑这个**（一键装环境/依赖/模型；`--check` 只体检） |
| `启动_MaiSubtitle.bat` | **日常用这个**（无窗口 + 崩溃/卡死自动重启） |
| `启动_MaiSubtitle_控制台.bat` | 想实时看日志 / 排障（关窗即退出） |
| `启动_MaiSubtitle_调试.bat` | 排障（`--debug`，卡死 15s 自动转储线程栈） |

> 缺依赖/模型时**启动器会停下来告诉你缺什么**（不会"双击了没反应"）；
> 可选件（FireRedVAD、标点 CPU 模型、Qwen3-ASR）不装也能用，会自动回落。

---

## 三、推荐配置（默认值就是这一套）

| 项 | 推荐值 | 为什么 |
|---|---|---|
| 识别 | **Whisper large-v3-turbo** | 全项目唯一 whisper 权重；准、快、中文直出简体 |
| 翻译 | **hymt2（混元 Hy-MT2）** | 专用翻译模型，术语与语境最稳；**权重没装会自动回落 Qwen2.5-CT2** |
| VAD | **firered（FireRedVAD）** | 100+ 语言 SOTA、切句最整；缺文件自动回落 Silero |
| 单句上限 | **4 s** | **体感延迟 ≈ 段长 + 0.7s** —— 最大的旋钮（调到 7~9s 会明显变慢） |
| 断句静音 | **500 ms** | 低延迟预设 |
| 流式 / 渐进 | **realtime + 开** | 边说边出原文，译文流式补上 |
| 每句停留 | **2000 ms** | 等译文定稿再让位（不会"还没看清就被挤走"） |
| 历史行 / 字号 | **2 行 / 14 + 19** | 信息量与可读性的平衡 |
| 音乐过滤 | **15 %** | 听歌、带 BGM 也能出字（35% 会把歌全丢掉） |
| 源语言 | 看内容定（`F5` 循环） | en 英 / ja 日 / ko 韩 / zh 中（不翻译）/ auto 自动 |

一键恢复上面这套：**设置 → 分句 → 「低延迟预设」**。

---

## 四、引擎与档位（怎么选）

### 识别（设置 →「识别引擎」）

| 选项 | 模型 / 体积 | 说明 | 什么时候选 |
|---|---|---|---|
| **Whisper（推荐）** | large-v3-turbo / ~1.5 GB | 本地 GPU，支持语种自动检测；最稳 | 默认就用它 |
| Qwen3-ASR-0.6B | ONNX INT4 / ~1.3 GB | 免 torch；**不做语种检测**，要固定源语言 | 想省显存 / ORT 想少占一点 |
| HTTP 服务 | 远端 | 转发到 `/v1/audio/transcriptions`（vLLM / qwen-asr-serve / FunASR） | 已有远程识别服务 |
| 火山引擎流式识别 | 云端 | WebSocket 双向流式 + 二遍识别，标点/ITN 由云端给；**音频上传云端、按量计费** | 本地没 GPU / 想要云端准确率 |

火山的字段（设置 → 识别引擎 → 火山流式）：**① API Key**（新版控制台，只填它就够；
旧版填 App Key + Access Key）→ **② 模型名称**（`bigmodel`，一般不用改）→
**③ 资源 ID**（1.0 = `volc.bigasr.sauc.duration`，2.0 = `volc.seedasr.sauc.duration`），
端点默认 `bigmodel_async` = 双向流式优化版（一般不用改）。
填完点 **「验证连接」**：真连云端一次（本机有 `testdata/_clip_zh.wav` 就发 3 秒真语音，
能看到识别出的文字），失败会带上 logid。
不做语种检测 → **务必固定「源语言」**；术语表会作为热词随请求上传（≤50 条）。
依赖 `websocket-client`（见 requirements.txt 的可选段）。

### 翻译（设置 →「翻译引擎」）

| 选项 | 模型 | 速度 | 特点 |
|---|---|---|---|
| qwen | Qwen2.5-1.5B-CT2 | 最快（~0.22 s/句） | 实时首选，也是 hymt2 的回落目标 |
| qwen3 | Qwen3-1.7B-CT2 | 稍慢（~0.37 s/句） | 质量略优、风格更稳 |
| **hymt2（推荐）** | 混元 Hy-MT2-1.8B（PyTorch） | 慢（~1.3 s/句） | 专为翻译训练，术语/上下文最强；**需自备权重** |

> 引擎全部不可用时**只显示原文**（不翻译），绝不把英文原文当译文。

### 分句（设置 →「VAD 引擎」）

| 选项 | 特点 |
|---|---|
| **firered（推荐）** | 多语种 SOTA，切句最整（实测 8 段 / 平均 3.06 s） |
| fsmn | 备选；连续语音下句子偏长（干净人声可用） |
| silero | 切得最碎（14 段 / 1.78 s），零额外依赖，延迟最低但常断半句 |

---

## 五、使用

```
┌────────────────────────────────────────────┐
│  历史行（灰色小字，默认 2 行）                │
│  当前原文行                                  │
│  当前译文行（大号字）                ◌ 识别中 │
└────────────────────────────────────────────┘
```
- 无边框悬浮窗，**可拖动**（拖动时文字照常刷新）；托盘左键显隐，右键完整菜单。
- 右下角小转圈 = 正在识别（10 s 无活动自动清除）。

| 键 | 作用 | 键 | 作用 |
|---|---|---|---|
| `F9` | 显隐字幕窗 | `F5` | 源语言（auto→en→ja→ko→zh） |
| `F10` | 鼠标点击穿透 | `F7` | 换显示器 |
| `F11` | 双语 / 仅译文 / 仅原文 | `F8` | 术语库编辑器 |
| `F12` | 暂停识别 | `F6` | 导出本次会话 SRT |

> `Esc` **不绑定任何功能**（游戏/视频里太常用，避免误触）；退出请用托盘菜单。

**术语库**（提升人名、专有名词准确率）：`F8` 打开编辑器，保存即生效；
文件是 `glossary.csv`（`src_lang,source,aliases,target_zh,force,priority,note,use`），
一条术语能同时用于「识别前偏置 + 识别后纠错 + 翻译强制译名」。
**它是你自己的数据、不进 git**；想新建就从仓库里的 `glossary.example.csv` 复制一份改名。

**配置文件**：`config.json` 是你本机配置（**里面会存火山 API Key 等私密信息，因此不进 git**；
仓库里给的是 `config.example.json` 模板）。它由程序自己读写，删掉也会按默认值重建。

---

## 六、常见问题

| 现象 | 处理 |
|---|---|
| **双击启动器"没反应"** | 先看是不是缺依赖/模型：用 `启动_MaiSubtitle_控制台.bat` 启动，报错会直接打在窗口里；或 `安装_首次使用.bat --check` 只体检。日志：`logs/live_demo_startup.log`（启动输出）、`logs/supervisor.log`（守护层） |
| 一直不出字幕 | ① 确认有声音在放、没按 `F12` 暂停 ② 看 `logs/live_demo.log` 的 `统计:` 是否在涨 ③ 有"音乐段过滤"告警 → 把音乐过滤降到 10~20% |
| 延迟大 | 设置 → 分句 →「低延迟预设」；保持 realtime + 渐进显示 |
| 中文视频没译文 | 中文源不翻译（`F5` 切 zh 时只出原文，属预期） |
| 缺模型 / 首次启动很慢 | 启动日志的「模型自检」会写明；`安装_首次使用.bat --check` 或 `.venv\Scripts\python.exe scripts/model_manager.py list` 看补救命令 |
| 卡住 / 闪退 | GPU 原生调用的已知问题，守护进程 25s 心跳超时自动重启；日志在 `logs/` |
| 没装 torch 能用吗 | 能。识别/翻译都是 CT2 / ONNX；只有"HF 权重 → CT2"那一次转换要 torch，装 CPU 版即可：`.venv\Scripts\python.exe -m pip install torch --find-links https://mirrors.aliyun.com/pytorch-wheels/cpu/`（`安装_首次使用.bat` 会自动装） |

---

## 七、其它

**离线：视频文件 → 双语 SRT/ASS**
```bash
python scripts/file_translate.py 视频.mp4            # 自动语言 + SRT + ASS
python scripts/file_translate.py 视频.mp4 --src ja --glossary terms.csv
```

**模型工具**
```bash
.venv\Scripts\python.exe scripts/setup_first_run.py --check   # 体检 + 打印"缺什么、下一步"
.venv\Scripts\python.exe scripts/preflight.py                 # 启动前体检（启动器自动跑）
.venv\Scripts\python.exe scripts/model_manager.py list|download default|convert qwen_1_5b|delete <name>|size
.venv\Scripts\python.exe scripts/gpu_probe.py                 # GPU / CUDA / cuDNN 三级诊断
```

**许可**：代码 **MIT**（见 `LICENSE`）；模型权重不在仓库里，由脚本从各自官方源下载，
许可随上游（Whisper / Qwen / FireRedVAD / FunASR 等）。

**贡献**：欢迎提 issue。回归脚本、基准脚本与开发文档不在发布包内。