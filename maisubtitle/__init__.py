"""MaiSubtitle 核心包：本地实时字幕翻译。

模块：
  audio      音频捕获(WASAPI loopback) 与媒体文件解码
  vad        Silero VAD 流式分段
  asr        faster-whisper 识别与语言检测
  translate  Qwen 系翻译引擎（CT2 / Hy-MT2）与术语保护
  glossary   术语库（别名/模糊匹配/热更新/命中日志）
  subtitles  SRT / ASS 导出
  live       实时字幕管线（悬浮窗 + 控制台）
  config     路径与用户配置
"""
__version__ = "0.1.0"
