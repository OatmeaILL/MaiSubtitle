"""离线文件管线：视频/音频 → 识别 → 术语 → 翻译 → SRT/ASS。

阶段 1 交付。区别于实时管线：无延迟压力，允许整段解码、更长上下文；
识别统一走 whisper（large-v3-turbo，全项目唯一的 whisper 权重）。
"""
import time
from pathlib import Path

from . import audio, asr as asr_mod
from .config import resolve_glossary_path
from .glossary import Glossary
from .subtitles import Cue, write_srt, write_ass
from .translate import (NullMT, force_zh,
                        make_translator, substitute_terms, terms_in,
                        to_simplified)
from .vad import SileroVAD, segment_audio

TRANSLATABLE = {"en", "ja", "ko"}


def _build_glossary_prompt(glossary: Glossary | None, lang: str) -> str | None:
    if not glossary:
        return None
    # 识别侧词表（use=both/asr，含常见听错写法）；use=mt 的词条不喂给识别引擎
    terms = glossary.asr_terms(lang)
    if not terms:
        return None
    sep = "、" if lang in ("ja", "ko") else ", "
    return sep.join(terms)


def translate_file(input_path: str | Path,
                   out_prefix: str | Path | None = None,
                   src: str = "auto",
                   engine: str = "qwen",
                   asr_model: str = "large-v3-turbo",
                   glossary_path: str | None = None,
                   bilingual: bool = True,
                   formats: tuple = ("srt", "ass"),
                   offset: float = 0.0,
                   progress=None) -> list[Cue]:
    """主入口：返回 cues 并写出字幕文件。

    progress: 可选回调 fn(stage: str, i: int, n: int)。
    """
    t_start = time.perf_counter()
    input_path = Path(input_path)
    out_prefix = Path(out_prefix or input_path.with_suffix(""))
    progress = progress or (lambda *a: None)

    pcm, duration = audio.decode_media(input_path)
    progress("decode", 0, 1)

    vad = SileroVAD(threshold=0.6)
    segments = segment_audio(pcm, vad, end_silence_ms=400)  # 离线宽容些
    progress("vad", len(segments), len(segments))

    asr = asr_mod.WhisperASR(asr_model, device="cuda", compute_type="int8_float16")

    gp = resolve_glossary_path(glossary_path)
    glossary = Glossary(gp) if gp else None
    try:
        mt = make_translator(engine)          # 构造即加载（load_s 已在 __init__ 里记）
    except Exception:
        mt = NullMT()                         # 全部不可用：只出原文（NLLB 已移除）

    cues: list[Cue] = []
    lang_lock: str | None = None if src == "auto" else src
    context: list[str] = []

    import re as _re
    n = len(segments)
    for i, seg in enumerate(segments):
        piece = pcm[seg["start"]:seg["end"]]
        dur = len(piece) / 16000
        if dur < 0.6:
            continue
        # 语言决策（同实时策略）
        lang = lang_lock
        if lang is None or src == "auto":
            det, prob = asr.detect_lang(piece)
            if det in TRANSLATABLE | {"zh"} and dur >= 1.0 and prob >= 0.6:
                lang_lock = det
                lang = det
                if lang != det:
                    context.clear()
            lang = lang_lock
        if lang is None:
            continue

        prompt = _build_glossary_prompt(glossary, lang)
        text, _meta = asr.transcribe(piece, language=lang, prompt=prompt,
                                     return_confidence=True)
        text = text.strip()
        # 语种纠偏（同实时管线）：锁语言/检测错位时，中文文本会被按外语送去翻译，
        # 翻译模型收到"中文→中文"就原样回声。按文字系统直接纠正。
        from .live import _script_lang
        script = _script_lang(text)
        if script and script != lang:
            lang = script
        if len(text) < 2:
            continue

        dst = ""
        if lang in TRANSLATABLE:
            canonical = text
            if glossary:
                canonical = glossary.canonicalize(text, lang)[0]
                # 只给本句真正出现的术语（整表会让小模型把没出现的词硬塞进译文）
                mt.set_terms(terms_in(canonical, glossary.terms_for(lang)))
            zh, _ = mt.translate(canonical, lang, context=context)
            if not _re.search(r"[\u4e00-\u9fff]", zh):
                zh, _ = mt.translate(canonical, lang, context=[])
            if not _re.search(r"[\u4e00-\u9fff]", zh):
                # 仍无中文（复读原文）：换"强制中文"提示词兜一次
                zh = force_zh(mt, canonical,
                              terms=glossary.terms_for(lang) if glossary else None)
            if glossary and _re.search(r"[\u4e00-\u9fff]", zh):
                # 术语校验：句中术语没进译文 → 替换源文术语再重译一次
                missed = [(a, b) for a, b in terms_in(canonical, glossary.terms_for(lang))
                          if b not in zh]
                if missed:
                    alt, _ = mt.translate(substitute_terms(canonical, missed), lang)
                    if alt and all(b in alt for _, b in missed):
                        zh = alt
            dst = zh
            if _re.search(r"[\u4e00-\u9fff]", dst or ""):
                context.append(dst)
                context[:] = context[-3:]

        if lang not in TRANSLATABLE:
            text = to_simplified(text)  # 中文语音原文繁体→简体
        dst = to_simplified(dst)        # 译文统一简体
        cue = Cue(seg["start"] / 16000, seg["end"] / 16000, text, dst)
        cues.append(cue)
        progress("asr", i + 1, n)

    if "srt" in formats:
        write_srt(cues, str(out_prefix) + ".srt", bilingual=bilingual, offset=offset)
    if "ass" in formats:
        write_ass(cues, str(out_prefix) + ".ass", bilingual=bilingual, offset=offset)

    elapsed = time.perf_counter() - t_start
    print(f"[file_translate] {input_path.name}: {len(cues)} cues / {duration:.1f}s 音频, "
          f"耗时 {elapsed:.1f}s (RTF {elapsed / max(duration, 1e-6):.2f})")
    return cues
