"""实时字幕管线：loopback → VAD → 锁语言 ASR → 翻译 → 回调输出。

阶段 2 交付。UI 无关（on_subtitle 回调），内置端到端延迟计量
（语音段结束墙钟时刻 → 字幕产出时刻）。
"""
import queue
import re
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from . import audio
from .asr import WhisperASR, make_asr, TRANSLATABLE, WS_BACKENDS
from .config import AppConfig, register_nvidia_dlls
from .glossary import Glossary
from .translate import (NullMT, force_zh, substitute_terms, terms_in,
                        to_simplified, translator_candidates)
from .vad import StreamingSegmenter

LANG_TAG = {"en": "EN", "ja": "日本語", "ko": "한국어", "zh": "中文"}


_MIN_PART = 12          # 分句后的最小长度：比它短就并回上一句
_CONNECTIVES = {        # 以这些词开头且很短的碎片才并回上一句（见 _split_long）
    "and", "but", "because", "so", "or", "if", "when", "while", "though",
    "although", "that", "which", "who", "then", "also", "however", "well",
    "yeah", "ok", "okay", "uh", "um",
}


def _split_runaway(text: str, limit: int) -> list[str]:
    """没有句末标点的超长"跑句"：在**逗号**处兜底切一次，别让一整行糊满屏。

    实锤（testdata/test real）：连续语音（游戏 CG / 直播解说）经常 100+ 字符
    一个句号都没有；旧行为是"无标点就不硬切" → 一行 30 多个词堆在屏幕上，
    读起来非常难受。只在明显超长（> limit×1.6）时才动，切点取**最靠近 limit
    的逗号/顿号**；找不到逗号就原样返回（绝不按字数硬切，那会把词劈开）。
    """
    if not limit or len(text) <= limit * 1.6:
        return [text]
    out: list[str] = []
    rest = text
    while len(rest) > limit * 1.6:
        win_a, win_b = int(limit * 0.6), int(limit * 1.4)
        best = None
        for m in re.finditer(r"[,，、;；]\s*", rest[:win_b]):
            if m.end() >= win_a and (best is None
                                     or abs(m.end() - limit) < abs(best - limit)):
                best = m.end()
        if best is None:
            break
        out.append(rest[:best].strip())
        rest = rest[best:]
    if not out:
        return [text]
    if rest:
        out.append(rest.strip())
    return out


def _split_long(text: str, limit: int) -> list[str]:
    """超长文本按句末标点再分句——解决"一大段糊在一起"。

    2026-09-15 增强（此前英文长句从不分句）：
      1. **英文句号也参与切分**：原来只认 。！？!?；;，所以英文长段落永远一整行；
         现在在"句号 + 空格 + 大写字母"处切，避免切坏 3.5 / Mr. 这类写法。
      2. **碎片合并**：短于 _MIN_PART(12) 的片段、上一句过短的、以及
         "以连接词开头且本身很短(<20)" 的碎片并回上一句——消除 "Yeah." 这类
         碎片单独成行；但不会把 Then/That 开头的正常句子并掉。
      3. 最多 3 条，避免把长句切得过碎；没有句末标点就不硬切。
    """
    if not limit or len(text) <= limit:
        return [text]
    parts = [p for p in re.split(r"(?<=[。！？!?；;…])|(?<=\.)(?=\s+[A-Z])", text)
             if p.strip()]
    if len(parts) < 2:
        # 没有句末标点可切：交给逗号兜底（连续语音的"跑句"，见 _split_runaway）
        return _split_runaway(text, limit)
    merged: list[str] = []
    for p in parts:
        if merged:
            prev = merged[-1]
            head = re.split(r"[\s,，、]+", p.strip(), maxsplit=1)[0].strip(".!?…").lower()
            # 只并"真碎片"：过短的句子、过短的上一句、或以连接词开头的**短**碎片。
            # 注意不能对所有连接词开头都合并——Then/That/Well 都是常见句首，
            # 一律合并会把刚切好的句子又并回去（实测踩过）。
            if (len(p.strip()) < _MIN_PART
                    or len(prev.strip()) < _MIN_PART
                    or (len(p.strip()) < 20 and head in _CONNECTIVES)):
                merged[-1] = prev + p
                continue
        merged.append(p)
    out, buf = [], ""
    for p in merged:
        if buf and (len(buf) + len(p)) > limit and len(out) < 2:
            out.append(buf)
            buf = p
        else:
            buf += p
    if buf:
        out.append(buf)
    # 统一去掉首尾空白：句号后切出的下一段自带前导空格，直接上屏会看到
    # "  Then I'll have more time…"（用户日志里出现过）；内部空格（合并碎片时
    # 由前一段留下的那个）必须保留，否则会粘成 "plans.So"。
    return [p.strip() for p in (out or [text])]


def _merge_continuation(tail: dict, lang: str, seg_end_wall: float,
                       src_text: str, gap_s: float, max_chars: int) -> str | None:
    """半句续接判定（append-and-correct）：该并回上一行吗？

    tail 记的是"最后一行"（cid/为什么结束/结束时刻/语种/文本）。只有上一行是
    **撞单句上限被强切**（cut=="max"）时才可能续接 —— 那种切法本来就不认识句子
    边界，后半截紧接着就会来。返回上一行文本表示要合并，None 表示新起一行。

    依据：LiveCaptions-Translator（显示单位是句子而不是音频块）、
    Buzz 的 append-and-correct。
    """
    if (tail.get("cid") is not None and tail.get("cut") == "max"
            and tail.get("lang") == lang
            and 0.0 <= (seg_end_wall - float(tail.get("t") or 0.0)) < gap_s
            and len(tail.get("text") or "") + len(src_text) <= max_chars):
        return tail.get("text")
    return None


# ---- 标点恢复（punctuation restoration）----
# 病根实测（testdata/test real，2026-09-16）：whisper 在 4s 强切出来的片段上
# 几乎不给句读（整段 20s 一次识别也只有开头的问号），而**停顿又不含句界信息**：
# 标准切句处的实际静音只有 0 / 70 / 150 / 0 ms，全片唯一的长停顿（0.46s）反而
# 落在一句话中间。所以"像 checklist 那样切句"只剩一条路 —— 把标点补出来。
# 本机 Qwen2.5-1.5B（已有权重、已在跑翻译）做这件事实测 0.2~0.4s/句，中英日韩都能补。
_PUNCT_SYS = ("你是字幕标点恢复器。输入是语音识别出的、没有标点的文本。"
              "任务：只补标点（. , ? !），不要改词、不要删词、不要翻译、不要解释。"
              "句子说完打句号；从句之间打逗号。只输出补好标点的那一行文本。"
              # 这一句是实测加上去的（2026-09-17）：不加时模型爱"顺手润色"——
              # 补冠词（say → **The** police say）、加连词（hero plays, **and** stand-up
              # shows）、改词形。25 条真实字幕行 A/B：两版都通过 19/25，但加上这句
              # 后"真的补出标点"的从 13 条升到 16 条（加词的都被校验拦下了）。
              "禁止增删单词：不要补 the / a / and 这类词，不要改词形"
              "（大小写与撇号除外）；每个词都必须原样保留，只在词与词之间插入标点。")

# 词切分（英文按词、CJK 按字）+ 句末标点集合；位置信息用来回看"词与词之间"的标点。
_SEAM_WORD_RE = re.compile(r"[A-Za-z0-9']+|[\u4e00-\u9fff]")
_SEAM_END = ".!?…。！？"
_PREROLL_S = 0.25          # 前卷长度（秒）：实测 0.2~0.5s 都能救回被切掉的词，0.8s 会退化
_PUNCT_CPU = [None]        # CPU 标点模型：None=没试过 / False=不可用 / 否则实例


def _punct_cpu(model_dir: str):
    """加载 CPU 标点模型（FunASR CT-Transformer ONNX），只加载一次；失败返回 None。"""
    if _PUNCT_CPU[0] is not None:
        return _PUNCT_CPU[0] or None
    try:
        from funasr_onnx import CT_Transformer
        _PUNCT_CPU[0] = CT_Transformer(str(model_dir), quantize=True,
                                       intra_op_num_threads=2)
    except Exception:
        _PUNCT_CPU[0] = False      # 缺依赖/缺模型：后续调用直接走 Qwen，不再重试
    return _PUNCT_CPU[0] or None


def _punct_restore(mt, text: str, engine: str = "auto", cpu_dir: str = "",
                   lang: str = "") -> str | None:
    """给"没有标点的识别文本"补标点；拿不到结果返回 None（调用方退回旧行为）。

    两条路（config.punct_engine）：
      * **cpu**  FunASR CT-Transformer（ONNX，CPU）：实测 **1~4ms**、不占 GPU、
                 加载 0.4s。但**只对中文可靠** —— 英文上它会乱插句号
                 （"We can go watch lion dances。 hero plays"、"Let's move。 then…"），
                 所以英文/日文/韩文不用它。
      * **qwen** 本地 Qwen（已有权重，本来就在跑翻译）：实测 0.3~0.5s、占 GPU，
                 但英文/日文/韩文的句读位置是对的。
      auto = 中英按语言自动挑（zh→cpu，其余→qwen）。

    只当**判据**用，绝不用它改写已上屏的字（见 _seam_breaks_sentence）：
    模型偶尔会顺手改词，所以下游一定要能"对不上就退回旧行为"。
    """
    if not text or len(text) < 6:
        return None
    eng = str(engine or "auto").strip().lower()
    use_cpu = eng == "cpu" or (eng == "auto" and lang == "zh")
    if use_cpu:
        m = _punct_cpu(cpu_dir)
        if m is not None:
            try:
                out = m(text)
                if isinstance(out, (list, tuple)):
                    out = out[0] if out else ""
                out = str(out or "").strip()
                if out:
                    return out
            except Exception:
                pass
        if eng == "cpu":
            return None            # 明确指定 cpu：拿不到就退回旧行为（不偷偷换 GPU）
    chat = getattr(mt, "chat", None)
    if chat is None:
        return None
    try:
        out = chat([{"role": "system", "content": _PUNCT_SYS},
                    {"role": "user", "content": text}])
        return str(out or "").strip() or None
    except Exception:
        return None


def _punct_same_words(orig: str, cand: str) -> bool:
    """补标点结果是否"只动了标点"？（纯函数，可单测）

    判据：两边去掉**所有非字母数字字符**（含 CJK、撇号、空格）再比，一字不差才通过。
    这是"绝不改字"的最后一道闸 —— 小模型偶尔会顺手改词/翻译/加解释（实锤：Qwen 把
    "we can go" 改成 "We can go" 这种大小写差异要放行，但少一个词绝不放行）。
    下划线也一并去掉：它算词字符，但几乎不可能是识别文本的一部分。
    """
    def _core(s: str) -> str:
        return re.sub(r"[\W_]+", "", str(s or ""), flags=re.UNICODE).lower()

    a, b = _core(orig), _core(cand)
    return bool(a) and a == b


def _punct_lang_for(text: str, lang: str) -> str:
    """补标点时按哪个语言选引擎：**以文本本身为准**，标签只作兜底。

    实锤（2026-09-17 离屏冒烟，源语言固定 zh 却在放英文 BBC）：语言标签是 zh，
    于是走 CPU 标点模型，它往英文里插了中文句号 —— "the biggest security。"。
    所以只看标签不够：纯拉丁文本一律按 en 处理（走 Qwen），有假名/谚文/汉字
    才交给 CPU（CPU 模型只对中文可靠，见 _punct_restore）。
    """
    return _script_lang(text) or ("en" if str(text).isascii() else lang)


def _punct_final_line(mt, text: str, engine: str = "auto", cpu_dir: str = "",
                      lang: str = "") -> str:
    """定稿前给整行补一次标点；**只加标点才采用**，否则原样返回。

    为什么只在"没有句末标点"时才调：whisper 自己给了句读的段（有 。！？.!?…）
    说明模型这次听清了句子边界，不必再花一次调用；连续语音里"撞上限强切"的段
    才是几乎不给句读的那一类（实测 4s 片段整段只有 1 个标点），也正是补标点收益
    最大的地方。省下的调用量很可观（多数正常段直接跳过）。
    """
    t = str(text or "")
    if mt is None or len(t) < 6 or any(ch in t for ch in _SEAM_END):
        return text
    cand = _punct_restore(mt, t, engine=engine, cpu_dir=cpu_dir, lang=lang)
    if cand and _punct_same_words(t, cand):
        return cand
    return text


def _seam_breaks_sentence(text2: str, tail_text: str, piece_text: str = "",
                          look_back: int = 3) -> bool:
    """接缝判据：模型认为"这里已经说完了"吗？（纯函数，可单测）

    text2 是**补好标点**的合并候选文本（tail_text + piece_text，见 _punct_restore），
    tail_text 是上一行已显示的文本，piece_text 是本段文本。三者的重合处就是接缝 ——
    音频在 4s 上限处被硬切，两截本来是不是同一句，只能看句读。

    做法（三步）：
      ① 在 text2 前 80% 的词里找 tail_text 结尾那 3 个词（退 2 个、再退 1 个）
         → 命中段最后一个词的位置就是接缝起点；
      ② 在它之后找 piece_text 开头的 2 个词（退 1 个）→ 接缝终点；找不到就用
         紧跟的下一个词兜底；
      ③ 看这段"接缝区间"里有没有句末标点（逗号不算）：
         有 → 已说完整 → 不合并（另起一行）；没有 → 照旧合并。
    三步里任何一步判不了 → 返回 False（合并）—— 保守兜底：判据失灵就退回旧行为，
    绝不因为判据失灵丢字或乱分行。

    实锤（testdata/test real，本机 Qwen 实测输出）：
      "We've got loads of fun and jinjou We can go watch lion dances, …"
      → 补标点得到 "…and jinjou**.** We can go watch…" → 接缝有句号 → 该分
      （checklist.txt 正是这么分的）。
      "Plus tens of Different treats to try out"
      → 补标点得到 "Plus tens of different treats to try out"（无内部标点）→ 该合
      （两截本是同一句）。

    依据：能定句的开源项目都靠**模型输出的标点**，没有一家靠音频停顿定句
    （whisper_streaming 用词级时间戳把标点映射回音频切点；WhisperLiveKit 的
    PUNCTUATION_MARKS + 尾部挂起；停顿只作硬上限兜底）。
    """
    if not text2 or not tail_text:
        return False
    # 统一撇号（Qwen 常把 ' 打成 ’，不归一化会让 "We've" 断成两个词、接缝找不着）
    text2 = text2.replace("\u2019", "'").replace("\u02bc", "'")
    tail_text = tail_text.replace("\u2019", "'").replace("\u02bc", "'")
    piece_text = (piece_text or "").replace("\u2019", "'")
    tail_words = _SEAM_WORD_RE.findall(tail_text.lower())
    if not tail_words:
        return False
    spans = [(m.group(0).lower(), m.start(), m.end())
             for m in _SEAM_WORD_RE.finditer(text2)]
    if len(spans) < 2:
        return False
    words2 = [w for w, _, _ in spans]
    limit = max(1, int(len(words2) * 0.8))
    # ① 接缝起点：上一行结尾的词在候选文本里的位置
    j = -1
    for k in range(min(look_back, len(tail_words)), 0, -1):
        needle = tail_words[-k:]
        for i in range(max(0, limit - k + 1)):
            if words2[i:i + k] == needle:
                j = i + k - 1
                break
        if j >= 0:
            break
    if j < 0 or j + 1 >= len(spans):
        return False
    # ② 接缝终点：本段开头 2 个词（退 1 个）在候选文本里的位置
    piece_words = _SEAM_WORD_RE.findall(piece_text.lower())
    q = -1
    for m in range(min(2, len(piece_words)), 0, -1):
        needle = piece_words[:m]
        for i in range(j + 1, len(words2) - m + 1):
            if words2[i:i + m] == needle:
                q = i
                break
        if q >= 0:
            break
    # ③ 接缝区间里的标点
    end = spans[q][1] if q > j else spans[j + 1][1]
    gap = text2[spans[j][2]:end]
    return any(c in _SEAM_END for c in gap)


def _is_echo_of_ctx(dst: str, ctx: list) -> bool:
    """译文是不是**原样复读**了上下文最后一条？

    半句续接（append-and-correct）时最容易发生：这一行是上一行的延长，而送给翻译
    模型的上下文里最后一条正是"本句前半截"的译文 —— 小模型会偷懒直接把它吐回来，
    结果**前半截翻了两遍、后半截根本没翻**（用户反馈："纠错之后翻译还是没纠错之前的"）。
    实测（Qwen3-1.7B-CT2，真实上下文）：
        入 "And then maybe the martial championships My god Plus tons of different treats to try out"
        出 "然后也许武术锦标赛，我的上帝，加。"   ← 与上一句译文一字不差
    修法见 process_segment：续接行翻译时把那条上下文摘掉（实测同句能翻全）。
    """
    return bool(ctx) and (dst or "").strip() == str(ctx[-1]).strip()


def _trim_overlap(prev_text: str, new_text: str, look: int = 12) -> str | None:
    """前卷解码的**重叠去重**：去掉 new_text 开头与 prev_text 结尾重复的部分。

    撞上限强切出来的片段解码时要带一点点前卷音频（见 process_segment 的"前卷"），
    代价是识别结果会把上一行结尾的词再吐一遍（"…show rover" + "rover around. …"）。
    这里按词找**严丝合缝**的重叠（上一行结尾 k 个词 == 新文本开头附近的 k 个词，
    k 从大到小试），把重复部分去掉，只留新内容：
        prev="…to show rover"  new="rover around. We've got loads…"
        → "around. We've got loads…"   ← 顺带把被切掉的 "around" 找回来了

    返回 None = **没找到重叠**（两段不是接着说的）→ 调用方退回原文本、不做任何删改；
    返回 "" = 整段都是上一行说过的话（重复/回声）→ 调用方整段丢弃。
    只看新文本前 look 个词，避免在长文本里"巧合匹配"把内容吃掉。
    """
    if not prev_text or not new_text:
        return None
    a = _SEAM_WORD_RE.findall(prev_text.lower())
    if not a:
        return None
    spans = [(m.group(0).lower(), m.start(), m.end())
             for m in _SEAM_WORD_RE.finditer(new_text)]
    if not spans:
        return None
    b = [w for w, _, _ in spans]
    for k in range(min(len(a), 8), 0, -1):
        needle = a[-k:]
        for i in range(0, min(len(b) - k + 1, look)):
            if b[i:i + k] != needle:
                continue
            j = i + k                      # 重叠部分之后的第一个词
            return new_text[spans[j][1]:].strip() if j < len(spans) else ""
    return None


def _seam_window(tail_text: str, piece_text: str, n: int = 12) -> str:
    """接缝判定的**输入窗**：上一行尾部 n 个词 + 本段开头 n 个词。

    判定只看接缝那一小段，把整句喂给模型纯属浪费 —— 补标点的耗时几乎全在**生成**
    （逐字产出整句），输入越长越慢。实测（Qwen3-1.7B）：整句 92 字 = 491ms，
    窗口约 25 词 ≈ 200ms。
    """
    def _cut(text: str, tail: bool) -> str:
        spans = list(_SEAM_WORD_RE.finditer(text or ""))
        if not spans or len(spans) <= n:
            return (text or "").strip()
        # ⚠ .start/.end 必须带括号调用：Match 的这俩是方法，漏括号会把
        # "方法对象"当切片下标 → TypeError（slice indices must be integers），
        # 且异常一抛整段字幕就丢了（实锤：中文视频里两句凭空消失，error.log
        # 里 live.py:339 反复出现——之前一直没定位到就是这个）。
        return (text[spans[-n].start():] if tail
                else text[:spans[n - 1].end()]).strip()

    parts = [p for p in (_cut(tail_text, True), _cut(piece_text, False)) if p]
    return " ".join(parts)


def _common_prefix(prev_text: str, new_text: str) -> str:
    """两次部分识别的**最长公共前缀**（LocalAgreement-2 的核心）。

    流式上屏"只追加不回退"的关键：只有连续两次识别都一样的开头才算确认。
    英文按词比（whisper 的输出带空格）；中日韩没有词间空格，退化成按字符比。
    """
    if " " in prev_text or " " in new_text:
        a, b, sep = prev_text.split(), new_text.split(), " "
    else:
        a, b, sep = list(prev_text), list(new_text), ""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return sep.join(b[:n])


_MUSIC_MARKS = "♪♫♬♩"
_MUSIC_ONLY = {"music", "音乐", "applause", "掌声", "laughter", "笑声", "bgm", "inst"}


def _is_music_only(text: str) -> bool:
    """是否为"只有音乐/掌声/笑声"的输出（whisper 遇到唱歌/纯音乐常这么吐）。

    去掉音符与标点后要么为空（只有 ♪♪♪），要么落在已知的非语音标记集合里。
    """
    t = (text or "").translate(str.maketrans("", "", _MUSIC_MARKS))
    t = re.sub(r"[\s\[\]【】()（）.,，。;；!！?？~～\-]+", "", t).lower()
    return (not t) or (t in _MUSIC_ONLY)


def _looks_like_loop(text: str) -> bool:
    """识别结果的"复读幻觉"检测（whisper 在近静音时最典型的故障形态）。

    实测案例：声音很小的时候，整屏变成 "rover rover rover rover …"。
    两条判据（都要求足够长，避免误伤正常重复口语）：
      1. 同一个词**连续**重复 ≥4 次（含逗号/顿号分隔）；
      2. 单个词占了长文本 40% 以上。
    """
    if not text:
        return False
    words = re.findall(r"[A-Za-z']+|[\u4e00-\u9fff]", text)
    if len(words) < 8:
        return False
    if re.search(r"(\b[\w']+\b)(?:[\s,，、]*\1){3,}", text, flags=re.I):
        return True
    if len(words) < 10:
        return False
    from collections import Counter
    most = Counter(w.lower() for w in words).most_common(1)[0][1]
    return most / len(words) > 0.4


# Whisper 的"经典幻觉句"：训练语料（YouTube 字幕）里满地都是这些结尾套话，
# 模型在音乐/静音/噪声片段上会**高置信度**地反复吐它们（日语尤其严重）。
# 实测用户日志：日语 147 段里 11 段是这些（ご視聴ありがとうございました ×9 等）。
_HALLUC_RAW = (
    # 日语
    "ご視聴ありがとうございました", "ご視聴ありがとうございます",
    "ご視聴ありがとうございました!", "チャンネル登録お願いします",
    "チャンネル登録よろしくお願いします", "おやすみなさい",
    "ありがとうございました", "ありがとうございます",
    # 英语
    "thankyouforwatching", "thanksforwatching", "pleasesubscribe",
    "subscribetomychannel", "amara.org", "subtitlesby", "theend",
    # 韩语
    "시청해주셔서감사합니다", "구독과좋아요부탁드립니다",
    # 中文
    "谢谢观看", "感谢观看", "谢谢大家观看", "请订阅", "字幕由",
)


def _norm_for_halluc(text: str) -> str:
    """幻觉句比对用的归一化：去空白/标点 + 转小写（CJK 保留，先做简体化对齐）。"""
    s = to_simplified(text or "")
    return re.sub(r"[\s\W_]+", "", s).lower()


_HALLUC_NORMS = tuple(dict.fromkeys(
    _norm_for_halluc(x) for x in _HALLUC_RAW if _norm_for_halluc(x)))


def _looks_like_prompt_echo(text: str, terms) -> bool:
    """"术语提示词回声"检测：输出几乎就是把 initial_prompt 复读一遍 → 丢弃。

    实测（2026-09-16 日志，同一句连出 3 次、全天出现 10 次）：
        "Genshin, Genshin wuthering waves, wuthering wave, wuthering waves rover, rover rover"
    术语表里恰好只有 Genshin / wuthering waves / rover —— 这是 whisper 在音乐、
    掌声、静音片段上把 initial_prompt **原样吐回来**（提示词偏置的经典副作用），
    词级复读门（_looks_like_loop）抓不到，因为它只查"同一个词连 ≥4 次 / 单词占比 >40%"。

    两条同时满足才判为回声（正常对白即使提到术语也不会同时满足）：
      ① ≥70% 的词来自提示词词表；
      ② 词形丰富度 unique/total ≤ 0.8（正常句子约 0.9；回声句反复堆同一批词）。
    另要求 ≥5 个词：短句不做判断，避免误伤"Genshin and rover"这类正常短句。
    """
    if not text or not terms:
        return False
    pat = r"[A-Za-z']+|[\u4e00-\u9fff]"
    words = re.findall(pat, text.lower())
    if len(words) < 5:
        return False
    vocab: set[str] = set()
    for t in terms:
        vocab.update(re.findall(pat, str(t).lower()))
    if not vocab:
        return False
    hit = sum(1 for w in words if w in vocab)
    if hit / len(words) < 0.7:
        return False
    return len(set(words)) / len(words) <= 0.8


def _looks_like_hallucination(text: str) -> bool:
    """命中经典幻觉句 → 丢弃。

    只对**短文本**（归一化后 ≤ 24 字）生效，且要求与黑名单条目整句相等或互相包含，
    避免误伤正常长对白。这些句子本身没有信息量，丢掉的收益远大于风险。
    """
    t = _norm_for_halluc(text)
    if not t or len(t) > 24:
        return False
    for p in _HALLUC_NORMS:
        if t == p or (len(t) >= 6 and (t in p or p in t)):
            return True
    return False


def _rms_int16(pcm) -> float:
    """16bit PCM 的均方根（能量门用；纯 CPU，开销可忽略）。"""
    if pcm is None or len(pcm) == 0:
        return 0.0
    a = np.asarray(pcm, dtype=np.float32)
    return float(np.sqrt(np.mean(a * a)))


def _script_lang(text: str) -> str | None:
    """按文字系统判语种：有假名→ja，有谚文→ko，只有汉字→zh，其余 None。

    用于纠偏语言锁错位：锁在 en/ja/ko 期间插播中文（或语种检测错位）时，
    whisper 会转出中文文本却按外语送去翻译——翻译模型收到"中文→中文"
    就原样回声，用户看到的"译文=原文"正是它。假名/谚文优先
    （日文语句几乎总带送假名，纯汉字日文极罕见）。
    """
    has_kana = has_hangul = has_han = False
    for ch in text:
        o = ord(ch)
        if 0x3040 <= o <= 0x30FF or ch == "ー":
            has_kana = True
        elif 0xAC00 <= o <= 0xD7AF:
            has_hangul = True
        elif 0x4E00 <= o <= 0x9FFF:
            has_han = True
    if has_kana:
        return "ja"
    if has_hangul:
        return "ko"
    if has_han:
        return "zh"
    return None


class LivePipeline:
    def __init__(self, cfg: AppConfig | None = None,
                 on_subtitle=None, on_state=None, on_partial=None,
                 on_activity=None,
                 on_translation_update=None, on_drop=None):
        """回调约定（都带 cue id：同一句的部分识别 / 最终识别 / 流式译文共用一个
        id，UI 据此更新同一行——上一句翻译迟到时也能补回它自己那一行，
        不会把当前句覆盖掉）：
          on_subtitle(cid, src, dst, lang, latency_ms)   —— 该句定稿
          on_partial(cid, src, lang, latency_ms)         —— 部分结果（原文先上屏；
                                                            实时模式还会边说边出）
          on_translation_update(cid, src, dst, lang)     —— 流式译文中间结果
          on_drop(cid)                                   —— 该句被门控丢弃，
                                                            界面应撤回它的行
          on_state(str)                                  —— 生命周期日志
        """
        register_nvidia_dlls()
        self.cfg = cfg or AppConfig()
        # on_activity(text)：给悬浮窗显示"识别中…/翻译中…"这类即时状态（不写日志）
        self.on_activity = on_activity or (lambda *a: None)
        self.on_subtitle = on_subtitle or (lambda *a: None)
        self.on_state = on_state or (lambda s: None)
        self.on_partial = on_partial or (lambda *a: None)
        self.on_translation_update = on_translation_update or (lambda *a: None)
        # on_drop(cid)：本段被门控丢弃（音乐/幻觉/复读/回声…）——UI 撤回它的行。
        # 实时模式下部分识别可能已经把半句推上屏了，不撤回就会留下"幽灵字幕"
        # （一句永远等不到定稿的残句，直到下一句把它顶进历史行）。
        self.on_drop = on_drop or (lambda *a: None)
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._worker = None
        self.latencies: list[float] = []
        self.session_cues: list[dict] = []
        self._cue_index: dict = {}       # cid → session_cues 下标（续接时原地更新）
        self._t0 = time.perf_counter()
        # 源语言：None=自动检测（默认）；en/ja/ko/zh=强制按该语言识别
        self._forced_lang: str | None = None
        self._src_ctx_clear = False      # 切换语言后要求清空上下文
        self.stats = {"segments": 0, "translations": 0, "errors": 0}
        self._mt_cache: dict[tuple[str, str], tuple[float, str]] = {}
        self._mt = None

    # ---- 生命周期 ----
    def start(self):
        self._stop.clear()
        self.enforce_log_quota(self.cfg.log_max_mb)
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def set_source_language(self, lang: str | None):
        """运行期切换源语言：None/"auto" = 自动检测；en/ja/ko/zh = 强制识别。

        切换会清空上下文（不同语言的前文没有参考价值）。
        """
        v = (lang or "auto").strip().lower()
        self._forced_lang = None if v in ("auto", "", "none") else v
        self._src_ctx_clear = True
        return self._forced_lang

    def stop(self):
        self._stop.set()
        if self._worker:
            self._worker.join(timeout=15)

    def pause(self):
        self._pause.set()

    def resume(self):
        self._pause.clear()

    @property
    def paused(self):
        return self._pause.is_set()

    def notify_safe(self, msg: str):
        """管线线程 → 托盘通知（经 on_subtitle 同级回调，宿主决定呈现）。"""
        try:
            self.on_state("notify: " + msg)
        except Exception:
            pass

    def latency_report(self) -> dict:
        if not self.latencies:
            return {"n": 0}
        import numpy as np
        a = np.array(self.latencies)
        return {"n": len(a), "median_ms": round(float(np.median(a))),
                "p95_ms": round(float(np.percentile(a, 95))), "max_ms": round(float(a.max()))}

    def export_session(self, path: str, bilingual: bool = True) -> int:
        """导出本次会话字幕为 SRT（阶段3 导出热键）。返回 cue 数。"""
        from .subtitles import Cue, write_srt
        cues = [Cue(c["start"], c["end"], c["src"], c["dst"]) for c in self.session_cues]
        write_srt(cues, path, bilingual=bilingual)
        return len(cues)

    # ---- 主循环 ----
    @staticmethod
    def enforce_log_quota(max_mb: int):
        """阶段5：logs/ 总量限额，超限删最旧文件。"""
        from .config import LOGS_DIR
        files = sorted(LOGS_DIR.glob("*"), key=lambda p: p.stat().st_mtime if p.is_file() else 0)
        total = sum(p.stat().st_size for p in files if p.is_file()) / 1024 / 1024
        i = 0
        while total > max_mb and i < len(files):
            f = files[i]
            if f.is_file():
                total -= f.stat().st_size / 1024 / 1024
                try:
                    f.unlink()
                except OSError:
                    pass
            i += 1

    def _run(self):
        cfg = self.cfg
        self.on_state("vad")

        def new_segmenter():
            """按配置的智能分句参数构造分段器（换设备后也要重建）。

            引擎可切：silero（默认） / fsmn（切句更整，实测平均段长 3.3s vs 1.4s）。
            FSMN 需要额外依赖（funasr-onnx）与模型目录，缺任一都**自动回退 Silero**。
            """
            engine = str(getattr(cfg, "vad_engine", "silero") or "silero").strip().lower()
            if engine == "firered":
                try:
                    from .vad_firered import FireRedVadSegmenter
                    seg = FireRedVadSegmenter(
                        model_dir=str(getattr(cfg, "vad_firered_dir",
                                              "models/fireredvad-onnx")),
                        min_silence_ms=float(
                            getattr(cfg, "firered_min_silence_ms", 700.0) or 700.0),
                        max_speech_s=float(
                            getattr(cfg, "firered_max_speech_s", 7.0) or 7.0))
                    self.on_state("vad: firered")
                    return seg
                except Exception as e:
                    self.on_state(f"FireRedVAD 不可用（{str(e)[:50]}）→ 回退 Silero")
            if engine == "fsmn":
                try:
                    from .vad_fsmn import FsmnSegmenter
                    seg = FsmnSegmenter(
                        model_dir=str(getattr(cfg, "vad_fsmn_dir", "models/fsmn-vad-onnx")),
                        max_segment_ms=float(getattr(cfg, "max_sentence_s", 10.0) or 10.0) * 1000)
                    self.on_state("vad: fsmn")
                    return seg
                except Exception as e:
                    self.on_state(f"FSMN-VAD 不可用（{str(e)[:50]}）→ 回退 Silero")
            return StreamingSegmenter(
                threshold=0.6,
                hold_ms=float(getattr(cfg, "merge_short_ms", 1200) or 0),
                end_silence_ms=float(getattr(cfg, "end_silence_ms", 400.0) or 400.0),
                max_segment_ms=float(getattr(cfg, "max_sentence_s", 10.0) or 10.0) * 1000,
                long_after_ms=float(getattr(cfg, "long_after_s", 6.0) or 6.0) * 1000,
                long_end_silence_ms=float(
                    getattr(cfg, "long_end_silence_ms", 260.0) or 260.0))

        segmenter = new_segmenter()
        self.on_state("asr")
        ctype = getattr(cfg, "asr_compute_type", None) or (
            "int8_float16" if cfg.asr_device == "cuda" else "int8")
        # 识别后端可插拔：whisper=本地（默认）；http=交给 vLLM / qwen-asr-serve /
        # FunASR 之类服务（换模型不改主程序，见 config.asr_backend）
        backend = str(getattr(cfg, "asr_backend", "whisper") or "whisper").strip().lower()
        # ---- GPU 子进程模式（config.gpu_subprocess）----
        # 识别/翻译模型活在子进程里，请求带超时：原生卡死只会拖住子进程，
        # 主进程（采集/VAD/悬浮窗）不受影响，超时后只重启子进程重载模型。
        # 详见 maisubtitle/gpu_proc.py 顶部说明与 HANDOVER 七节的实锤 A/B。
        gpu_worker = None
        if bool(getattr(cfg, "gpu_subprocess", False)):
            from .gpu_proc import GpuWorker, RemoteASR, RemoteMT
            gpu_worker = GpuWorker(cfg, on_state=self.on_state)
            gpu_worker.start()
            self.on_state("gpu: 识别/翻译在子进程中（卡死只重启子进程）")
            whisper = RemoteASR(gpu_worker, backend)
            self.on_state(f"识别后端: {backend}（子进程）")
            mt = RemoteMT(gpu_worker, cfg.engine)
            self._mt = mt
            self._gpu_worker = gpu_worker
        else:
            try:
                whisper = make_asr(backend, cfg.asr_model, cfg.asr_device, ctype,
                                   str(getattr(cfg, "asr_http_url", "") or ""),
                                   str(getattr(cfg, "asr_http_model", "") or ""),
                                   qwen_dir=str(getattr(cfg, "asr_qwen_dir", "") or ""),
                                   ws=cfg.ws_opts())
            except Exception as e:
                self.on_state(f"识别后端 {backend} 不可用（{str(e)[:60]}）→ 回退 whisper")
                backend = "whisper"
                whisper = WhisperASR(cfg.asr_model, device=cfg.asr_device,
                                     compute_type=ctype)
        if gpu_worker is None and backend != "whisper":
            if self._forced_lang is None:
                # HTTP / qwen3-onnx 都不做语种检测 → 建议固定源语言（F5）
                self.on_state("warn: 该识别后端不做语种检测，建议固定源语言（F5）")
            _url = getattr(whisper, "url", "")
            self.on_state(f"识别后端: {backend}" + (f" {_url}" if _url else ""))
        if backend in WS_BACKENDS:
            # 云端同样不做语种检测，而语言标签决定"翻不翻、按什么语言翻"。
            # 这句提醒在子进程模式下也要出（上面的通用提醒只覆盖非子进程路径）。
            _sl = str(getattr(cfg, "source_language", "auto") or "auto").strip().lower()
            if _sl in ("", "auto", "none"):
                self.on_state("warn: 火山流式识别不做语种检测 → 请固定源语言"
                              "（设置里选，或运行时 F5）")
            if not (str(getattr(cfg, "asr_ws_api_key", "") or "").strip()
                    or (str(getattr(cfg, "asr_ws_app_key", "") or "").strip()
                        and str(getattr(cfg, "asr_ws_access_key", "") or "").strip())):
                # 没密钥 = 每段都会失败（子进程模式还不会自动回落 whisper）→
                # 除了日志，还要弹托盘通知，否则用户只看到"字幕一直不动"
                self.on_state("warn: 火山流式识别未填密钥 → 识别会全部失败")
                self.notify_safe("火山流式识别没填密钥：设置 → 识别引擎 → "
                                 "火山引擎流式识别，填 X-Api-Key（新版控制台）")
        self.on_state("engine")
        if gpu_worker is None:
            mt = None
            # 引擎选择（候选表在 translate.translator_candidates，离线/子进程共用同一份，
            # 这里只多一层"逐候选记日志"）：
            #   hymt2 → Hy-MT2 专用模型（缺权重自动回落 QwenCT2）
            #   qwen  → QwenCT2（最快）
            #   qwen3 → Qwen3-1.7B-CT2（新一代，稍慢）→ QwenCT2
            for mk in translator_candidates(cfg.engine):
                try:
                    mt = mk()
                    break
                except Exception as e:
                    self.on_state(f"翻译引擎候选不可用({str(e)[:120]})")
            if mt is None:
                # NLLB 已移除：没有兜底模型了 —— 宁可只出原文，也不假装翻译
                mt = NullMT()
                self.on_state("warn: 翻译引擎全部不可用 → 只出原文")
                self.notify_safe("翻译模型不可用：本次只显示原文，不翻译")
            self._mt = mt
            self.on_state(f"{cfg.engine} 就绪 {type(mt).__name__} {mt.load_s:.1f}s")
        glossary = Glossary(gp) if (gp := cfg.glossary_file) else None

        # ---- 模型预热：先跑一次极短推理，把 CUDA kernel 首次编译 / 显存首次分配的
        # 开销提前消化掉。否则第一句字幕明显偏慢（实测首句常比后续慢 30~80%）。
        t_warm = time.perf_counter()
        try:
            whisper.transcribe(np.zeros(16000, dtype=np.int16), language="en",
                               beam_size=1)
            if hasattr(mt, "translate"):
                mt.translate("ok", "en", context=[])
            self.on_state(f"预热完成 {time.perf_counter() - t_warm:.1f}s")
        except Exception as e:
            self.on_state(f"预热跳过（{str(e)[:60]}）")

        # 源语言模式：auto=自动检测；指定语言时强制按该语言识别（不检测、不切换）
        _sl = str(getattr(cfg, "source_language", "auto") or "auto").strip().lower()
        self._forced_lang = None if _sl in ("auto", "", "none") else _sl
        lang_lock: str | None = self._forced_lang
        last_detect_ts = 0.0        # 上次语言检测时刻
        lang_recheck_s = 20.0       # 锁语言后的复检周期（秒），防止语言切换一直不被发现
        force_detect = True         # 首段必检；识别为空时置 True 立即复检
        context: deque = deque(maxlen=max(3, cfg.context_sentences))
        # ---- 半句续接状态（append-and-correct，见 process_segment）----
        # 上一段是撞上限被强切的话，紧接着的下一段就接到同一行上。
        tail = {"cid": None, "cut": "", "t": 0.0, "lang": "", "text": "", "pcm": None}
        merge_gap_s = 6.0          # 两截之间最多隔多久算"同一句"（秒）
        # 接缝标点判定：要合并之前，先让本地模型给"两截拼起来"的文本补标点，
        # 接缝处有句末标点就不合并（详见 _seam_breaks_sentence / _punct_restore）
        seam_punct = bool(getattr(cfg, "seam_punct", True))
        # 补标点引擎（cpu/qwen/auto，见 _punct_restore）：中文用 CPU 小模型（1~4ms），
        # 英文等用 Qwen（0.3~0.5s）；判定输入只取接缝窗口，别整句喂进去
        punct_engine = str(getattr(cfg, "punct_engine", "auto") or "auto")
        punct_cpu_dir = str(getattr(cfg, "punct_cpu_dir",
                                   "models/punc-ct-transformer-zh-en-onnx"))
        # 每行**定稿前**补标点（见 _punct_final_line）：whisper 在"撞上限强切"的
        # 段上几乎不给句读，补一次上屏的句子才读得下去。带"不许改词"校验，校验不过
        # 就用原文。英文等走 Qwen，每句多 0.3~0.5s（中文走 CPU，1~4ms）。
        punct_final = bool(getattr(cfg, "punct_final", True))
        # 前卷：撞上限强切出来的片段补 0.25s 前文再解码，找回被切掉的半个词
        pre_roll = bool(getattr(cfg, "pre_roll", True))
        # 合并后一行的长度上限：超过就不再往这行上接了（宁可另起一行，
        # 也不要把屏幕塞满）。取 split_long_chars 的 1.6 倍，与 _split_long
        # 的"超长才切"口径一致。
        _sc = int(getattr(cfg, "split_long_chars", 60) or 0)
        merge_max_chars = int(_sc * 1.6) if _sc else 240
        # 循环节拍：决定"语音结束→段吐出"的量化误差。实测中位延迟
        # 250ms 节拍 258ms → 100ms 节拍 176ms → 50ms 节拍 161ms；
        # 节拍只影响调用次数不影响 VAD 总帧数，所以 CPU 基本不变，取 100ms。
        tick_s = max(0.05, getattr(cfg, "loop_tick_ms", 100) / 1000.0)
        progressive = bool(getattr(cfg, "progressive_display", True))
        # 显示模式：bilingual 双语 / target 仅译文 / source 仅原文
        display_mode = getattr(cfg, "display_mode", "bilingual")
        if display_mode not in ("bilingual", "target", "source"):
            display_mode = "bilingual"
        # 仅原文模式不做翻译——省掉整段翻译的开销
        need_translation = display_mode != "source"
        # 渐进显示只在"当前行会显示原文"时才有意义（双语 / 仅原文）
        progressive = progressive and display_mode in ("bilingual", "source")
        # 实时模式：边说边做部分识别；sentence 模式：整句说完才识别一次
        realtime = str(getattr(cfg, "stream_mode", "realtime")) == "realtime"
        partial_interval_s = max(0.3, getattr(cfg, "partial_interval_ms", 700) / 1000.0)
        partial_min_s = float(getattr(cfg, "partial_min_s", 0.8) or 0.8)
        # 流式翻译：译文边生成边上屏（仅 CT2 引擎支持，其他实现自动回落整段翻译）
        use_stream = bool(getattr(cfg, "stream_translation", True))
        cap = audio.LoopbackCapture().start()   # 音频流持续打开，循环内只 drain 不重建
        cur_device = cap.device_name
        probe_pa = None                          # 复用的探测用 PyAudio
        next_probe = time.perf_counter() + 2.0
        self._t0 = time.perf_counter()

        def translate(text: str, lang: str, ctx: list | None = None) -> str:
            # 引擎都支持上下文（CT2 / Hy-MT2；NullMT 返回空串）。
            # ctx：本次要用的上下文（续接行会摘掉"本句前半截"那条，见 process_segment）
            zh, _ = mt.translate(text, lang,
                                 context=list(context) if ctx is None else ctx)
            zh = to_simplified(zh)
            if not re.search(r"[\u4e00-\u9fff]", zh):
                # 小模型偶发"原样回声"：去掉上下文再试一次
                zh, _ = mt.translate(text, lang, context=[])
                zh = to_simplified(zh)
            if re.search(r"[\u4e00-\u9fff]", zh):
                context.append(zh)
            return zh

        def _emit_cue(cid, src_text: str, dst: str, lang: str,
                      t_end: float, dur: float, lat_ms: float):
            """记会话（供导出）+ 上屏（蓝牙延迟补偿也在这里）。

            同一个 cid 再进来（半句续接把整句并回同一行）时**更新原条目**而不是
            追加 —— 否则导出的 SRT 会出现两行重叠的半句。
            """
            cue = {"start": max(0.0, t_end - dur), "end": t_end,
                   "src": src_text, "dst": dst, "lang": lang}
            idx = self._cue_index.get(cid)
            if idx is None:
                self._cue_index[cid] = len(self.session_cues)
                self.session_cues.append(cue)
                self.stats["segments"] += 1
            else:
                # 续接：起点保持最早的那次（否则导出/段长统计只剩后半截）
                cue["start"] = min(self.session_cues[idx]["start"], cue["start"])
                self.session_cues[idx] = cue

            def emit():
                self.on_subtitle(cid, src_text, dst, lang, round(lat_ms))

            offset_ms = int(getattr(cfg, "display_offset_ms", 0) or 0)
            if offset_ms > 0:
                # 阶段5：蓝牙等外设延迟补偿（显示延后，不打进延迟统计）
                threading.Timer(offset_ms / 1000, emit).start()
            else:
                emit()

        def process_partial(piece, cid):
            """实时模式：把"正在说的部分"先识别出来上屏（beam=1，求快）。

            这样字幕是"边说边长"的，而不是等整句说完才蹦出一行。
            **跑在独立的单槽线程里**（见下方 partial_q）：绝不排队，也绝不会
            把最终识别堵在后面（曾因共用队列把 final 延迟拖到 11.8 秒）。

            上屏文本**只追加、不回退**（LocalAgreement-2，见 _common_prefix）：
            连续两次部分识别都同意的词前缀才算"确认"。旧实现每 0.8s 把整段重识别
            的结果**直接覆盖**上屏，模型一改词（"their"→"there"）屏幕上的字就跳，
            用户反馈"流式模式用起来很难受"——所以他把流式关了（stream_mode=sentence）。
            """
            _force = getattr(self, "_forced_lang", None)
            if (_force is None and lang_lock is None) or display_mode == "target":
                return                       # 还没定语言 / 仅译文模式：先不显示半成品
            try:
                t0 = time.perf_counter()
                text, _ = whisper.transcribe(piece, language=_force or lang_lock,
                                             beam_size=1)
                cost_ms = round((time.perf_counter() - t0) * 1000)
            except Exception:
                return
            text = to_simplified(text.strip())
            if len(text) < 2:
                return
            st = partial_state
            if st["cid"] != cid:
                # 新段：第一次假设直接当"已确认"（否则要等第二次识别才有字可看）
                st["cid"], st["prev"], st["done"] = cid, text, text
            else:
                agree = _common_prefix(st["prev"], text)
                if len(agree) > len(st["done"]):
                    st["done"] = agree            # 只追加，绝不缩短
                st["prev"] = text
            if len(st["done"]) < 2:
                return
            self.on_partial(cid, st["done"], _force or lang_lock, cost_ms)

        def process_segment(piece, seg_end_wall, dur, cid=None, music_risk=False,
                            cut=""):
            """一个语音段的完整处理：语言 → 识别 →（可选重听）→ 翻译 → 回调。

            跑在独立工作线程里，这样主循环可以继续以 100ms 节拍采集+VAD，
            不会因为一次 ASR+翻译（约 1.2s）而憋住音频、堆积分段。

            cid：本段字幕的编号（实时模式下"部分识别"与最终识别共用一个 id，
            悬浮窗据此更新同一行；上一句翻译迟到时也靠它补回自己的那一行）。
            music_risk：VAD 语音占比低于 min_speech_ratio（人声压在配乐上/音乐段）
            —— 照常识别，但用更严的事后标准决定去留（见门①b）。
            cut：本段为什么结束 —— "max"（撞单句上限被强切）/ "silence"（等到了
            句尾静音）/ "flush"（音频断流收尾）。判为 "max" 时下一段会被续接到
            同一行（见下方"半句续接"）。
            """
            nonlocal lang_lock, last_detect_ts, force_detect
            # 用户切换了源语言：清上下文（跨语言的前文没有参考价值）
            if getattr(self, "_src_ctx_clear", False):
                context.clear()
                self._src_ctx_clear = False
            # 语言检测**不必每段都做**：detect_lang 是一次完整的 whisper
            # 编码器前向，锁住语言后再跑纯属重复。改为：锁前每段检测，
            # 锁定后只按 lang_recheck_s 周期复检（或识别为空时立即复检）。
            det = None
            now = time.perf_counter()
            forced = getattr(self, "_forced_lang", None)
            if forced:
                # 强制语言模式：不检测、不切换，直接按用户选定的语言识别
                lang = forced
                lang_lock = forced
            elif (lang_lock is None or force_detect
                    or (now - last_detect_ts) >= lang_recheck_s):
                det, prob = whisper.detect_lang(piece)
                last_detect_ts = now
                force_detect = False
                if det in TRANSLATABLE | {"zh"} and dur >= 1.0 and prob >= 0.6:
                    if lang_lock != det:
                        lang_lock = det
                        context.clear()
            if not forced:
                lang = lang_lock or (det if det in TRANSLATABLE else None)
            if lang is None:
                return
            prompt = None
            asr_terms_used: list[str] = []
            if glossary:
                # 识别侧词表：只取 use=both/asr 的词条，且带上"常见听错写法"
                # （use=mt 的词条只用于翻译，不喂给识别引擎，避免无中生有）
                asr_terms_used = glossary.asr_terms(lang)
                if asr_terms_used:
                    prompt = ("、" if lang in ("ja", "ko") else ", ").join(asr_terms_used)
            # ---- 前卷（pre-roll）：撞上限强切出来的片段，前面补 0.25s 音频再解码 ----
            # 实锤（testdata/test real）：4s 强切把 "…to show rover | around We've got…"
            # 切开后，whisper 解本段会把开头那半个词（around）**整段丢掉** —— 字幕直接
            # 少一个词。补 0.25s 前卷后得到 "rover around. We've got loads of fun in jinjou."
            # （前卷长度实测：0.2/0.3/0.5s 都能救回来，0.8s 会让模型退化只吐 "Oh my god"）。
            pre_pcm = None
            if (pre_roll and tail.get("pcm") is not None and tail.get("cut") == "max"
                    and 0.0 <= (seg_end_wall - float(tail.get("t") or 0.0)) < merge_gap_s):
                pre_pcm = np.concatenate([tail["pcm"], piece])
            text, _meta = whisper.transcribe(pre_pcm if pre_pcm is not None else piece,
                                             language=lang, prompt=prompt,
                                             return_confidence=True,
                                             beam_size=getattr(cfg, "beam_size", 5))
            text = text.strip()

            def _drop(counter: str):
                """丢弃本段：计数 + **撤回已上屏的部分结果**（否则那半句会一直挂在
                屏幕上，等下一句才被顶掉 —— "幽灵字幕"）。"""
                self.stats[counter] = self.stats.get(counter, 0) + 1
                if cid is not None:
                    try:
                        self.on_drop(cid)
                    except Exception:
                        pass
                self.on_activity("")

            # 前卷解码的重叠去重：把"上一行已经显示过的词"从本段文本里去掉。
            overlap_ok = False
            if pre_pcm is not None:
                trimmed = _trim_overlap(str(tail.get("text") or ""), text)
                if trimmed:
                    text, overlap_ok = trimmed, True
                    self.stats["pre_stitch"] = self.stats.get("pre_stitch", 0) + 1
                elif trimmed == "":
                    # 整段都是上一行说过的话（重复/回声）→ 丢弃，别在屏幕上再贴一遍
                    _drop("dropped_repeat")
                    return
                else:
                    # 没找到重叠：两段不是接着说的（模型没把上一行的词吐回来）
                    # → 文本原样保留，但下面**不做合并**，避免屏幕上出现重复内容
                    self.stats["pre_nostitch"] = self.stats.get("pre_nostitch", 0) + 1

            # 门：音乐/掌声/笑声（对唱歌的识别本来就不靠谱）→ 纯标记整段丢弃，
            # 人声夹着 ♪ 的把音符去掉继续用
            if _is_music_only(text):
                _drop("dropped_music")
                return
            if _MUSIC_MARKS in text:
                cleaned = text.translate(str.maketrans("", "", _MUSIC_MARKS)).strip()
                if len(cleaned) >= 3:
                    text = cleaned
            # 门①：模型自己判断"这段不是人说话"（近静音/纯噪声）→ 丢弃
            if text and float(_meta.get("no_speech_prob", 0.0) or 0.0) > 0.70:
                force_detect = True
                _drop("dropped_nospeech")
                return
            # 门①b：music_risk 段（VAD 语音占比低，多半是人声压在配乐上）用更严的
            # 事后标准淘汰 —— 这类段以前在入队前就被整段扔了（漏句），现在照常识别、
            # 由模型自评（no_speech_prob / avg_logprob）决定去留，并留一条实时提示。
            if music_risk:
                nsp = float(_meta.get("no_speech_prob", 0.0) or 0.0)
                alp = float(_meta.get("avg_logprob", 0.0) or 0.0)
                # 阈值刻意宽松：漏句比多出一行更让人难受（用户实测最烦"放着放着不动了"）。
                # no_speech_prob 0.45 对应 WhisperLive 的口径；avg_logprob 只在明显是
                # 胡言乱语（<-2.0，正常句子一般 > -1.0）时才丢。
                if nsp > 0.45 or alp < -2.0:
                    _drop("dropped_music")
                    now_w = time.perf_counter()
                    if now_w - music_warn_t[0] > 20.0:
                        music_warn_t[0] = now_w
                        self.on_state(
                            f"warn: 音乐段过滤 {self.stats['dropped_music']} 段"
                            f"（无语音置信度 {nsp:.2f}、对数概率 {alp:.2f}）"
                            f"——阈值可在设置里调")
                    return
            # 门③：经典幻觉句（"ご視聴ありがとうございました"这类结尾套话）→ 丢弃
            if drop_halluc and _looks_like_hallucination(text):
                _drop("dropped_halluc")
                return
            # 门②：复读幻觉（同一词连续 ≥4 次，或单词占比 >40%）→ 丢弃
            if _looks_like_loop(text):
                _drop("dropped_loop")
                return
            # 门②b：术语提示词回声（音乐/伴奏段把 initial_prompt 原样复读出来）→ 丢弃
            if _looks_like_prompt_echo(text, asr_terms_used):
                _drop("dropped_echo")
                return
            if not text:
                force_detect = True      # 识别为空：可能换语言了，下段强制复检
            # 语种纠偏（不额外花一次检测）：锁语言期间插播中文等错位场景，
            # 按"这段文字本身是什么文字系统"直接纠正，避免"中文→中文"式回声。
            script = _script_lang(text)
            if forced:
                pass                     # 强制语言模式：文本长什么样都按用户选的语言处理
            elif script and script != lang:
                lang = script
            elif script is None and lang in ("ja", "ko") and len(text) >= 6:
                # 纯 ASCII 不可能是日文/韩文（假名/谚文必然出现）→ 实际是英语。
                # 否则提示词会说"把这句韩语对白翻译成中文"，模型极易直接复读原文。
                lang = "en"
            if len(text) < 2:
                return
            lat_ms = (time.perf_counter() - seg_end_wall) * 1000
            self.latencies.append(lat_ms)
            src_text = to_simplified(text)
            # 识别侧术语纠错：把"听错但很接近术语"的词掰回术语表写法
            # （如 rower → rover）。字幕、上下文、导出都用纠正后的文本，
            # 用户在屏幕上直接看到正确写法，而不是等翻译阶段才纠正。
            if glossary:
                fixed, ghits = glossary.fix_asr(src_text, lang)
                if fixed != src_text:
                    src_text = fixed
                    self.stats["asr_term_fix"] = (self.stats.get("asr_term_fix", 0)
                                                  + len(ghits))
            # ---- 定稿补标点（punct_final）----
            # 放在这里（而不是上屏前）是有意的：补出来的句读随后会参与
            #   ① 半句续接判定（接缝有没有句末标点）② _split_long 的按句切分
            # —— 有了句读，长段才可能切在真正的句子边界上，而不是靠字数硬猜。
            # 时间如实计进延迟（这次调用确实推后了上屏，不粉饰）。
            if punct_final:
                _t_pf = time.perf_counter()
                _pf = _punct_final_line(mt, src_text, engine=punct_engine,
                                        cpu_dir=punct_cpu_dir,
                                        lang=_punct_lang_for(src_text, lang))
                if _pf != src_text:
                    src_text = _pf
                    self.stats["punct_final"] = self.stats.get("punct_final", 0) + 1
                lat_ms += (time.perf_counter() - _t_pf) * 1000
            cid = cid if cid is not None else f"s{int(seg_end_wall * 1000)}"
            # ---- 半句续接（append-and-correct）----
            # 上一段是"撞单句上限被强切"的（cut=="max"）且这一段紧接着就来
            # → 本来就是同一句被切开了：合并到同一行（复用 cid，界面按 id 原地
            # 更新），而不是新起一行制造半句碎片。
            # 实锤（testdata/test real，18 秒处）：上限在两个半句之间硬切 →
            # "Plus, tens of." + "Different treats to try out."（标准切句是
            # "Plus tons of different treats to try out." 一整句）。
            # 依据：LiveCaptions-Translator"显示单位是句子而不是音频块"、
            # Buzz 的 append-and-correct。
            # 合并后**绝不重新分句**：重切会让已经显示出来的那行变短/换切点
            # （实测 whisper 轮："We can go watch lion dances, hero plays, stand up
            # shows" 被重切成 "…hero plays,"，屏幕上的字往回缩）。
            # 只追加、不改写已有内容，才叫 append-and-correct；长度由
            # _merge_continuation 的 max_chars 把关。
            prev_text = _merge_continuation(tail, lang, seg_end_wall, src_text,
                                            merge_gap_s, merge_max_chars)
            if prev_text is not None and pre_pcm is not None and not overlap_ok:
                # 前卷解码里没出现上一行的词 → 两段不是接着说的（模型听到的是
                # 另一段话）→ 不合并，否则屏幕上会出现重复内容
                prev_text = None
                self.stats["merge_veto"] = self.stats.get("merge_veto", 0) + 1
            # ---- 接缝标点判定：该合还是该分，看模型给不给句读 ----
            # 旧行为是"撞上限被切 + 6s 内 → 无条件拼成一行"，它不认识句子边界：
            # 上半句其实已经说完了（"…and jinjou."）也会被硬粘上下一句。
            # 这里把两截拼起来交给本地模型**只补标点**，看接缝处有没有句末标点
            # （见 _seam_breaks_sentence）；判不出来就回到旧行为（合并）。
            # 为什么不能让 whisper 自己给标点：实测 4s 强切出来的片段几乎没有句读，
            # 拉长窗口（1.5/4/6/8s 都试过）也不稳定；而停顿不含句界信息（标准切句处
            # 的静音只有 0~150ms）。代价：这类段多一次 Qwen 调用（实测 0.2~0.4s），
            # 所以只在本就"要合并"的时候才做。
            if prev_text is not None and seam_punct:
                t_seam = time.perf_counter()
                # 只喂"接缝窗口"（上一行尾 12 词 + 本段前 12 词）：补标点的耗时几乎
                # 全在逐字生成，整句喂进去纯属浪费（实测 92 字 491ms → 窗口 ~200ms）
                punct = _punct_restore(mt, _seam_window(prev_text, src_text),
                                       engine=punct_engine, cpu_dir=punct_cpu_dir,
                                       lang=lang)
                # 这次判定确实把上屏推迟了，如实记进延迟（别粉饰）
                lat_ms += (time.perf_counter() - t_seam) * 1000
                if punct:
                    if _seam_breaks_sentence(punct, prev_text, src_text):
                        prev_text = None          # 接缝成句 → 另起一行
                        self.stats["seam_split"] = self.stats.get("seam_split", 0) + 1
                    else:
                        self.stats["seam_join"] = self.stats.get("seam_join", 0) + 1
                else:
                    self.stats["seam_skip"] = self.stats.get("seam_skip", 0) + 1
            if prev_text is not None:
                src_text = f"{prev_text} {src_text}".strip()
                cid = tail["cid"]
                parts = [src_text]
                merged_line = True       # 本行是上一行的延长（半句续接）
            else:
                # 智能分句：整段太长时按句末标点拆成多条（时间按比例分给各条）
                parts = _split_long(
                    src_text, int(getattr(cfg, "split_long_chars", 60) or 0))
                merged_line = False
            n = max(1, len(parts))
            # 续接状态：记"最后一行"的 id/文本，下一段撞上限被切时接到它上面
            tail.update(cid=(cid if n == 1 else f"{cid}#{n - 1}"), cut=cut,
                        t=seg_end_wall, lang=lang,
                        text=parts[-1] if parts else src_text,
                        # 尾部音频留给下一段做前卷（只取最后 0.25s：刚好够模型
                        # 认出被切开的那个词的词头，再长会让解码退化）
                        pcm=piece[-int(_PREROLL_S * 16000):])
            for i, part in enumerate(parts):
                # **第一段永远用基 cid**（不叫 cid#0）：续接把文本并回来、再按
                # 标点重新分句时，原来那一行还是原来那一行（原地更新），不会凭空
                # 多出一条重复行 —— 实测踩过："We can go watch … Shows." 出现两次。
                sub_cid = cid if i == 0 else f"{cid}#{i}"
                part_dur = dur / n
                part_end = seg_end_wall - (n - 1 - i) * part_dur
                # 识别刚出、译文还没好：先把原文推给悬浮窗，感知延迟直接砍半
                if progressive and lang in TRANSLATABLE:
                    self.on_partial(sub_cid, part, lang, round(lat_ms))
                dst = ""
                cache_hit = False
                cache_key = (part, lang)
                # 送给翻译模型的上下文：**续接行**（本段是上一行的延长）时，最后一条
                # 就是"本句前半截"的译文 —— 交给模型会诱导它原样复读，实测延长后的
                # 译文与上一句一字不差、后半截根本没翻。这次翻译把它摘掉。
                #  不要"完全去掉上下文"：实测无上下文会翻成中英夹杂
                #（"然后 maybe 的 martial championships 我的 god Plus 各种不同的 treats"），
                #   比复读还糟。
                ctx_mt = list(context)
                if merged_line and ctx_mt:
                    ctx_mt = ctx_mt[:-1]
                if lang in TRANSLATABLE and need_translation:
                    # 重复句缓存：同一句（120s 内）直接复用译文，跳过整次生成
                    hit = self._mt_cache.get(cache_key)
                    if hit and time.time() - hit[0] < 120.0:
                        dst = hit[1]
                        cache_hit = True
                        self.stats["mt_cache_hit"] = self.stats.get("mt_cache_hit", 0) + 1
                        self.on_translation_update(sub_cid, part, dst, lang)
                        context.append(dst)
                    else:
                        self.on_activity("翻译中…")
                if lang in TRANSLATABLE and need_translation and not cache_hit:
                    # 术语表**只给本句真正出现的术语**：
                    # 整张表会让小模型把没出现的词也硬塞进译文
                    # （实测：句中没有 rover，整表时译文也会冒出"漂泊者"）。
                    hits_t = terms_in(part, glossary.terms_for(lang)) if glossary else []
                    if glossary and hasattr(mt, "set_terms"):
                        mt.set_terms(hits_t)
                    canonical = part
                    if glossary:
                        canonical, hits = glossary.canonicalize(part, lang)
                    # 流式翻译：译文逐 token 增长，边生成边上屏（仅 CT2 引擎支持）
                    if (use_stream and hasattr(mt, "translate_stream")):
                        for partial in mt.translate_stream(canonical, lang,
                                                           context=ctx_mt):
                            dst = to_simplified(partial)
                            # 只推已经出中文的中间态，避免先闪一段英文。
                            # 续接行**也推**：显示层是"只前进的逐字显露"（见 overlay._set_target），
                            # 中间态比屏上短时它一个字都不动，所以既不会闪也不会往回缩，
                            # 反而让"译文变长"这件事看起来是连续的。
                            if re.search(r"[\u4e00-\u9fff]", dst):
                                self.on_translation_update(sub_cid, part, dst, lang)
                        # ①b 复读上一句译文（小模型把上下文最后一条原样吐回来）→
                        #     换"强制中文"提示词重译一次（见 _is_echo_of_ctx）
                        if _is_echo_of_ctx(dst, ctx_mt):
                            alt = to_simplified(force_zh(
                                mt, canonical,
                                terms=glossary.terms_for(lang) if glossary else None))
                            if re.search(r"[\u4e00-\u9fff]", alt):
                                dst = alt
                                self.stats["mt_echo_fix"] = (
                                    self.stats.get("mt_echo_fix", 0) + 1)
                        if not re.search(r"[\u4e00-\u9fff]", dst):
                            # ① 复读原文：去掉上下文整段重试一次
                            dst, _ = mt.translate(canonical, lang, context=[])
                            dst = to_simplified(dst)
                        if not re.search(r"[\u4e00-\u9fff]", dst):
                            # ② 仍无中文：换"强制中文"提示词再兜一次（不带语种假设）
                            dst = to_simplified(force_zh(
                                mt, canonical,
                                terms=glossary.terms_for(lang) if glossary else None))
                        if re.search(r"[\u4e00-\u9fff]", dst):
                            # 流式结果同样进上下文（否则"前几句"永远是空）
                            context.append(dst)
                        else:
                            # ③ 三次都没中文：宁可留空（只显示原文），
                            #    也绝不把英文原文当成"译文"显示出来。
                            dst = ""
                            self.stats["zh_fail"] = self.stats.get("zh_fail", 0) + 1
                        # ④ 术语校验：本句出现的术语若没进译文，用"源文替换术语"
                        #    再重译一次（小模型对 system 里的术语表常常不买账）
                        try:
                            missed = [(a, b) for a, b in hits_t if b not in dst]
                            if missed and hasattr(mt, "chat"):
                                alt, _ = mt.translate(
                                    substitute_terms(canonical, missed), lang,
                                    context=ctx_mt)
                                alt = to_simplified(alt)
                                if alt and all(b in alt for _, b in missed):
                                    dst = alt
                        except Exception:
                            pass
                        if dst and re.search(r"[\u4e00-\u9fff]", dst):
                            # 只在译文有效时写缓存；并限制条数避免无限增长
                            self._mt_cache[cache_key] = (time.time(), dst)
                            if len(self._mt_cache) > 64:
                                for k in list(self._mt_cache)[:16]:
                                    self._mt_cache.pop(k, None)
                        self.on_activity("")
                        # ⑤ 反向校验：译文里出现了"本句原文没有"的术语译名
                        #    → 模型在硬塞术语（整表诱导）。去掉术语表重译一次。
                        try:
                            if glossary and dst:
                                present = {a for a, _ in hits_t}
                                invented = [b for a, b in glossary.terms_for(lang)
                                            if b in dst and a not in present]
                                if invented:
                                    mt.set_terms([])
                                    alt, _ = mt.translate(canonical, lang,
                                                          context=ctx_mt)
                                    alt = to_simplified(alt)
                                    if alt and not any(b in alt for b in invented):
                                        dst = alt
                                    mt.set_terms(hits_t)   # 还原（下一段也会重设）
                        except Exception:
                            pass
                    else:
                        dst = translate(canonical, lang, ctx_mt)
                        # 与流式路径同样的复读兜底（见 _is_echo_of_ctx）
                        if _is_echo_of_ctx(dst, ctx_mt):
                            alt = to_simplified(force_zh(
                                mt, canonical,
                                terms=glossary.terms_for(lang) if glossary else None))
                            if re.search(r"[\u4e00-\u9fff]", alt):
                                dst = alt
                                self.stats["mt_echo_fix"] = (
                                    self.stats.get("mt_echo_fix", 0) + 1)
                        if dst and re.search(r"[\u4e00-\u9fff]", dst):
                            self._mt_cache[cache_key] = (time.time(), dst)
                            if len(self._mt_cache) > 64:
                                for k in list(self._mt_cache)[:16]:
                                    self._mt_cache.pop(k, None)
                        self.on_activity("")
                    self.stats["translations"] += 1
                else:
                    part = to_simplified(part)   # 中文语音原文繁体→简体
                _emit_cue(sub_cid, part, dst, lang,
                          part_end - self._t0, part_dur, lat_ms)
            # 本段处理完毕：收掉"识别中…"转圈。只在 need_translation 为假时才会
            # 漏掉这一步（仅原文模式跑完 ASR 后直接 _emit_cue）—— 漏掉的话活动
            # 提示永远停在那里，10s 后被悬浮窗当"卡死"清掉，日志里就是
            # "管线 10s 无活动，多半是 GPU 调用卡死"的**误报**。
            self.on_activity("")

        # ---- 工作线程：ASR + 翻译（与采集/VAD 并行）----
        # 队列元素统一为 (kind, piece, seg_end_wall, dur, cid, music_risk, cut)
        work_q: queue.Queue = queue.Queue(maxsize=4)
        worker_item_t = [0.0]      # 当前正在处理的条目的开始时刻（0 = 空闲）
        stuck_warned = [False]     # 卡住告警只打一次，恢复后自动解除
        # 最终识别/翻译期间持有；部分识别只非阻塞地尝试拿（定义必须在启动线程之前）
        gpu_busy = threading.Lock()

        def worker():
            while True:
                item = work_q.get()
                if item is None:
                    return
                kind, piece, seg_end_wall, dur, cid, music_risk, cut = item
                worker_item_t[0] = time.perf_counter()
                try:
                    if kind == "partial":
                        process_partial(piece, cid)
                    else:
                        with gpu_busy:      # 期间不做部分识别，保证最终识别的延迟
                            process_segment(piece, seg_end_wall, dur, cid,
                                            music_risk, cut)
                except Exception as e:
                    self.stats["errors"] += 1
                    import traceback as _tb
                    # 留证要能定位到"哪一行"：以前只记最后两行（源码行 + caret），
                    # 一旦异常出在第三方库里就完全没法查（实锤：error.log 里只剩
                    # 一行 `~~~~^^^^^^^^^`，反查半天没结果）。现在按帧记
                    # "文件:行 函数 ← 源码"，取**最内层 2 帧 + 最外层 1 帧**
                    #（最外层是"谁调用的"，最内层是"错在哪"）。
                    try:
                        frames = _tb.extract_tb(e.__traceback__)
                        keep = frames[-2:] + frames[:1]
                        info = " | ".join(
                            f"{Path(fr.filename).name}:{fr.lineno} {fr.name} ← "
                            f"{(fr.line or '').strip()[:60]}" for fr in keep)
                    except Exception:
                        info = _tb.format_exc().splitlines()[-2][:160]
                    self.on_state("error: " + str(e)[:140] + " || " + info[:400])
                finally:
                    worker_item_t[0] = 0.0

        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()

        # ---- 部分识别专用：单槽队列 + 独立线程 ----
        # 只保留"最新一次"部分识别：队列满说明上一次还没做完，直接丢弃本次。
        # 关键：**部分识别永远让位给最终识别**——GPU 忙或有最终任务积压时直接跳过
        # （实测：让它和最终识别并行跑会把 final 延迟从 ~900ms 拖到 30 秒）。
        partial_q: queue.Queue = queue.Queue(maxsize=1)
        # 部分识别的"确认前缀"状态（LocalAgreement-2，见 _common_prefix）
        partial_state = {"cid": None, "prev": "", "done": ""}

        def partial_worker():
            while True:
                item = partial_q.get()
                if item is None:
                    return
                piece, cid = item
                try:
                    if gpu_busy.acquire(blocking=False):
                        try:
                            process_partial(piece, cid)
                        finally:
                            gpu_busy.release()
                except Exception:
                    pass

        partial_thread = threading.Thread(target=partial_worker, daemon=True,
                                          name="asr-partial")
        partial_thread.start()
        # 部分识别节流：距上次至少 partial_interval_s，且要多出 0.45s 新内容；
        # 句子太长（> partial_max_s）就停手，等最终识别
        partial_max_s = float(getattr(cfg, "partial_max_s", 6.0) or 6.0)
        next_partial = [time.perf_counter()]
        last_partial_dur = [0.0]
        backlog_warned = [False]
        min_sentence_s = float(getattr(cfg, "min_sentence_s", 0.6) or 0.6)
        # 能量门阈值（16bit PCM 的 RMS）：低于它视为"没有人在说话"。
        # 60 约等于 -54 dBFS，正常说话即使很小声音的 RMS 也有几百，
        # 空调/底噪一般在 20~50，所以 60 既挡得住幻觉又不会砍掉轻声说话。
        min_rms = float(getattr(cfg, "min_rms", 60) or 0)
        # 丢弃"经典幻觉句"（可在设置里关掉，万一它误伤了真实对白）
        drop_halluc = bool(getattr(cfg, "drop_hallucinations", True))
        # 语音占比低于它 → 视为音乐/纯伴奏（VAD 阈值 0.6 下的"有声帧"比例）
        min_speech_ratio = float(getattr(cfg, "min_speech_ratio", 0.35) or 0)
        # 断流收尾：音频一断（视频暂停/CG 结束/播放器静默）多久就把"正在说的最后一句"
        # 立刻定稿。默认 0.9s —— 比句尾静音(0.7s)略长，不会误伤正常换气。
        idle_flush_s = max(1.0, float(getattr(cfg, "idle_flush_ms", 2500)) / 1000.0)
        last_audio_t = [time.perf_counter()]     # 上次收到音频的墙钟
        music_warn_t = [0.0]                     # 音乐过滤提示限频
        self.on_state("running")

        try:
            while not self._stop.is_set():
                time.sleep(tick_s)
                if self._pause.is_set():
                    cap.drain()          # 暂停期间丢弃音频，避免恢复时堆积
                    continue
                audio16k = cap.drain()
                wall_now = time.perf_counter()
                # 阶段5：默认输出设备热切换（**每 10s** 才探测一次，复用同一个 PyAudio）
                # 探测要枚举 WASAPI 设备（原生 COM 调用），之前 2s 一次过于频繁；
                # 原始崩溃现场显示"所有 Python 线程都空闲时发生 AV"，怀疑过这类
                # 高频原生调用，故降到 10s（换设备的跟随延迟最多 10s，可接受）。
                if wall_now >= next_probe:
                    next_probe = wall_now + 10.0
                    try:
                        if probe_pa is None:
                            probe_pa = audio.pyaudio.PyAudio()
                        new_default = audio.get_default_loopback(probe_pa)
                        if new_default["name"] != cur_device:
                            cap.stop()
                            cap.terminate()
                            cap = audio.LoopbackCapture().start()
                            cur_device = cap.device_name
                            segmenter = new_segmenter()   # 换设备后重置分段状态
                            self.on_state(f"设备切换 → {cur_device}")
                    except Exception:
                        probe_pa = None      # 探测对象失效则下次重建
                if not len(audio16k):
                    # 音频断流（视频暂停 / 一段 CG 结束 / 播放器静默）：
                    # **不能直接 continue** —— VAD 只在收到新音频时才能累计静音、判定句尾，
                    # 否则"最后一句"会一直挂着，直到下一段音频进来才蹦出来。
                    # 用户实测表现：字幕整体落后一句、非常不连贯。
                    if wall_now - last_audio_t[0] < idle_flush_s:
                        continue
                    last_audio_t[0] = wall_now
                    segments, fed_total = [], segmenter.total_samples
                    flush_fn = getattr(segmenter, "flush", None)
                    if flush_fn is not None:
                        try:
                            seg = flush_fn()
                        except Exception as e:
                            seg = None
                            self.on_state(f"warn: 断流收尾失败 {type(e).__name__}: "
                                          f"{str(e)[:60]}")
                        # 只收尾"确实说了一会儿"的句子：太短的（<1.0s）等下一段
                        # 音频接进来更合理，也避免在游戏静音片段里切出碎片
                        if seg and len(seg["pcm"]) >= 16000:
                            segments = [seg]
                            self.on_state(f"收尾待定句（断流 {idle_flush_s:.1f}s，"
                                          f"{len(seg['pcm'])/16000:.1f}s）")
                    if not segments:
                        continue
                else:
                    last_audio_t[0] = wall_now
                    # 流式分段：只在句尾静音/超长时才输出，长句不会被切断
                    segments = segmenter.feed(audio16k)
                    fed_total = segmenter.total_samples
                # 处理线程卡住检测（GPU 被别的程序占满时可能发生）：写日志留证
                if worker_item_t[0]:
                    if (not stuck_warned[0]
                            and wall_now - worker_item_t[0] > 20.0):
                        stuck_warned[0] = True
                        self.on_state("warn: 单条处理超过 20s（GPU 繁忙？），"
                                      "字幕可能暂停，恢复后自动继续")
                else:
                    stuck_warned[0] = False
                    if work_q.empty():
                        backlog_warned[0] = False   # 队列恢复清空，允许下次再告警
                # 实时模式：把"正在说的部分"定期送去部分识别，字幕边说边长
                if realtime:
                    pend = segmenter.pending()
                    if pend is None:
                        last_partial_dur[0] = 0.0
                    else:
                        pdur = len(pend["pcm"]) / 16000
                        if (partial_min_s <= pdur <= partial_max_s
                                and wall_now >= next_partial[0]
                                and pdur >= last_partial_dur[0] + 0.45
                                and work_q.empty()):      # 有最终任务积压就让位
                            next_partial[0] = wall_now + partial_interval_s
                            last_partial_dur[0] = pdur
                            try:
                                partial_q.put_nowait((pend["pcm"], pend["seg_id"]))
                            except queue.Full:
                                pass      # 上一次部分识别还没做完：丢这次，不排队
                for seg in segments:
                    piece = seg["pcm"]
                    dur = len(piece) / 16000
                    if dur < min_sentence_s:
                        continue
                    # 近静音护栏（能量门）：声音很小/只有环境底噪时，VAD 也会切出片段，
                    # 喂给 whisper 极易触发"复读幻觉"（实测整屏 "rover rover rover…"）。
                    # 这里直接丢弃：省一次 GPU 推理，也避免幻觉字幕。
                    piece_rms = _rms_int16(piece)
                    if piece_rms < min_rms:
                        self.stats["skipped_quiet"] = (
                            self.stats.get("skipped_quiet", 0) + 1)
                        continue
                    # 语音占比门：VAD 判定"有声"的帧占比过低 → 多半是音乐/纯伴奏。
                    # 2026-09-16 改：**不再整段丢弃**（旧行为在游戏实况里 5/13 段被丢，
                    # 用户看到的就是"放着放着不动了"）。改标记为 music_risk 照常送识别，
                    # 由识别后的门（no_speech_prob / 音乐标记 / 回声门 / 复读门）决定去留
                    # —— WhisperLive 的经验：VAD 只负责切段，音频一律送模型，事后过滤。
                    music_risk = (float(seg.get("speech_ratio", 1.0) or 0.0)
                                  < min_speech_ratio)
                    # 语音段结束的墙钟时刻（样本号 → 墙钟，fed_total 对应 wall_now）
                    seg_end_wall = wall_now - (fed_total - seg["end_sample"]) / 16000
                    # 队列满：丢最旧的，投递永不阻塞（见 except 分支）
                    self.on_activity("识别中…")
                    try:
                        work_q.put_nowait(("final", piece, seg_end_wall, dur,
                                           seg.get("seg_id"), music_risk,
                                           seg.get("cut", "")))
                    except queue.Full:
                        # 有积压：**丢最旧的一条**而不是等空位。
                        # 旧写法在这里阻塞等待，一旦 GPU 繁忙（例如同时打游戏）
                        # 采集+VAD 循环就整体停住 → 后面完全不出字幕，表现为"卡死"。
                        try:
                            work_q.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            work_q.put_nowait(("final", piece, seg_end_wall, dur,
                                               seg.get("seg_id"), music_risk,
                                               seg.get("cut", "")))
                        except queue.Full:
                            pass
                        if not backlog_warned[0]:
                            backlog_warned[0] = True
                            self.on_state("warn: 处理积压（GPU 繁忙），已丢弃较旧的语音段")
        finally:
            try:
                work_q.put(None, timeout=2)   # 别让退出卡在满队列上
            except Exception:
                pass
            worker_thread.join(timeout=10)
            try:
                partial_q.put_nowait(None)
            except Exception:
                pass
            partial_thread.join(timeout=5)
            cap.stop()
            cap.terminate()
            if probe_pa is not None:
                try:
                    probe_pa.terminate()
                except Exception:
                    pass
            if gpu_worker is not None:
                gpu_worker.stop()          # 收掉 GPU 子进程（否则退出后它还挂着显存）
            self.on_state("stopped")
