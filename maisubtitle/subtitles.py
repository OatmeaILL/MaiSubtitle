"""字幕导出：SRT / ASS，单语与双语，时间戳校正偏移。"""
from dataclasses import dataclass


@dataclass
class Cue:
    start: float   # 秒
    end: float
    src: str
    dst: str = ""


def _offset(cues: list[Cue], offset: float) -> list[Cue]:
    if offset == 0:
        return cues
    return [Cue(max(0.0, c.start + offset), max(0.0, c.end + offset), c.src, c.dst)
            for c in cues]


def _srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ass_time(t: float) -> str:
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def write_srt(cues: list[Cue], path: str, bilingual: bool = True,
              offset: float = 0.0):
    cues = _offset(cues, offset)
    with open(path, "w", encoding="utf-8-sig", newline="\n") as f:
        for i, c in enumerate(cues, 1):
            f.write(f"{i}\n{_srt_time(c.start)} --> {_srt_time(c.end)}\n")
            if bilingual and c.dst:
                f.write(f"{c.src}\n{c.dst}\n\n")
            elif bilingual:
                f.write(f"{c.src}\n\n")
            else:
                f.write(f"{c.dst or c.src}\n\n")


ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Src,Microsoft YaHei,42,&H00E8E8E8,&H000000FF,&H00000000,&H7F000000,0,0,0,0,100,100,0,0,1,2,1,2,60,60,58,1
Style: Dst,Microsoft YaHei,52,&H00FFFFFF,&H000000FF,&H00000000,&H7F000000,-1,0,0,0,100,100,0,0,1,2,1,2,60,60,100,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def write_ass(cues: list[Cue], path: str, bilingual: bool = True,
              offset: float = 0.0):
    r"""双语=译文大字在上、原文小字在下（\N 分行）。"""
    cues = _offset(cues, offset)
    with open(path, "w", encoding="utf-8-sig", newline="\n") as f:
        f.write(ASS_HEADER)
        for c in cues:
            if bilingual and c.dst:
                text = f"{c.dst}\\N{c.src}"
            elif bilingual:
                text = c.src
            else:
                text = c.dst or c.src
            f.write(f"Dialogue: 0,{_ass_time(c.start)},{_ass_time(c.end)},Dst,,0,0,0,,{text}\n")
