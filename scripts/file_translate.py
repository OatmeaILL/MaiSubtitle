"""阶段1 CLI：视频/音频文件 → 双语字幕（SRT/ASS）。

用法：
    uv run python scripts/file_translate.py 视频.mp4
    uv run python scripts/file_translate.py 视频.mp4 -o 输出名 --src ja --mono
    uv run python scripts/file_translate.py 视频.mp4 --glossary myterms.csv
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from maisubtitle.offline import translate_file  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="离线文件 → 双语字幕")
    p.add_argument("input")
    p.add_argument("-o", "--out", default=None, help="输出前缀（默认与输入同名）")
    p.add_argument("--src", default="auto", choices=["auto", "en", "ja", "ko"],
                   help="源语言，auto=自动检测")
    p.add_argument("--engine", default="qwen",
                   choices=["hymt2", "qwen", "qwen3"])
    p.add_argument("--model", default="large-v3-turbo", choices=["large-v3-turbo"])
    p.add_argument("--glossary", default=None, help="术语库 CSV/JSON 路径")
    p.add_argument("--mono", action="store_true", help="仅译文")
    p.add_argument("--formats", default="srt,ass", help="srt,ass 组合")
    p.add_argument("--offset", type=float, default=0.0, help="时间戳偏移秒")
    args = p.parse_args()

    def progress(stage, i, n):
        if stage in ("asr",) and n:
            print(f"\r识别中 {i}/{n}", end="", flush=True)
    print()

    cues = translate_file(args.input, args.out, src=args.src, engine=args.engine,
                          asr_model=args.model, glossary_path=args.glossary,
                          bilingual=not args.mono,
                          formats=tuple(x.strip() for x in args.formats.split(",")),
                          offset=args.offset, progress=progress)
    out_prefix = Path(args.out or Path(args.input).with_suffix(""))
    print(f"\n完成：{len(cues)} 条字幕")
    for f in args.formats.split(","):
        fp = Path(str(out_prefix) + "." + f.strip())
        if fp.exists():
            print(f"  {fp}  ({fp.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
