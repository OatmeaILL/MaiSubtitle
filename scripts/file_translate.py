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
    p.add_argument("input", nargs="+", help="视频/音频文件（可给多个）")
    p.add_argument("-o", "--out", default=None, help="输出前缀（默认与输入同名）")
    p.add_argument("--src", default="auto", choices=["auto", "en", "ja", "ko", "zh"],
                   help="源语言，auto=自动检测；zh=中文（繁→简，不翻译）")
    p.add_argument("--engine", default="qwen",
                   choices=["hymt2", "qwen", "qwen3"])
    p.add_argument("--model", default="large-v3-turbo", choices=["large-v3-turbo"])
    p.add_argument("--glossary", default=None, help="术语库 CSV/JSON 路径")
    p.add_argument("--mono", action="store_true", help="仅译文")
    p.add_argument("--formats", default="srt,ass", help="srt,ass 组合")
    p.add_argument("--offset", type=float, default=0.0,
                   help="时间戳偏移秒（正=字幕整体后移，如录屏比原声晚开；负=前移）")
    args = p.parse_args()
    # --formats 校验：以前随便填（如 vtt）会静默不产出任何文件，最后只报"完成"
    _fmts = [x.strip().lower() for x in args.formats.split(",") if x.strip()]
    _bad = [f for f in _fmts if f not in ("srt", "ass")]
    if _bad:
        p.error(f"--formats 只支持 srt/ass 的组合，收到未知格式: {','.join(_bad)}")
    args.formats = ",".join(_fmts)

    if len(args.input) > 1 and args.out:
        p.error("--out 只适用于单个输入文件")

    def progress(stage, i, n, done_s=0.0, total_s=0.0, elapsed_s=0.0):
        if stage == "decode":
            print("解码中…", flush=True)
        elif stage == "vad":
            print(f"分句：{n} 段", flush=True)
        elif stage == "asr" and n:
            eta = elapsed_s / max(done_s, 0.1) * max(total_s - done_s, 0.0)
            print(f"\r识别+翻译 {i}/{n}（{done_s:.0f}/{total_s:.0f}s）"
                  f"· 已用 {elapsed_s:.0f}s · 剩余约 {eta:.0f}s   ",
                  end="", flush=True)
    print()

    failed = 0
    total_cues = 0
    for inp in args.input:
        try:
            print(f"== {inp}")
            warns: list = []
            cues = translate_file(inp, args.out, src=args.src, engine=args.engine,
                                  asr_model=args.model, glossary_path=args.glossary,
                                  bilingual=not args.mono,
                                  formats=tuple(x.strip() for x in args.formats.split(",")),
                                  offset=args.offset, progress=progress,
                                  warnings=warns)
            total_cues += len(cues)
            print(f"\n完成：{len(cues)} 条字幕")
            out_prefix = Path(args.out or Path(inp).with_suffix(""))
            for f in args.formats.split(","):
                fp = Path(str(out_prefix) + "." + f.strip())
                if fp.exists():
                    print(f"  {fp}  ({fp.stat().st_size // 1024} KB)")
            for w in warns:
                print(f"[注意] {w}")
        except Exception as e:
            failed += 1
            print(f"\n[错误] {inp}: {type(e).__name__}: {e}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
