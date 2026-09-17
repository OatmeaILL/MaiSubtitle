"""把 Qwen 官方权重转换为 CTranslate2 int8_float16（翻译模型，装环境的关键一步）。

前置：HF 权重已下载
      1.5b → models/Qwen2.5-1.5B-Instruct-hf/（download_models.py --only qwen_1_5b_hf）
      1.7b → models/Qwen3-1.7B-hf/         （download_models.py --only qwen3_1_7b_hf）
      需要 transformers + torch（本机 venv 通过 zz-maisub-external-torch.pth 借外部 CUDA torch；
      其他环境用 pip install torch transformers 即可，CPU 版也行，转换只用 CPU）

产出：models/<对应>-ct2/，并**补拷 tokenizer_config.json**（带对话模板）——
      QwenCT2 的轻量 tokenizer 靠它渲染提示词，缺了会起不来。

用法：uv run python scripts/convert_qwen_ct2.py [1.5b|1.7b]
"""
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from maisubtitle.config import MODELS_DIR  # noqa: E402

SPECS = {
    "1.5b": ("Qwen2.5-1.5B-Instruct-hf", "Qwen2.5-1.5B-Instruct-ct2", "qwen_1_5b_hf"),
    "1.7b": ("Qwen3-1.7B-hf", "Qwen3-1.7B-ct2", "qwen3_1_7b_hf"),
}
# CT2 转换器默认只拷需要的几个文件；这几个必须手动补（对话模板/特殊 token）
EXTRA_FILES = ("tokenizer_config.json", "special_tokens_map.json", "generation_config.json")


def main():
    key = (sys.argv[1] if len(sys.argv) > 1 else "1.5b").strip().lower()
    if key not in SPECS:
        print(f"未知规格: {key}（可用: {'/'.join(SPECS)}）")
        return 2
    src_name, dst_name, dl_item = SPECS[key]
    src, dst = MODELS_DIR / src_name, MODELS_DIR / dst_name
    if not src.exists() or not any(src.glob("*.safetensors")):
        print(f"缺少 HF 权重: {src}\n  先运行: python scripts/download_models.py --only {dl_item}")
        return 2
    if (dst / "model.bin").exists():
        print(f"已转换过，跳过（{dst}）")
        return 0
    print(f"开始转换 {src_name} → {dst_name}（int8_float16，需几分钟）…")
    t0 = time.perf_counter()
    subprocess.run([
        sys.executable, "-m", "ctranslate2.converters.transformers",
        "--model", str(src), "--output_dir", str(dst),
        "--quantization", "int8_float16",
        "--copy_files", "tokenizer.json",
    ], check=True)
    for extra in EXTRA_FILES:
        if (src / extra).exists():
            shutil.copy2(src / extra, dst / extra)
    print(f"转换完成，用时 {time.perf_counter() - t0:.0f}s → {dst}")
    print("（设置里的「翻译引擎」用默认 qwen 即可；qwen3 需 1.7b 版）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())