"""GPU 能力三级探测：驱动 → CTranslate2/CUDA → onnxruntime DirectML。

输出 JSON 结论，供各 POC 与正式管线决定运行档位：
  gpu_full   : CUDA 可用（ASR/翻译都能上 GPU）
  dml_only   : 仅 DirectML（只能加速 onnxruntime，如 VAD；CT2/sherpa 走 CPU）
  cpu_only   : 纯 CPU
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poc.common.paths import register_nvidia_dlls  # noqa: E402


def probe_driver():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        return out or None
    except Exception:
        return None


def probe_ct2_cuda():
    try:
        import ctranslate2
        n = ctranslate2.get_cuda_device_count()
        info = {"cuda_device_count": n}
        if n > 0:
            # 实际初始化一次，验证 cuDNN/cuBLAS 链路完整
            try:
                specs = sorted(ctranslate2.get_supported_compute_types("cuda"))
                info["cuda_compute_types"] = specs
                info["cuda_ok"] = True
            except Exception as e:
                info["cuda_ok"] = False
                info["cuda_error"] = str(e)[:200]
        else:
            info["cuda_ok"] = False
        return info
    except Exception as e:
        return {"cuda_ok": False, "cuda_error": str(e)[:200]}


def probe_dml():
    try:
        import onnxruntime as ort
        return {
            "ort_version": ort.__version__,
            "providers": ort.get_available_providers(),
            "dml_ok": "DmlExecutionProvider" in ort.get_available_providers(),
        }
    except Exception as e:
        return {"dml_ok": False, "error": str(e)[:200]}


def main():
    result = {
        "driver": probe_driver(),
        "dll_dirs_registered": register_nvidia_dlls(),
        "ctranslate2": probe_ct2_cuda(),
        "onnxruntime": probe_dml(),
    }
    ct2 = result["ctranslate2"]
    if ct2.get("cuda_ok"):
        result["mode"] = "gpu_full"
    elif result["onnxruntime"].get("dml_ok"):
        result["mode"] = "dml_only"
    else:
        result["mode"] = "cpu_only"
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
