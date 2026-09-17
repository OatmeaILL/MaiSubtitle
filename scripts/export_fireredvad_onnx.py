# -*- coding: utf-8 -*-
"""把 FireRedVAD Stream-VAD 从 PyTorch 导出成 ONNX（CPU 推理，主程序无需 torch）。

    .venv\\Scripts\\python.exe scripts/export_fireredvad_onnx.py

产出（models/fireredvad-onnx/）：
  stream_vad.onnx   — DetectModel（含 KV/FSMN 缓存，动态 T）
  cmvn.npz          — CMVN 均值与逆标准差（避免运行时依赖 kaldiio）
  meta.json         — 帧参数（帧长/帧移/mel 数）与导出信息

注意：特征提取用 kaldi_native_fbank（C++ 扩展，不需要 torch），
      运行时依赖 = onnxruntime + numpy + kaldi_native_fbank。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="models/fireredvad/Stream-VAD")
    ap.add_argument("--out", default="models/fireredvad-onnx")
    ap.add_argument("--frames", type=int, default=1, help="导出时的最小帧数（动态轴）")
    args = ap.parse_args()

    src = ROOT / args.src
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    if not (src / "model.pth.tar").exists():
        print(f"缺少权重: {src}")
        return 1

    from fireredvad.core.detect_model import DetectModel
    from fireredvad.core.audio_feat import CMVN

    model = DetectModel.from_pretrained(str(src))
    model.eval()

    # --- 1) 探缓存结构：先空跑一次拿到 caches 的形状 ---
    a = model.dfsmn.fc1[0].in_features      # 输入维度（mel 数）
    with torch.no_grad():
        dummy = torch.zeros(1, 8, a)
        _probs, caches = model.forward(dummy, caches=None)
    n_cache = len(caches)
    cache_shapes = [tuple(c.shape) for c in caches]
    print(f"输入维度 idim={a}｜缓存 {n_cache} 个｜形状 {cache_shapes}")
    print(f"输出 probs 形状 {tuple(_probs.shape)}")

    # --- 2) 导出 ONNX（T 动态） ---
    onnx_path = out / "stream_vad.onnx"
    T = args.frames
    feat = torch.zeros(1, T, a)
    zeros = [torch.zeros(s) for s in cache_shapes]
    in_names = ["feat"] + [f"cache{i}" for i in range(n_cache)]
    out_names = ["prob"] + [f"out_cache{i}" for i in range(n_cache)]
    # 动态轴：feat/prob 的时间维是 axis 1；**缓存的时间维是 axis 2**
    # （缓存最后一维是 FSMN 记忆窗，最大 19 帧，会随累计长度增长到 19 后封顶）
    dynamic = {"feat": {1: "T"}, "prob": {1: "T"}}
    for i in range(n_cache):
        dynamic[f"cache{i}"] = {2: "Tc"}
        dynamic[f"out_cache{i}"] = {2: "Tc"}

    with torch.no_grad():
        torch.onnx.export(
            model, (feat, zeros), str(onnx_path),
            input_names=in_names, output_names=out_names,
            dynamic_axes=dynamic, opset_version=13,
            do_constant_folding=True)
    print(f"已导出: {onnx_path} ({onnx_path.stat().st_size/1e6:.2f} MB)")

    # --- 3) CMVN 落盘（运行时不再需要 kaldiio） ---
    cmvn = CMVN(str(src / "cmvn.ark"))
    np.savez(out / "cmvn.npz", means=cmvn.means, istd=cmvn.inverse_std_variances)
    print(f"已导出: {out/'cmvn.npz'} (dim={cmvn.dim})")

    # --- 4) meta ---
    meta = {"idim": int(a), "n_cache": int(n_cache),
            "cache_shapes": [list(s) for s in cache_shapes],
            "frame_length_ms": 25.0, "frame_shift_ms": 10.0,
            "sample_rate": 16000, "mel_bins": int(a),
            "src_model": str(args.src), "opset": 13}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    print(f"已导出: {out/'meta.json'}")

    # --- 5) 数值对齐校验：torch vs onnxruntime ---
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.RandomState(0)
    feat_np = rng.randn(1, 16, a).astype(np.float32)
    with torch.no_grad():
        p_t, c_t = model.forward(torch.from_numpy(feat_np),
                                 caches=[torch.zeros(s) for s in cache_shapes])
    feed = {"feat": feat_np}
    for i in range(n_cache):
        feed[f"cache{i}"] = np.zeros(cache_shapes[i], dtype=np.float32)
    o = sess.run(None, feed)
    d_prob = float(np.abs(o[0] - p_t.numpy()).max())
    d_cache = max(float(np.abs(o[1 + i] - c_t[i].numpy()).max()) for i in range(n_cache))
    print(f"\n数值对齐：prob 最大差 {d_prob:.2e}｜cache 最大差 {d_cache:.2e}")
    ok = d_prob < 1e-4 and d_cache < 1e-4
    print("校验:", "PASS ✓（ONNX 与 torch 一致）" if ok else "FAIL ✗")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
