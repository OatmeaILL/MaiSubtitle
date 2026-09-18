# -*- coding: utf-8 -*-
"""部署跑分：本机装了哪些模型 → VAD × 识别 × 翻译 全组合实测 → 程序推荐最优搭配。

    .venv\\Scripts\\python.exe scripts/bench_deploy.py              # 跑分（4 语种切片）
    .venv\\Scripts\\python.exe scripts/bench_deploy.py --list         # 只看本机有哪些模型
    .venv\\Scripts\\python.exe scripts/bench_deploy.py --no-judge     # 不调 LLM 打分（快一倍）

为什么要它
  用户在自己电脑上部署时最想知道的是"我该选哪套搭配"。所以这个脚本：
    1. **只对已下载的模型开放跑分** —— 没下的那部分不参与，只列出来并给出补下命令
       （不拿"以为有其实会去联网下"的模型去跑，避免卡 100s 超时）；
    2. 对 VAD × 识别 × 翻译 做**全组合**实测，素材是 4 种语言的真实切片
       （testdata/_clip_{en,ja,ko,zh}.wav，随仓库发布；参考文本 _clip_*.ref.txt）；
    3. 记录 分句结果 / 识别结果与延迟 / 翻译结果与延迟 / 总延迟 / 出现的异常；
    4. 用提示词注入翻译模型让它给自己打分（**仅供参考**：同一模型自评有自我偏好），
       推荐排名以**程序算法**为准 —— 质量与延迟加权，并单列"延迟最低"榜。

打分口径
  识别准确率 line_score：对每句参考文本，在识别出的所有段里找最相似的一条（difflib），
      取平均 ×100 —— **同时惩罚切错句和识别错**，与 grand_prix.py 同口径，可对照。
  识别字符准确率 char_acc：全文去空白后的字符级准确率（1 − 编辑距离/参考长度）。
  译文质量 judge：把原文与译文交给翻译模型自己打分（0~100，仅参考）。
  术语保护率 term_on/term_off：带/不带术语库两次翻译，指定译名在译文里出现的比例。
  延迟 asr_ms / mt_ms / e2e_ms / onset_ms（中位数）：onset = 段长 + 识别 + 翻译，
      是用户真正体感的延迟（与 HANDOVER §五 口径一致）。
  综合分 = 识别准确率×0.55 + 译文质量×0.30 + 延迟分×0.15 − 异常扣分
      延迟分 = 1500ms 以内 100 分，之后每多 50ms 扣 1 分（最低 0）。
      **推荐 = 综合分最高**；同分按体感延迟更低优先。

引擎初始化铁律：同进程 ORT CUDA 会话必须先于 CT2 创建（否则 cudnnGetLibConfig 127 直接退出），
所以子进程里先建 Qwen3-ASR（ORT），再建 whisper（CT2）与翻译引擎（CT2）。
识别引擎按识别后端分进程跑（全部驻留会撑爆 8GB 显存、慢 10 倍）；
翻译模型多于 --max-mts-per-proc 时再按批拆进程（同一批共用子进程）。
"""
import argparse
import difflib
import json
import re
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from maisubtitle.audio import decode_media  # noqa: E402
from maisubtitle.config import MODELS_DIR, AppConfig  # noqa: E402
from model_manager import CATALOG, dir_size_mb, model_path  # noqa: E402

# ---- 候选（键名 → 模型管理器里的下载项 + 展示名）----
# 只列**本地有权重**的：火山流式识别是云端服务（没有本地权重可下载），
# 它在这里的角色是"给切片产出标准参考文本"（见 HANDOVER 跑分一节），不参与跑分。
ASR_CHOICES = [
    ("whisper", "whisper_turbo", "Whisper large-v3-turbo"),
    ("qwen3-onnx", "qwen3_asr", "Qwen3-ASR-0.6B ONNX"),
]
VAD_CHOICES = [
    ("firered", "firered", "FireRedVAD"),
    ("fsmn", "fsmn_vad", "FSMN-VAD"),
    ("silero", "silero_vad", "Silero VAD"),
]
MT_CHOICES = [
    ("hymt2", "hymt2", "Hy-MT2-1.8B"),
    ("qwen", "qwen_1_5b_ct2", "Qwen2.5-1.5B-CT2"),
    ("qwen3", "qwen3_1_7b_ct2", "Qwen3-1.7B-CT2"),
]

# 就绪判据必须是**真权重**文件（不能拿 config.json 顶：下到一半也会被判成已就绪，
# 见 HANDOVER §七十七 / 同类坑 §六十九）。
READY_MARK = {
    "whisper_turbo": "model.bin",
    "qwen3_asr": "encoder.int4.onnx",
    "hymt2": "model.safetensors",
    "qwen_1_5b_ct2": "model.bin",
    "qwen3_1_7b_ct2": "model.bin",
    "firered": "stream_vad.onnx",
    "fsmn_vad": "model_quant.onnx",
    "silero_vad": "",
}

CLIPS = ["en", "ja", "ko", "zh"]
LANG_NAME = {"en": "英语", "ja": "日语", "ko": "韩语", "zh": "中文"}

# 跑分用的人造术语库：从 4 份参考文本里挑的专有名词 / 外来词（真实术语库由用户自备）。
# 目的：验证"术语库到底有没有用" —— 同一批句子翻两次（不带 / 带术语），比指定译名的出现率。
# ⚠ 识别侧偏置只喂规范写法（aliases 进提示词会让模型照抄错写法，见 HANDOVER §五）。
BENCH_TERMS = {
    "en": [("football scores", "足球比分"), ("podcast", "播客"),
           ("technology", "科技"), ("website", "网站")],
    "ja": [("リューネル王国", "琉尼尔王国"), ("シルヴァリオの塔", "西尔瓦里奥之塔"),
           ("アステリアの祭り", "阿斯特里亚祭典"), ("ヴェルミリオンの旗", "维尔米利翁之旗"),
           ("ソルディア川", "索尔迪亚河"), ("ナルヴィアの遺跡", "纳尔维亚遗迹"),
           ("エンデュミオン号", "恩底弥翁号")],
    "ko": [("인테리어", "室内装修"), ("리메이크", "重制版"), ("썸네일", "缩略图"),
           ("파생형", "衍生型"), ("수영장", "游泳池")],
    "zh": [],                       # 源文就是中文 → 不做术语测试
}

JUDGE_SYS = (
    "你是字幕质量评审。给定外语原文和它的中文译文，按 0~100 打分。"
    "评分维度：忠实（无漏译、无编造）、通顺（像人话的中文口语）、专有名词正确。"
    "只输出一行 JSON，不要任何解释，格式：{\"score\": 85, \"why\": \"不超过20字的理由\"}")
JUDGE_USER = "原文（{lang}）：\n{src}\n\n译文（中文）：\n{zh}\n\n请只输出一行 JSON。"


# ---------------- 本机模型可用性（只对已下载的开放跑分）----------------

def model_ready(name: str) -> bool:
    """下载项是否真的就绪（判据是真权重文件，不是"目录存在"）。"""
    mark = READY_MARK.get(name)
    p = model_path(name)
    if not mark:
        return dir_size_mb(p) > 0.1
    return (p / mark).exists()


def model_mb(name: str) -> float:
    return dir_size_mb(model_path(name))


def available_choices(choices, skip_remote=True):
    """返回 (可用, 缺失)。缺失项带 model_manager 的下载名，直接给补下命令。"""
    ok, miss = [], []
    for key, dl, label in choices:
        if key == "qwen3-onnx":
            # Qwen3-ASR 的目录是自动侦测的（可能不在 models/ 下），跟 make_asr 共用一份判据
            from maisubtitle.asr import find_qwen_dir
            found = find_qwen_dir("") is not None
            mb = 0.0 if not found else model_mb(dl)
            ready = found
        else:
            ready, mb = model_ready(dl), model_mb(dl)
        (ok if ready else miss).append((key, dl, label, mb))
    return ok, miss


def print_models():
    cfg = AppConfig()
    print("=== 本机可用模型（只对已下载的开放跑分）===")
    ok_vad, miss_vad = available_choices(VAD_CHOICES)
    ok_asr, miss_asr = available_choices(ASR_CHOICES)
    ok_mt, miss_mt = available_choices(MT_CHOICES)
    for title, ok, miss in (("VAD 分句", ok_vad, miss_vad),
                            ("识别", ok_asr, miss_asr),
                            ("翻译", ok_mt, miss_mt)):
        print(f"[{title}]")
        for key, dl, label, mb in ok:
            print(f"  ✔ {label:26s} {mb:7.0f} MB")
        for key, dl, label, mb in miss:
            hint = CATALOG.get(dl, ("", "", "", False))[0]
            print(f"  ✘ {label:26s} 未下载 → python scripts/model_manager.py download {key}")
            if hint in ("__local__",):
                print("       （该项需按 model_manager list 的手动步骤自备）")
        print()
    print("跳过：云端后端（火山流式识别 asr_backend=ws / HTTP 服务）没有本地权重，不参与跑分")
    if not ok_vad:
        print("[注意] 本机没有可用的 VAD 模型 → 无法分句，跑分无法进行")
    if not ok_asr:
        print("[注意] 本机没有可用的识别模型 → 识别部分不开放")
    if not ok_mt:
        print("[注意] 本机没有可用的翻译模型 → 翻译部分不开放（只能出原文）")
    print(f"当前配置：识别 {cfg.asr_backend} / 翻译 {cfg.engine} / VAD {cfg.vad_engine}"
          f"（配置只是参考，跑分按本机实际有的模型全组合来）")
    print()
    return ok_vad, ok_asr, ok_mt


# ---------------- 打分（纯函数，可离线单测）----------------

def norm_text(t: str) -> str:
    t = (t or "").lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def norm_words(t: str):
    t = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in (t or "").lower())
    return t.split()


def line_score(ref_lines, hyp_lines):
    """返回 (识别准确率, 多出段比例)。

    对每句参考文本，在识别出的所有段里找最相似的一条；多出段（匹配不上任何参考句的
    识别段）单列成比例 —— 切太碎或幻觉会让它升高。与 grand_prix.py 同口径。
    """
    refs = [norm_text(r) for r in ref_lines if (r or "").strip()]
    hyps = [norm_text(h) for h in hyp_lines if (h or "").strip()]
    if not refs or not hyps:
        return 0.0, 1.0
    scores = []
    for r in refs:
        scores.append(max(difflib.SequenceMatcher(None, r, h).ratio() for h in hyps))
    matched = sum(1 for h in hyps
                  if max(difflib.SequenceMatcher(None, r, h).ratio() for r in refs) >= 0.55)
    return st.mean(scores) * 100, 1 - matched / len(hyps)


def char_acc(ref_lines, hyp_lines) -> float:
    """全文拼接后的字符级准确率（1 − 编辑距离 / 参考长度）。"""
    ref = "".join(norm_text(r) for r in ref_lines).replace(" ", "")
    hyp = "".join(norm_text(h) for h in hyp_lines).replace(" ", "")
    if not ref:
        return 0.0
    try:
        from Levenshtein import distance as _lev
    except Exception:                      # 依赖缺失时退化：全文相似度
        return difflib.SequenceMatcher(None, hyp, ref).ratio() * 100
    return max(0.0, 1 - _lev(hyp, ref) / len(ref)) * 100


def pct(values, p: float) -> float:
    v = sorted(values)
    if not v:
        return 0.0
    return v[min(int(len(v) * p), len(v) - 1)]


def has_cjk(t: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in (t or ""))


def zh_ratio(texts) -> float:
    texts = [t for t in texts if t]
    if not texts:
        return 0.0
    return sum(1 for t in texts if has_cjk(t)) / len(texts)


def len_ratio(src_texts, zh_texts) -> float:
    """译文长度 / 原文长度（只出一两个字、或大段解释会露出来）。"""
    a = sum(len(t or "") for t in src_texts)
    b = sum(len(t or "") for t in zh_texts)
    return b / a if a else 0.0


def parse_judge(out: str):
    """从模型回包里抠出 {"score": 85, ...} → (分数 | None, 理由)。"""
    m = re.search(r"\{.*\}", out or "", re.S)
    if not m:
        return None, ""
    try:
        js = json.loads(m.group(0))
    except Exception:
        return None, ""
    try:
        s = float(js.get("score"))
    except Exception:
        return None, str(js.get("why", ""))[:30]
    return max(0.0, min(100.0, s)), str(js.get("why", ""))[:30]


def chat(mt, msgs, max_tokens: int = 160) -> str:
    """统一调用翻译模型的对话接口（NullMT 没有 max_tokens 参数）。"""
    try:
        return mt.chat(msgs, max_tokens=max_tokens)
    except TypeError:
        return mt.chat(msgs)


def judge_pair(mt, lang: str, src: str, zh: str):
    """让翻译模型给"原文 → 译文"打分。返回 (分数 | None, 理由)。"""
    if not src or not zh:
        return None, ""
    msgs = [{"role": "system", "content": JUDGE_SYS},
            {"role": "user", "content": JUDGE_USER.format(
                lang=LANG_NAME.get(lang, lang), src=src[:400], zh=zh[:400])}]
    try:
        return parse_judge(chat(mt, msgs, max_tokens=64))
    except Exception as e:
        return None, f"<{type(e).__name__}>"


def term_rate(terms, zh_texts) -> float:
    """指定译名在译文里出现的比例（术语保护率）。"""
    if not terms or not zh_texts:
        return 0.0
    joined = "\n".join(t or "" for t in zh_texts)
    hit = sum(1 for _s, z in terms if z and z in joined)
    return hit / len(terms)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ---------------- 引擎加载 ----------------

def load_asr(key: str):
    """按识别后端构造识别引擎。**顺序铁律：ORT CUDA 会话先于 CT2 创建。**"""
    if key == "qwen3-onnx":
        from maisubtitle.asr_qwen_onnx import Qwen3AsrOnnx
        return Qwen3AsrOnnx(device="cuda")
    from maisubtitle.asr import WhisperASR
    return WhisperASR("large-v3-turbo", device="cuda", compute_type="float16")


def make_seg(vad: str, max_s: float, min_sil: float):
    if vad == "firered":
        from maisubtitle.vad_firered import FireRedVadSegmenter
        return FireRedVadSegmenter(str(MODELS_DIR / "fireredvad-onnx"),
                                  min_silence_ms=min_sil, max_speech_s=max_s)
    if vad == "fsmn":
        from maisubtitle.vad_fsmn import FsmnSegmenter
        return FsmnSegmenter(str(MODELS_DIR / "fsmn-vad-onnx"))
    from maisubtitle.vad import StreamingSegmenter
    return StreamingSegmenter(threshold=0.6, end_silence_ms=400,
                              max_segment_ms=max_s * 1000)


def load_clips(langs):
    """读 4 份切片 + 参考文本（参考文本来自 testdata/_clip_XX.ref.txt）。"""
    out = {}
    for lang in langs:
        wav = ROOT / "testdata" / f"_clip_{lang}.wav"
        ref = ROOT / "testdata" / f"_clip_{lang}.ref.txt"
        if not wav.exists():
            print(f"[注意] 缺切片 {wav} → 跳过 {lang}")
            continue
        pcm, dur = decode_media(str(wav))
        lines = []
        if ref.exists():
            lines = [x.strip() for x in ref.read_text(encoding="utf-8").splitlines()
                      if x.strip()]
        else:
            print(f"[注意] 缺参考文本 {ref} → {lang} 只记录结果不打分")
        out[lang] = {"pcm": pcm, "dur": dur, "ref": lines}
    return out


def run_vad(seg, pcm):
    """流式分句（100ms 分块，与真实运行同一节奏），返回 (段列表, 分句耗时 s)。"""
    t0 = time.perf_counter()
    segs = []
    for i in range(0, len(pcm), 1600):
        segs.extend(seg.feed(pcm[i:i + 1600]))
    return segs, time.perf_counter() - t0


def pick_judge_rows(rows, per_lang: int):
    """每个语种均匀挑几行做 LLM 打分（省时间，代表性够）。"""
    out = []
    for lang in CLIPS:
        rs = [r for r in rows if r["lang"] == lang]
        if not rs or per_lang <= 0:
            continue
        if len(rs) <= per_lang:
            out += rs
            continue
        step = len(rs) / per_lang
        for i in range(per_lang):
            out.append(rs[int(i * step)])
    return out


# ---------------- 子进程：真正跑分 ----------------

def run_child(args) -> int:
    import os
    # onnxruntime 的 C++ 警告在 Windows 控制台是宽字符乱码，压掉（不影响结果）
    os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")
    vads = [v for v in args.vads.split(",") if v]
    mts = [m for m in args.mts.split(",") if m]
    langs = [x for x in args.langs.split(",") if x]
    asr_key = args.asr

    clips = load_clips(langs)
    if not clips:
        print("[错误] 没有任何可用切片")
        return 2

    print(f"=== 子进程：识别 {asr_key} × 翻译 {mts} × VAD {vads} ===", flush=True)
    errors = []

    # 顺序铁律：ORT CUDA（Qwen3-ASR）先建，CT2（whisper / 翻译）后建
    try:
        asr = load_asr(asr_key)
        print(f"  识别 {asr_key} 就绪", flush=True)
    except Exception as e:
        print(f"[错误] 识别 {asr_key} 加载失败：{type(e).__name__}: {e}")
        return 3
    mts_ok = {}
    for k in mts:
        try:
            from maisubtitle.translate import make_translator
            mts_ok[k] = make_translator(k)
            print(f"  翻译 {k} 就绪（{type(mts_ok[k]).__name__}）", flush=True)
        except Exception as e:
            errors.append({"where": f"加载翻译 {k}", "error": f"{type(e).__name__}: {e}"})
            print(f"[错误] 翻译 {k} 加载失败：{type(e).__name__}: {e}", flush=True)
    vads_ok = {}
    for k in vads:
        try:
            vads_ok[k] = make_seg(k, args.max_seg_s, args.min_silence_ms)
            print(f"  VAD {k} 就绪", flush=True)
        except Exception as e:
            errors.append({"where": f"加载 VAD {k}", "error": f"{type(e).__name__}: {e}"})
            print(f"[错误] VAD {k} 加载失败：{type(e).__name__}: {e}", flush=True)

    # 热身：GPU 首次调用有编译/初始化开销，先暖机再计时
    for lang, c in clips.items():
        warm = c["pcm"][:16000 * 2]
        try:
            asr.transcribe(warm, language=lang)
        except Exception:
            pass
    for mt in mts_ok.values():
        try:
            mt.translate("warm up", "en")
        except Exception:
            pass
    print("  热身完成\n", flush=True)

    # 每个 VAD 分句一次，所有翻译组合共用（同一 ASR 在子进程里只跑一次）
    segs_by_vad = {}
    for vad, seg in vads_ok.items():
        per_lang = {}
        for lang, c in clips.items():
            segs, t_vad = run_vad(seg, c["pcm"])
            lens = [len(s["pcm"]) / 16000 for s in segs]
            per_lang[lang] = {
                "segs": segs, "vad_ms": round(t_vad * 1000, 1),
                "n_seg": len(segs),
                "seg_len_ms": round(st.mean(lens) * 1000, 1) if lens else 0.0,
                "max_seg_ms": round(max(lens) * 1000, 1) if lens else 0.0,
                "short_segs": sum(1 for x in lens if x < 1.0),
                "speech_ratio": round(st.mean([s.get("speech_ratio", 0.0)
                                                for s in segs]) if segs else 0.0, 3),
            }
            print(f"  VAD {vad:8s} {lang}：{len(segs)} 段｜平均 {per_lang[lang]['seg_len_ms']:.0f}ms"  # noqa: E501
                  f"｜最长 {per_lang[lang]['max_seg_ms']:.0f}ms"
                  f"｜短碎片 {per_lang[lang]['short_segs']}｜{t_vad*1000:.0f}ms", flush=True)
        segs_by_vad[vad] = per_lang
    print(flush=True)

    results = []
    for vad in vads_ok:
        # 识别：每个语种只跑一次（子进程只有一个识别引擎）
        rec = {}
        for lang, c in clips.items():
            texts, ms, errs = [], [], []
            for s in segs_by_vad[vad][lang]["segs"]:
                t1 = time.perf_counter()
                try:
                    txt = asr.transcribe(s["pcm"], language=lang)[0]
                except Exception as e:
                    txt = ""
                    errs.append(f"{type(e).__name__}: {e}")
                ms.append((time.perf_counter() - t1) * 1000)
                texts.append((txt or "").strip())
            rec[lang] = {"texts": texts, "ms": ms, "errors": errs}
            n_ok = sum(1 for t in texts if t)
            print(f"  识别 {vad:8s} {lang}：{n_ok}/{len(texts)} 段有输出"
                  f"｜{st.mean(ms) if ms else 0:.0f}ms/段"
                  + (f"｜异常 {len(errs)}" if errs else ""), flush=True)
        print(flush=True)

        for mname, mt in mts_ok.items():
            row = {"vad": vad, "asr": asr_key, "mt": mname, "langs": {},
                    "errors": list(errors)}
            for lang, c in clips.items():
                texts = rec[lang]["texts"]
                asr_ms = rec[lang]["ms"]
                segs = segs_by_vad[vad][lang]["segs"]
                seg_lens = [len(s["pcm"]) / 16000 * 1000 for s in segs]
                zh_texts, mt_ms = [], []
                pairs = []          # (识别, 译文) 配对：judge 抽样用（空识别段不翻译，索引别错位）
                for t in texts:
                    if not t:
                        continue
                    t2 = time.perf_counter()
                    try:
                        zh = mt.translate(t, lang)[0]
                    except Exception as e:
                        zh = ""
                        row["errors"].append({"where": f"翻译 {lang}", "error": f"{type(e).__name__}: {e}"})
                    mt_ms.append((time.perf_counter() - t2) * 1000)
                    zh_texts.append((zh or "").strip())
                    pairs.append((t, (zh or "").strip()))
                ls, extra = line_score(c["ref"], texts)
                acc = char_acc(c["ref"], texts) if c["ref"] else 0.0
                e2e = [a + m for a, m in zip(asr_ms, mt_ms)]
                onset = [l + e for l, e in zip(seg_lens, e2e)]
                info = {
                    "ref_n": len(c["ref"]), "hyp_n": len([t for t in texts if t]),
                    "line_score": round(ls, 1) if c["ref"] else None,
                    "extra": round(extra, 2) if c["ref"] else None,
                    "char_acc": round(acc, 1) if c["ref"] else None,
                    "asr_ms": round(pct(asr_ms, .5)), "asr_p90": round(pct(asr_ms, .9)),
                    "mt_ms": round(pct(mt_ms, .5)), "mt_p90": round(pct(mt_ms, .9)),
                    "e2e_ms": round(pct(e2e, .5)), "onset_ms": round(pct(onset, .5)),
                    "zh_ratio": round(zh_ratio(zh_texts), 3),
                    "len_ratio": round(len_ratio(texts, zh_texts), 2),
                    "rec": texts[:3], "zh": zh_texts[:3], "pairs": pairs,
                }
                # 术语对照：同一批句子翻两次（不带 / 带术语），看指定译名出现率
                if args.terms and BENCH_TERMS.get(lang):
                    terms = BENCH_TERMS[lang]
                    off = []
                    for t in texts[:args.term_rows]:
                        if not t:
                            continue
                        try:
                            off.append((mt.translate(t, lang)[0] or "").strip())
                        except Exception:
                            off.append("")
                    mt.set_terms(terms)
                    on = []
                    for t in texts[:args.term_rows]:
                        if not t:
                            continue
                        try:
                            on.append((mt.translate(t, lang)[0] or "").strip())
                        except Exception:
                            on.append("")
                    mt.set_terms([])
                    info["term_off"] = round(term_rate(terms, off), 2)
                    info["term_on"] = round(term_rate(terms, on), 2)
                    info["term_n"] = len(terms)
                row["langs"][lang] = info
            # LLM 打分（仅供参考）：每个语种挑几行，让该组合的翻译模型自评
            if args.judge:
                scores, whys = [], []
                for lang in clips:
                    rows = pick_judge_rows(
                        [{"lang": lang, "src": s, "zh": z} for s, z in
                         row["langs"][lang].get("pairs", [])], args.judge_rows)
                    for r in rows:
                        s, why = judge_pair(mt, lang, r["src"], r["zh"])
                        if s is not None:
                            scores.append(s)
                            whys.append(f"{lang}:{s:.0f} {why}")
                row["judge"] = round(st.mean(scores), 1) if scores else None
                row["judge_n"] = len(scores)
                row["judge_notes"] = whys[:6]
            else:
                row["judge"] = None
                row["judge_n"] = 0
            # judge 是组合级评分（抽样句来自各语种），同步进每个语种，方便分语种表显示
            for lang in row["langs"]:
                row["langs"][lang]["judge"] = row["judge"]
            row["score"] = combo_score(row)
            results.append(row)
            print(f"  {vad:8s}+{asr_key:10s}+{mname:6s} 识别 "
                  f"{avg_lang(row, 'line_score'):5.1f}｜译文 "
                  f"{avg_lang(row, 'judge') if row['judge'] is not None else -1:5.1f}｜"
                  f"onset {avg_lang(row, 'onset_ms'):5.0f}ms｜综合 {row['score']:5.1f}"
                  + (f"｜异常 {len(row['errors'])}" if row["errors"] else ""), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写入 {out}", flush=True)
    return 0


def avg_lang(row, key):
    vals = [v[key] for v in row["langs"].values() if v.get(key) is not None]
    return st.mean(vals) if vals else 0.0


def combo_score(row) -> float:
    """综合分 = 识别准确率×0.55 + 译文质量×0.30 + 延迟分×0.15 − 异常扣分。

    译文质量先看组合级 judge，再看各语种的（抽样分句判分后同步进 langs 的那份），
    都没有才退到启发式（译文含中文率：只出原文 / 复读原文会被压到 0）。
    """
    acc = avg_lang(row, "line_score")
    judge = row.get("judge")
    if judge is None:
        vals = [v["judge"] for v in row["langs"].values()
                if v.get("judge") is not None]
        judge = st.mean(vals) if vals else None
    if judge is None:      # 没跑 LLM 打分才退启发式（真打了 0 分要保留）
        judge = st.mean([v["zh_ratio"] * 100 for v in row["langs"].values()]) if row["langs"] else 0.0
    onset = avg_lang(row, "onset_ms")
    lat = clamp(100 - max(0.0, onset - 1500) / 50, 0.0, 100.0)
    s = acc * 0.55 + judge * 0.30 + lat * 0.15
    return round(max(0.0, s - min(len(row["errors"]) * 5, 20)), 1)


# ---------------- 主进程：调度子进程 + 汇总排名 ----------------

GPU_QUIET_MB = 1800.0     # 低于这个占用视为"GPU 空闲"（驱动 + 桌面底噪实测约 1.2GB；
                          # 主程序驻留识别+翻译引擎后 4GB 起，一眼能区分）


def gpu_used_mb() -> float:
    """当前显存占用（MB）；探测不到（无 N 卡 / 驱动异常）返回 -1，由调用方跳过检查。"""
    try:
        import subprocess as _sp
        r = _sp.run(["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, timeout=10)
        return float(r.stdout.decode("gbk", "replace").strip().splitlines()[0])
    except Exception:
        return -1.0


def wait_gpu_free(timeout_s: float = 60.0, force: bool = False):
    """GPU 被占时先等占用方退出 —— **跑分和主程序抢 GPU 会让数据严重失真**。

    实锤：主程序驻留 whisper+hymt2 时再跑分 = 显存挤兑，识别 320ms → 11750ms/段
    （慢 30 倍，2026-09-18 冒烟实测）。设置里的「部署跑分」会先退出主程序再弹本窗口，
    但用户手动跑分时可能忘关主程序 → 这里兜底：等它退出（最多 60s），超时问一句。
    """
    if force:
        return
    used = gpu_used_mb()
    if used < 0 or used <= GPU_QUIET_MB:
        return
    print(f"[注意] GPU 已被占用 {used:.0f}MB（多半是 MaiSubtitle 主程序还开着，"
          "或别的程序在用显卡）")
    print("       跑分要独占 GPU，否则结果失真。等它退出（最多 60s）…", flush=True)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(2)
        used = gpu_used_mb()
        if 0 <= used <= GPU_QUIET_MB:
            print("       GPU 已空闲，开始跑分\n", flush=True)
            return
    try:
        r = input(f"等待超时（仍占 {used:.0f}MB）。先退出占用 GPU 的程序再跑分最准；"
                  "仍要继续吗？[y/N] ").strip().lower()
    except EOFError:
        r = ""
    if r != "y":
        print("[未执行] 已取消。退出 MaiSubtitle / 其它占显卡的程序后重跑即可。")
        sys.exit(2)
    print("[注意] 你选择了继续：以下延迟数据偏高，仅供参考\n", flush=True)


def run_parent(args) -> int:
    ok_vad, ok_asr, ok_mt = print_models()
    if args.list:
        return 0
    if not ok_vad or not ok_asr or not ok_mt:
        print("[错误] 三样里缺了东西，跑分无法完整进行（见上面的补下命令）")
        if not ok_vad or not ok_asr:
            return 2
        print("[注意] 只有识别没有翻译 → 只跑识别部分（译文留空）")

    wait_gpu_free(force=args.force)

    langs = [x for x in args.langs.split(",") if x]
    vads = [k for k, _d, _l, _m in ok_vad]
    asrs = [k for k, _d, _l, _m in ok_asr]
    mts = [k for k, _d, _l, _m in ok_mt]
    if not mts:
        print("[注意] 没有可用翻译模型 → 只跑识别（无法评估译文）")

    batches = [mts[i:i + args.max_mts_per_proc]
                for i in range(0, len(mts), args.max_mts_per_proc)] or [[]]
    print(f"=== 跑分组合：{len(vads)} VAD × {len(asrs)} 识别 × {len(mts)} 翻译 = "
          f"{len(vads) * len(asrs) * len(mts)} 组 ===")
    print(f"    按识别引擎分 {len(asrs)} 个子进程；翻译模型每批 ≤{args.max_mts_per_proc} 个 "
          f"→ 共 {len(asrs) * len(batches)} 个子进程（全驻留会撑爆 8GB 显存）")
    print(f"    素材：{len(langs)} 个语种切片（testdata/_clip_*.wav）；"
          f"LLM 打分 {'开（每语种 ' + str(args.judge_rows) + ' 句）' if args.judge else '关'}")
    print(flush=True)

    all_rows = []
    failed = []
    for asr in asrs:
        for bi, batch in enumerate(batches, 1):
            out = ROOT / "logs" / f"bench_deploy_{asr}_b{bi}.json"
            if out.exists():
                out.unlink()
            cmd = [sys.executable, str(Path(__file__)), "--child", "--asr", asr,
                   "--vads", ",".join(vads), "--mts", ",".join(batch),
                   "--langs", ",".join(langs), "--out", str(out),
                   "--max-seg-s", str(args.max_seg_s),
                   "--min-silence-ms", str(args.min_silence_ms),
                   "--judge-rows", str(args.judge_rows),
                   "--term-rows", str(args.term_rows)]
            if args.judge:
                cmd.append("--judge")
            if args.terms:
                cmd.append("--terms")
            print(f"--- 子进程 {asr}（翻译批 {bi}/{len(batches)}：{batch}）---", flush=True)
            r = subprocess.run(cmd)
            # 子进程失败要**传出去**，还要验产物在不在（"退出码 0 但没产出"也算失败）
            if r.returncode != 0 or not out.exists():
                failed.append(f"{asr} 批{bi}（退出码 {r.returncode}，"
                               f"产物{'在' if out.exists() else '没生成'}）")
                print(f"[错误] 子进程 {asr} 批{bi} 失败", flush=True)
                continue
            all_rows += json.loads(out.read_text(encoding="utf-8"))
            print(flush=True)

    if not all_rows:
        print("[错误] 所有子进程都失败了，没有跑分结果")
        return 1
    if failed:
        print(f"[注意] {len(failed)} 个子进程失败：{'；'.join(failed)}")
        print("       → 重跑该子进程即可（可加 --mts 只跑缺的那几个翻译模型）")

    print_results(all_rows, langs, args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"clips": langs, "n_combo": len(all_rows),
         "failed_children": failed, "rows": all_rows},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n完整结果（含每段识别/译文与分句明细）已写入 {out}")
    write_ref_result(all_rows, langs, failed)
    return 0


def write_ref_result(rows, langs, failed):
    """跑分结果落一份**精简版**到 docs/bench_dev.json，随 git 发布给用户参考。

    用户在自己机器上跑完，最想要的是"开发机跑出来什么样、我的机器差在哪"——
    原始 JSON（logs/）含每段识别/译文文本，几百 KB 且 logs/ 不进 git；
    这里只留关键指标（去掉 rec/zh/pairs 大文本），docs/ 随仓库与发布包走。
    docs/ 不存在（如发布工作副本）就静默跳过，不影响主流程。
    """
    slim_rows = []
    for r in rows:
        slim_rows.append({
            "vad": r["vad"], "asr": r["asr"], "mt": r["mt"], "score": r["score"],
            "judge": r.get("judge"), "judge_n": r.get("judge_n", 0),
            "judge_notes": r.get("judge_notes", []),
            "n_err": len(r.get("errors", [])),
            "langs": {lg: {k: v.get(k) for k in
                           ("ref_n", "hyp_n", "line_score", "char_acc", "extra",
                            "asr_ms", "asr_p90", "mt_ms", "mt_p90", "e2e_ms",
                            "onset_ms", "zh_ratio", "len_ratio",
                            "term_on", "term_off", "term_n", "judge")}
                      for lg, v in r["langs"].items()},
        })
    doc = {
        "note": "开发机（RTX 4060 Laptop 8GB）部署跑分结果，供你自己跑分后对照参考。"
                "跑法：python scripts/bench_deploy.py；切片 testdata/_clip_*.wav。"
                "onset = 段长 + 识别 + 翻译（体感延迟）；judge = 翻译模型自评（仅供参考）。",
        "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU (8GB)",
        "clip_sources": {"en": "BBC Real Easy English 播客（参考文本=火山识别整理）",
                          "ja": "TTS 逐句拼接（参考文本=逐句原文）",
                          "ko": "韩语游戏解说（参考文本=火山+whisper 双源共识）",
                          "zh": "TTS 逐句拼接（参考文本=逐句原文）"},
        "clips": langs, "failed_children": failed, "rows": slim_rows,
    }
    try:
        dst = ROOT / "docs" / "bench_dev.json"
        if (ROOT / "docs").is_dir():
            dst.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                           encoding="utf-8")
            print(f"精简参考结果已写入 {dst}（随 git 发布，用户跑分后可对照）")
    except OSError as e:
        print(f"[注意] 参考结果没写成（不影响跑分）：{e}")


def print_results(rows, langs, args):
    print()
    print("=" * 108)
    print("=== 跑分结果（4 语种切片平均；onset = 段长 + 识别 + 翻译）===")
    print("=" * 108)
    head = (f"{'排名':<4}{'VAD':<10}{'识别':<12}{'翻译':<10}"
            f"{'识别准':>7}{'字符准':>7}{'译文质':>7}{'术语':>7}{'onset':>9}{'综合':>7}{'异常':>5}")
    print(head)
    ranked = sorted(rows, key=lambda r: (-r["score"], avg_lang(r, "onset_ms")))
    for i, r in enumerate(ranked, 1):
        jd = r.get("judge")
        jd_s = f"{jd:7.1f}" if jd is not None else "      -"
        tr = [v["term_on"] for v in r["langs"].values() if "term_on" in v]
        tr_s = f"{st.mean(tr)*100:6.0f}%" if tr else "      -"
        print(f"{i:<4}{r['vad']:<10}{r['asr']:<12}{r['mt']:<10}"
              f"{avg_lang(r, 'line_score'):7.1f}{avg_lang(r, 'char_acc'):7.1f}"
              f"{jd_s}{tr_s}{avg_lang(r, 'onset_ms'):9.0f}{r['score']:7.1f}"
              f"{len(r['errors']):5d}")
    print()
    print("=== 程序算法推荐（综合分 = 识别准×0.55 + 译文质×0.30 + 延迟分×0.15 − 异常×5）===")
    for i, r in enumerate(ranked[:args.top], 1):
        jd_s = f"{r['judge']:.1f}" if r.get("judge") is not None else "—"
        print(f"  {i}. {r['vad']} + {r['asr']} + {r['mt']}：综合 {r['score']:.1f}"
              f"（识别 {avg_lang(r, 'line_score'):.1f}｜译文 {jd_s}｜"
              f"体感延迟 {avg_lang(r, 'onset_ms'):.0f}ms"
              f"｜异常 {len(r['errors'])}）")
    print()
    print("=== 只看延迟最低（同等质量下最跟手）===")
    for i, r in enumerate(sorted(rows, key=lambda r: avg_lang(r, "onset_ms"))[:args.top], 1):
        print(f"  {i}. {r['vad']} + {r['asr']} + {r['mt']}："
              f"体感延迟 {avg_lang(r, 'onset_ms'):.0f}ms"
              f"（识别 {avg_lang(r, 'asr_ms'):.0f} + 翻译 {avg_lang(r, 'mt_ms'):.0f}"
              f" + 段长 {avg_lang(r, 'onset_ms') - avg_lang(r, 'e2e_ms'):.0f}）"
              f"｜识别 {avg_lang(r, 'line_score'):.1f}")

    best = ranked[0]
    print()
    print("=" * 108)
    print(f"=== 分语种明细（综合分第 1 名：{best['vad']} + {best['asr']} + {best['mt']}）===")
    print("=" * 108)
    print(f"{'语种':<6}{'参考句':>7}{'识别段':>7}{'识别准':>8}{'字符准':>8}"
          f"{'译文质':>8}{'术语':>7}{'识别':>8}{'翻译':>8}{'onset':>9}{'异常':>5}")
    for lang in langs:
        v = best["langs"].get(lang)
        if not v:
            continue
        jd = v.get("judge")
        jd_s = f"{jd:8.1f}" if jd is not None else "       -"
        tr_s = f"{v['term_on']*100:6.0f}%" if "term_on" in v else "      -"
        print(f"{lang:<6}{v['ref_n']:7d}{v['hyp_n']:7d}"
              f"{(v['line_score'] if v['line_score'] is not None else -1):8.1f}"
              f"{(v['char_acc'] if v['char_acc'] is not None else -1):8.1f}"
              f"{jd_s}{tr_s}{v['asr_ms']:8.0f}{v['mt_ms']:8.0f}{v['onset_ms']:9.0f}"
              f"{len(best['errors']):5d}")
    print()
    print("=== 样例（识别 → 译文；术语库开/关对照）===")
    for lang in langs:
        v = best["langs"].get(lang)
        if not v:
            continue
        for i in range(min(2, len(v["rec"]))):
            print(f"  [{lang}] 识别: {v['rec'][i][:70]}")
            print(f"       译文: {v['zh'][i][:70] if i < len(v['zh']) else ''}")
        if "term_on" in v:
            print(f"       术语保护率：不带 {v['term_off']*100:.0f}% → 带 {v['term_on']*100:.0f}%"
                  f"（共 {v['term_n']} 条术语）")
    if best.get("judge_notes"):
        print()
        print("  LLM 打分理由（**仅供参考**，同一个模型自评有自我偏好）：")
        for n in best["judge_notes"]:
            print(f"    {n}")
    print()
    print("[说明] 识别准确率 = 对每句参考文本找最相似识别段的相似度均值（惩罚切错句与识别错）；")
    print("       译文质量 = 提示词注入翻译模型自评（有自我偏好，仅参考）；")
    print("       推荐以程序算法为准（综合分），延迟是用户体感的 onset（段长+识别+翻译）。")


def main():
    ap = argparse.ArgumentParser(description="部署跑分：全组合实测 + 程序推荐最优搭配")
    ap.add_argument("--list", action="store_true", help="只看本机有哪些模型，不跑分")
    ap.add_argument("--langs", default="en,ja,ko,zh", help="参与跑分的语种切片")
    ap.add_argument("--max-seg-s", type=float, default=7.0, help="VAD 单句上限（秒）")
    ap.add_argument("--min-silence-ms", type=float, default=500.0, help="VAD 句尾静音（毫秒）")
    ap.add_argument("--max-mts-per-proc", type=int, default=1,
                    help="每个子进程最多同时驻留几个翻译模型（默认 1：whisper 1.6G + "
                         "hymt2 3.9G 已接近真实部署的驻留量，再多就 WDDM 溢出、慢 10 倍以上）")
    ap.add_argument("--judge", action="store_true", default=True,
                    help="用提示词让翻译模型给自己打分（默认开）")
    ap.add_argument("--no-judge", dest="judge", action="store_false")
    ap.add_argument("--judge-rows", type=int, default=2, help="每语种挑几行做 LLM 打分")
    ap.add_argument("--terms", action="store_true", default=True,
                    help="跑术语库开/关对照（默认开）")
    ap.add_argument("--no-terms", dest="terms", action="store_false")
    ap.add_argument("--term-rows", type=int, default=4, help="每语种挑几行做术语对照")
    ap.add_argument("--top", type=int, default=3, help="推荐榜显示前几名")
    ap.add_argument("--out", default="logs/bench_deploy.json")
    ap.add_argument("--force", action="store_true",
                    help="跳过 GPU 占用检测（默认：被占就等主程序退出，跑分要独占 GPU）")
    ap.add_argument("--asr", default="", help="（内部）子进程只跑这个识别后端")
    ap.add_argument("--vads", default="", help="（内部）子进程要跑的 VAD")
    ap.add_argument("--mts", default="", help="（内部）子进程要跑的翻译模型")
    ap.add_argument("--child", action="store_true", help="（内部）以子进程模式运行")
    args = ap.parse_args()
    if args.child:
        return run_child(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
