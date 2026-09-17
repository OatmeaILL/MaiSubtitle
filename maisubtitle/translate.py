"""翻译：Qwen 系引擎（CT2 / Hy-MT2）+ 术语保护 + 语言与文本工具。"""
import json
import re
import time
from pathlib import Path

from .config import MODELS_DIR


def to_simplified(text: str) -> str:
    """中文显示统一简体（whisper 中文转写常带繁体；译文防御性转换）。"""
    if not text:
        return text
    from zhconv import convert as _convert
    return _convert(text, "zh-cn")


class TermProtector:
    """层3 术语保护：唯一三位数占位符（避开源文数字）→ 翻译 → 还原。"""

    def __init__(self):
        self.masked: dict[str, str] = {}
        self._n = 0

    def mask(self, text: str, hits: list[dict]) -> str:
        self.masked = {}
        self._n = 0
        used = set(re.findall(r"\d{3}", text))
        out, last = [], 0
        for h in hits:
            out.append(text[last:h["start"]])
            self._n += 1
            ph = str(900 + self._n)
            while ph in used:
                self._n += 1
                ph = str(900 + self._n)
            used.add(ph)
            self.masked[ph] = h["target_zh"]
            out.append(ph)
            last = h["end"]
        out.append(text[last:])
        return "".join(out)

    def unmask(self, translated: str) -> tuple[str, list[str]]:
        missing, out = [], translated
        for ph, zh in self.masked.items():
            if ph in out:
                out = out.replace(ph, zh)
            else:
                missing.append(ph)
        out = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", out)
        return out, missing

    def post_fix(self, translated: str, target_zh: str,
                 ratio_threshold: float = 68.0) -> tuple[str, bool]:
        """译文侧模糊修正：找最接近目标词的窗口替换。"""
        from rapidfuzz import fuzz
        best_ratio, best = 0.0, None
        n = len(target_zh)
        for size in (n - 1, n, n + 1):
            if size <= 0:
                continue
            for i in range(0, len(translated) - size + 1):
                r = fuzz.ratio(translated[i: i + size], target_zh)
                if r > best_ratio:
                    best_ratio, best = r, (i, i + size)
        if best and best_ratio >= ratio_threshold:
            i, j = best
            return translated[:i] + target_zh + translated[j:], True
        return translated, False


_FORCE_ZH_SYS = ("你只会输出简体中文。把用户给的外语对白翻译成自然的简体中文口语，"
                 "严禁照抄原文、严禁输出英文、不要解释、不要加引号。")


def force_zh_messages(text: str, terms: list[tuple[str, str]] | None = None) -> list[dict]:
    """兜底重试用的提示词：不带任何语种假设，只强调"必须输出中文"。"""
    sysmsg = _FORCE_ZH_SYS
    if terms:
        sysmsg += ("\n术语表（句中出现的词语必须使用给定中文译名）：\n"
                   + "\n".join(f"{s} => {z}" for s, z in terms))
    return [{"role": "system", "content": sysmsg},
            {"role": "user", "content": text}]


def force_zh(mt, text: str, terms: list[tuple[str, str]] | None = None) -> str:
    """用"强制中文"提示词再翻一次——应对模型"复读原文"（译文=英文）的情况。

    只在主路径与"去上下文重试"都没出中文时调用；不支持对话模板的引擎返回空串。
    """
    if not hasattr(mt, "chat"):
        return ""
    try:
        return mt.chat(force_zh_messages(text, terms))
    except Exception:
        return ""


def terms_in(text: str, terms: list[tuple[str, str]] | None) -> list[tuple[str, str]]:
    """句中真正出现的术语（用于"必须在译文里用给定译名"的校验）。"""
    low = (text or "").lower()
    return [(s, z) for s, z in (terms or []) if s and s.lower() in low]


def substitute_terms(text: str, terms: list[tuple[str, str]] | None) -> str:
    """把源文里的术语直接替换成目标中文（术语兜底重译用）。"""
    out = text
    for s, z in (terms or []):
        if s:
            out = re.sub(re.escape(s), z, out, flags=re.I)
    return out


def _qwen_messages(text: str, src_lang: str, context: list[str] | None,
                   terms: list[tuple[str, str]] | None = None) -> list[dict]:
    """Qwen 翻译的对话模板。

    QwenCT2（CTranslate2）与 Hy-MT2 共用这一份，保证换实现不改提示词、译文可比。

    术语这块踩过坑（2026-09-14 实测 Qwen3-1.7B）：**只在 system 里给术语表时，
    小模型经常无视**（rover 照样翻成"探测车"）。把"本句出现的术语"再放到 user
    消息里点名强调后即稳定遵守；配合调用方的"缺术语→替换源文重译"兜底更稳。
    """
    sysmsg = ("你是资深影视字幕译者，把对白翻译成自然的简体中文口语。"
              "要求：无论原文是什么语言，译文永远用简体中文；"
              "贴合语境像人话，不逐词直译；代词和主语参照前文，"
              "冗余代词可省略；只输出中文译文本身。")
    if terms:
        tbl = "\n".join(f"{s} => {z}" for s, z in terms)
        sysmsg += ("\n术语表：句中出现的下列词语必须严格使用给定中文译名"
                   "（无论读音像什么都不得改写或意译）：\n" + tbl)
    msgs = [{"role": "system", "content": sysmsg}]
    for c in (context or [])[-5:]:
        if c and re.search(r"[\u4e00-\u9fff]", c):
            msgs.append({"role": "assistant", "content": c})
    lang_name = {"ja": "日语", "ko": "韩语", "en": "英语"}.get(src_lang, "")
    # 句内术语点名强调（小模型对 system 里的术语表不敏感，这条是关键）
    hit = terms_in(text, terms)
    remind = ""
    if hit:
        remind = "（" + "；".join(f"「{s}」必须译成「{z}」" for s, z in hit) + "）"
    msgs.append({"role": "user",
                 "content": f"把下面这句{lang_name}对白翻译成简体中文{remind}：\n{text}"})
    return msgs


class _ChatMLTokenizer:
    """轻量聊天 tokenizer：tokenizers 直读 + jinja2 渲染对话模板（不 import transformers）。

    transformers 首次 import 实测约 **6 秒**（启动延迟最大单项，
    且期间会饿住 Qt 主线程造成悬浮窗"卡住"）。而 QwenCT2 只需要它的三件事：
    渲染 chat_template / encode / decode。改用：
      - `tokenizers` 库（faster-whisper 已依赖，import≈0）读 tokenizer.json；
      - `jinja2`（≈0.03s，也是既有依赖）渲染 tokenizer_config.json 里的 chat_template。
    输出已与 AutoTokenizer 做逐字节/逐 id 对比验证（对照脚本见开发仓）。
    """

    def __init__(self, tok_dir: Path):
        import jinja2
        from tokenizers import Tokenizer
        cfg = json.loads((tok_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
        self._tk = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
        self._tpl = jinja2.Environment().from_string(cfg["chat_template"])
        eos = cfg.get("eos_token") or "<|im_end|>"
        if isinstance(eos, dict):            # 老格式 {"content": "..."}
            eos = eos.get("content", "<|im_end|>")
        self.eos_token_id = self._tk.token_to_id(eos)

    def apply_chat_template(self, messages: list[dict], tokenize: bool = False,
                            add_generation_prompt: bool = True, **kw) -> str:
        ctx = {"messages": messages,
               "add_generation_prompt": add_generation_prompt,
               "bos_token": "", "eos_token": ""}
        ctx.update(kw)
        return self._tpl.render(**ctx)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._tk.encode(text, add_special_tokens=add_special_tokens).ids

    def convert_ids_to_tokens(self, ids):
        return [self._tk.id_to_token(int(i)) for i in ids]

    def convert_tokens_to_ids(self, tokens):
        return [self._tk.token_to_id(t) for t in tokens]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return self._tk.decode(list(ids), skip_special_tokens=skip_special_tokens)


def _load_chat_tokenizer(tok_dir: Path):
    """优先轻量实现（≈0.1s）；缺 tokenizer.json/对话模板时回落 transformers（≈6s，保底）。"""
    try:
        if ((tok_dir / "tokenizer.json").exists()
                and (tok_dir / "tokenizer_config.json").exists()):
            cfg = json.loads((tok_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
            if cfg.get("chat_template"):
                return _ChatMLTokenizer(tok_dir)
    except Exception:
        pass
    from transformers import AutoTokenizer      # 兜底：首次 import 约 6s
    return AutoTokenizer.from_pretrained(str(tok_dir))


class QwenCT2:
    """Qwen2.5-1.5B-Instruct 的 CTranslate2 版（推荐，默认路线）。

    权重与 PyTorch 版相同，只是解码换成 CT2 的 C++ 实现：一次 generate_batch
    在 C++ 里连续发 kernel，不再由 Python 逐 token 调度，因此绕开了
    Windows(WDDM) 下「约 500 次 kernel 启动/token」的开销。

    转换（一次性，需几分钟）：
        ct2-transformers-converter --model models/Qwen2.5-1.5B-Instruct-hf \
            --output_dir models/Qwen2.5-1.5B-Instruct-ct2 \
            --quantization int8_float16 --copy_files tokenizer.json
      注意：**不要 --copy_files config.json**，CT2 自己要写 config.json，会撞名报错。
    """

    def __init__(self, device: str = "cuda", compute_type: str = "int8_float16",
                 model_dir: str = "Qwen2.5-1.5B-Instruct-ct2",
                 tok_dir: str = "Qwen2.5-1.5B-Instruct-ct2",
                 template_kwargs: dict | None = None):
        import ctranslate2
        t0 = time.perf_counter()
        ct2_dir = MODELS_DIR / model_dir
        # 优先用 CT2 目录里自带的 tokenizer（含 tokenizer_config.json 的对话模板）；
        # 早期只拷了 tokenizer.json 的转换产物则回落到 tok_dir（默认同 CT2 目录）。
        # 用轻量 tokenizer（tokenizers+jinja2）替代 transformers：省约 6s 启动导入，
        # 也避免导入期间饿住 Qt 主线程（悬浮窗"点一下才刷新"的主因）。
        tok_src = ct2_dir if (ct2_dir / "tokenizer_config.json").exists() \
            else (MODELS_DIR / tok_dir)
        self.tok = _load_chat_tokenizer(tok_src)
        self.gen = ctranslate2.Generator(str(ct2_dir),
                                         device=device, compute_type=compute_type)
        self.eos = self.tok.eos_token_id
        # Qwen3 系列默认开思考模式（先输出 <think>…</think>），字幕场景必须关掉；
        # 对没有该变量的模板（如 Qwen2.5）传了也无害。
        self.template_kwargs = template_kwargs or {"enable_thinking": False}
        self.load_s = time.perf_counter() - t0
        self.terms: list[tuple[str, str]] = []

    def set_terms(self, terms: list[tuple[str, str]]):
        self.terms = list(terms)

    def chat(self, messages: list[dict], max_tokens: int = 160) -> str:
        prompt = self.tok.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True,
                                              **self.template_kwargs)
        ids = self.tok.encode(prompt, add_special_tokens=False)
        results = self.gen.generate_batch(
            [self.tok.convert_ids_to_tokens(ids)],
            max_length=max_tokens, include_prompt_in_result=False,
            beam_size=1, end_token=[self.eos])
        out_ids = self.tok.convert_tokens_to_ids(list(results[0].sequences[0]))
        return self.tok.decode(out_ids, skip_special_tokens=True).strip()

    def translate(self, text: str, src_lang: str,
                  context: list[str] | None = None) -> tuple[str, dict]:
        msgs = _qwen_messages(text, src_lang, context, self.terms)
        t0 = time.perf_counter()
        out = self.chat(msgs)
        return out, {"latency_s": round(time.perf_counter() - t0, 3),
                     "engine": "qwen-ct2"}

    def translate_stream(self, text: str, src_lang: str,
                         context: list[str] | None = None,
                         max_tokens: int = 160):
        """流式翻译：逐 token 产出**累积**译文（生成完最后一个 token 迭代结束）。

        只在 CT2 引擎上可用（generate_tokens 逐 token 回调，强制 beam=1，
        与我们的贪心解码一致，结果与非流式相同）。解码半个多字节字符时
        decode 会暂时产生 \ufffd，此时跳过本次产出，避免 UI 闪现乱码。
        """
        msgs = _qwen_messages(text, src_lang, context, self.terms)
        prompt = self.tok.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True,
                                              **self.template_kwargs)
        ids = self.tok.encode(prompt, add_special_tokens=False)
        tokens = self.tok.convert_ids_to_tokens(ids)
        got: list[int] = []
        for step in self.gen.generate_tokens([tokens], max_length=max_tokens,
                                             end_token=[self.eos]):
            token = getattr(step, "token", None)
            if token is None:
                continue
            got.append(self.tok.convert_tokens_to_ids([token])[0])
            partial = self.tok.decode(got, skip_special_tokens=True).strip()
            if partial.endswith("\ufffd"):
                continue
            yield partial


class HyMT2:
    """腾讯混元 Hy-MT2-1.8B：专用多语言翻译模型（33 语种，支持术语/上下文指令）。

    优点：专为翻译训练，术语一致性与上下文衔接是它的强项（官方还赞助了
    WMT26 视频字幕翻译任务），小语料场景的译文比通用 instruct 模型更专业。
    代价：架构 HunYuanDenseV1ForCausalLM **不被 CTranslate2 支持**（4.8.2 没有
    Hunyuan loader），只能走 transformers/PyTorch —— 逐 token 的 kernel 启动
    开销避不开，速度比 QwenCT2 慢。engine="hymt2" 选择。

    官方推荐采样参数是 temperature=0.7/top_p=0.6，字幕场景改用贪心解码
    （确定性、更快），配合 repetition_penalty=1.05 防复读。
    """

    def __init__(self, device: str = "cuda", model_dir: str = "Hy-MT2-1.8B",
                 dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        path = str(MODELS_DIR / model_dir)
        t0 = time.perf_counter()
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, dtype=getattr(torch, dtype), device_map=device,
            trust_remote_code=True)
        self.model.eval()
        self.load_s = time.perf_counter() - t0
        self.terms: list[tuple[str, str]] = []

    def set_terms(self, terms: list[tuple[str, str]]):
        self.terms = list(terms)

    def chat(self, messages: list[dict], max_tokens: int = 160) -> str:
        prompt = self.tok.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens,
                                      do_sample=False, repetition_penalty=1.05)
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip()

    def translate(self, text: str, src_lang: str,
                  context: list[str] | None = None) -> tuple[str, dict]:
        parts = []
        if self.terms:
            parts.append("术语对照（译文中出现的词语必须使用给定译名）:\n"
                         + "\n".join(f"{s} => {z}" for s, z in self.terms))
        if context:
            parts.append("前文译文（代词与称谓要衔接）:\n"
                         + "\n".join(f"- {c}" for c in list(context)[-5:]))
        parts.append("将以下文本翻译成简体中文，注意只需要输出翻译后的结果，"
                     "不要额外解释:\n\n" + text)
        msgs = [{"role": "user", "content": "\n\n".join(parts)}]
        t0 = time.perf_counter()
        out = self.chat(msgs)
        return out, {"latency_s": round(time.perf_counter() - t0, 3),
                     "engine": "hymt2"}


class NullMT:
    """所有翻译候选都不可用时的占位：只出原文（dst 留空），不假装翻译。

    以前这一层由 NLLB 兜底；NLLB 引擎已按用户要求整体移除（2026-09-17），
    再没有可用的兜底模型 —— 宁可留空，也绝不把英文原文当译文显示。
    """
    engine = "none"
    load_s = 0.0

    def set_terms(self, terms):        # 与真实翻译器同接口，管线无须改动
        pass

    def translate(self, text: str, src_lang: str,
                  context: list[str] | None = None) -> tuple[str, dict]:
        return "", {"engine": "none"}

    def chat(self, messages: list[dict]) -> str:
        return ""


def translator_candidates(engine: str = "qwen") -> list:
    """按 engine 给出**候选构造器**（按优先级）；全部不可用时由调用方退 NullMT。

    "hymt2" → Hy-MT2 专用翻译模型，缺权重时退回 QwenCT2
    "qwen"  → QwenCT2（最快，默认）
    "qwen3" → Qwen3-1.7B-CT2（新一代，质量略优、稍慢），失败退回 QwenCT2

    实时管线（live.py，要逐候选记日志）与离线/子进程都取这一份，
    免得两处各写一张表、改一处忘另一处（曾经就是逐字重复的两份）。
    """
    return {
        "hymt2": [HyMT2, QwenCT2],
        "qwen":  [QwenCT2],
        "qwen3": [lambda: QwenCT2(model_dir="Qwen3-1.7B-ct2",
                                  tok_dir="Qwen3-1.7B-ct2"), QwenCT2],
    }.get(engine, [QwenCT2])


def make_translator(engine: str = "qwen", **kw):
    """按 engine 取一个翻译器（离线管线/基准脚本用；live.py 有带日志的版本）。

    全部不可用时抛最后一个异常，由调用方决定退 NullMT（只出原文）。
    """
    last = None
    for mk in translator_candidates(engine):
        try:
            return mk(**kw)
        except Exception as e:      # 记下最后一个异常继续尝试
            last = e
    raise last
