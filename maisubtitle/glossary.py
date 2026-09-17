"""术语库：词条/别名/模糊匹配/热更新/命中日志/CSV+JSON。"""
import csv
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rapidfuzz import fuzz, process as rf_process
from rapidfuzz.distance import Levenshtein


_LANG_ALIAS = {
    "english": "en", "en": "en", "英文": "en", "英语": "en",
    "japanese": "ja", "ja": "ja", "jp": "ja", "日文": "ja", "日语": "ja",
    "korean": "ko", "ko": "ko", "kr": "ko", "韩文": "ko", "韩语": "ko",
    "chinese": "zh", "zh": "zh", "cn": "zh", "中文": "zh",
    "common": "common", "通用": "common",
}


def normalize_lang(value: str) -> str:
    """把 'English'/'英文'/'JP' 之类写法统一成 en/ja/ko/zh/common。

    否则 CSV 里写 'English' 时，代码按 'en' 过滤永远匹配不上，术语库等于失效。
    """
    return _LANG_ALIAS.get((value or "").strip().lower(), (value or "").strip().lower())


@dataclass
class TermEntry:
    src_lang: str          # en / ja / ko / common（+ zh/cn 兼容）
    source: str
    aliases: list = field(default_factory=list)
    target_zh: str = ""
    force: str = "建议"     # 强制 / 建议
    priority: int = 2
    note: str = ""
    use: str = "both"      # both / asr / mt —— 识别与翻译的术语库分开时的开关
                           #   asr ：只用于识别（提示词偏置 + 近音纠错），不进翻译术语表
                           #   mt  ：只用于翻译（强制中文译名），不喂给识别引擎
                           #   both：两边都用（默认，兼容旧文件）

    def in_asr(self) -> bool:
        return self.use in ("both", "asr", "")

    def in_mt(self) -> bool:
        return self.use in ("both", "mt", "")


def _norm(text: str) -> str:
    t = text.lower()
    t = t.translate(str.maketrans(
        "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ０１２３４５６７８９",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"))
    return re.sub(r"[\s\W_]+", "", t, flags=re.UNICODE)


class Glossary:
    FUZZY_THRESHOLD = 85

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.terms: list[TermEntry] = []
        self.hit_log: list[dict] = []
        self.asr_fix_count = 0            # 识别侧纠错累计次数（供状态/日志显示）
        self._mtime = 0.0
        self.load()

    def load(self):
        if not self.path.exists():
            self.terms, self._mtime = [], 0.0
            return
        if self.path.suffix == ".json":
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.terms = [TermEntry(**d) for d in data]
        else:
            self.terms = []
            with open(self.path, newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    self.terms.append(TermEntry(
                        src_lang=row["src_lang"], source=row["source"],
                        aliases=[a for a in row.get("aliases", "").split("|") if a],
                        target_zh=row.get("target_zh", ""), force=row.get("force", "建议"),
                        priority=int(row.get("priority", 2) or 2),
                        note=row.get("note", ""),
                        use=(row.get("use") or "both").strip().lower()))
        for t in self.terms:                 # src_lang 归一化（English→en 等）
            t.src_lang = normalize_lang(t.src_lang)
        self._mtime = self.path.stat().st_mtime
        self._index = {}
        for t in self.terms:
            for variant in [t.source] + t.aliases:
                key = _norm(variant)
                if not key:
                    continue
                cur = self._index.get(key)
                if cur is None or t.priority < cur.priority:
                    if cur is not None:
                        self.hit_log.append({"event": "conflict", "key": key,
                                             "winner": t.source, "loser": cur.source})
                    self._index[key] = t

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.suffix == ".json":
            self.path.write_text(json.dumps([asdict(t) for t in self.terms],
                                            ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            with open(self.path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["src_lang", "source", "aliases", "target_zh",
                            "force", "priority", "note", "use"])
                for term in self.terms:
                    w.writerow([term.src_lang, term.source, "|".join(term.aliases),
                                term.target_zh, term.force, term.priority,
                                term.note, term.use])
        self._mtime = self.path.stat().st_mtime

    def check_reload(self) -> bool:
        m = self.path.stat().st_mtime
        if m != self._mtime:
            self.load()
            return True
        return False

    @staticmethod
    def _build_norm_map(text: str) -> list[int]:
        mapping = []
        for i, ch in enumerate(text):
            if ch.isspace() or re.match(r"[\W_]", ch, re.UNICODE):
                continue
            mapping.append(i)
        return mapping

    def match(self, text: str, src_lang: str | None = None,
              scope: str = "asr") -> list[dict]:
        """在文本里找术语命中。scope='asr' 只认识别侧词条，'mt' 只认翻译侧，
        'any' 不限（术语库编辑器用）。"""
        self.check_reload()
        norm = _norm(text)
        norm_to_raw = self._build_norm_map(text)
        def _ok(t):
            if not (src_lang in (None, "auto") or t.src_lang in (src_lang, "common")):
                return False
            if scope == "any":
                return True
            return t.in_asr() if scope == "asr" else t.in_mt()
        candidates = [t for t in self.terms if _ok(t)]
        hits, used = [], []
        for t in candidates:
            for variant in [t.source] + t.aliases:
                nv = _norm(variant)
                if not nv:
                    continue
                pos = norm.find(nv)
                if pos < 0 or pos + len(nv) - 1 >= len(norm_to_raw):
                    continue
                s, e = norm_to_raw[pos], norm_to_raw[pos + len(nv) - 1] + 1
                if not any(s < ue and e > us for us, ue in used):
                    hits.append({"start": s, "end": e, "source": text[s:e],
                                 "matched_variant": variant, "target_zh": t.target_zh,
                                 "term": t.source, "how": "exact"})
                    used.append((s, e))
                    break
        covered = set()
        for s, e in used:
            covered.update(range(s, e))
        words = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text)
                 if not any(x in covered for x in range(m.start(), m.end()))]
        for ws, we, w in words:
            nw = _norm(w)
            if len(nw) < 4:
                continue
            key, how = None, ""
            match = rf_process.extractOne(nw, list(self._index.keys()),
                                          scorer=fuzz.ratio,
                                          score_cutoff=self.FUZZY_THRESHOLD)
            if match:
                key, score, _ = match
                how = f"fuzzy:{score:.0f}"
            else:
                near = self._near_miss(nw)      # 近音/近形错词兜底（识别纠错用）
                if near:
                    key, how = near
            if key:
                t = self._index[key]
                if _ok(t):
                    hits.append({"start": ws, "end": we, "source": w,
                                 "matched_variant": key, "target_zh": t.target_zh,
                                 "term": t.source, "how": how})
        hits.sort(key=lambda h: h["start"])
        return hits

    def _near_miss(self, nw: str) -> tuple[str, str] | None:
        """近音/近形错词的兜底匹配（比 fuzz.ratio 更宽容，但有防误改护栏）。

        为什么需要它：`fuzz.ratio("rower","rover") = 80`，低于常规阈值 85，
        而"漏一个字母/错一个字母"恰恰是识别最常见的错法。直接降阈值会把
        "cover" 也改成 "rover"，所以改用**编辑距离 + 首字母护栏**：

          1. 首字母必须相同（听错极少连首音都换）；
          2. 长度差 ≤1，且两边都 ≥4 字符；
          3. 编辑距离：等长时 ≤1；长度差 1 时 ≤2 且较短一侧 ≥6。

        这样 "rower→rover"、"Genish→Genshin" 能纠正，
        "cover/under/water" 这类正常词不会被误改。
        """
        if len(nw) < 4:
            return None
        best: tuple[int, str] | None = None
        for key in self._index:
            if abs(len(key) - len(nw)) > 1 or min(len(key), len(nw)) < 4:
                continue
            if key[0] != nw[0]:
                continue
            d = Levenshtein.distance(nw, key)
            if len(key) == len(nw):
                ok = d <= 1
            else:
                ok = d <= 2 and min(len(key), len(nw)) >= 6
            if ok and (best is None or d < best[0]):
                best = (d, key)
        return (best[1], f"near:{best[0]}") if best else None

    def fix_asr(self, text: str, src_lang: str | None = None) -> tuple[str, list[dict]]:
        """识别纠错：把识别结果里"听错但很接近术语"的词掰回术语表写法。

        用途与 canonicalize 相同（都返回改正后的文本 + 命中列表），
        但这里的语义是"**识别侧**纠错"：字幕显示、上下文、导出 SRT 都用纠正后的文本，
        用户在屏幕上看到的就是这个词的正确写法（而不是等翻译时才纠正）。

        与 canonicalize 的唯一区别：**拉丁词必须整词命中**。否则别名 "rower"
        会把 "grower" 里的子串也替换掉（canonicalize 走的是子串匹配，历史原因）。
        """
        hits = [h for h in self.match(text, src_lang, scope="asr")
                if self._whole_word_ok(text, h)]
        if not hits:
            return text, []
        out, last = [], 0
        for h in hits:
            out.append(text[last:h["start"]])
            out.append(h["term"])
            last = h["end"]
            self.hit_log.append({"time": time.time(), "input": h["source"],
                                 "term": h["term"], "target": h["target_zh"],
                                 "how": "asr:" + str(h.get("how", ""))})
        out.append(text[last:])
        self.asr_fix_count += len(hits)
        return "".join(out), hits

    @staticmethod
    def _whole_word_ok(text: str, h: dict) -> bool:
        """拉丁词要求整词边界（前后都不是字母数字），CJK 不受此限。"""
        s, e = h["start"], h["end"]
        span = text[s:e]
        if not re.search(r"[A-Za-z]", span):
            return True
        before = text[s - 1] if s > 0 else ""
        after = text[e] if e < len(text) else ""
        return not (before.isalnum() or after.isalnum())

    def canonicalize(self, text: str, src_lang: str | None = None) -> tuple[str, list[dict]]:
        """变体/别名 → 规范源词（用于 ASR 偏置与 LLM 术语表路线）。

        拉丁词同样要求**整词命中**（`_whole_word_ok`）：否则别名 "rower"
        会把 "grower" 的中间替换掉，译文输入就被改坏了。
        """
        hits = [h for h in self.match(text, src_lang, scope="mt")
                if self._whole_word_ok(text, h)]
        out, last = [], 0
        for h in sorted(hits, key=lambda x: x["start"]):
            out.append(text[last:h["start"]])
            out.append(h["term"])
            last = h["end"]
            self.hit_log.append({"time": time.time(), "input": h["source"],
                                 "term": h["term"], "target": h["target_zh"],
                                 "how": h["how"]})
        out.append(text[last:])
        return "".join(out), hits

    def terms_for(self, src_lang: str) -> list[tuple[str, str]]:
        """翻译侧术语表（term → 中文译名）。use='asr' 的词条只用于识别，不进这里。"""
        return [(t.source, t.target_zh) for t in self.terms
                if t.src_lang in (src_lang, "common") and t.in_mt()]

    def asr_terms(self, src_lang: str) -> list[str]:
        """识别侧词表（喂给识别引擎做偏置用）——**只放规范写法，不放 aliases**。

        use='mt' 的词条只用于翻译，不喂给识别引擎——避免"没说的话也被听出来"。

        2026-09-16 实验（同一素材对照，探针脚本见开发仓）：
        把 aliases（"听错写法"）也塞进提示词时，whisper 会在含糊处**照抄这些错写法**：
            有 aliases: "We've got loads of fun in Genshin, Genisho"（术语 + 别名 两个都吐出来）
            无 aliases: "We've got loads of fun in Jinjo."（与音频一致）
        术语注入计数 2 → 0。别名的作用是**识别后**用精确匹配把听错的词掰回来
        （见 fix_asr），而不是反过来教模型听错。
        """
        out: list[str] = []
        for t in self.terms:
            if t.src_lang not in (src_lang, "common") or not t.in_asr():
                continue
            out.append(t.source)
        return out
