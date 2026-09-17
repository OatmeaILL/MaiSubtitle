# -*- coding: utf-8 -*-
"""字幕顺序上屏队列：新句必须先"当上当前行、停留够久"，再让位给下一句。

为什么需要它
------------
管线处理是**突发**的：GPU 稍慢或一段积压后，会一次吐出 2~3 句（实测日志
`20:29:14/15` 三句同秒、`20:29:23` 两句同秒）。如果一次性丢给悬浮窗，只有
最后一句会"当过当前行"，前面几句直接掉进历史行一闪而过 —— 观众根本没法看。

规则（用户要求：按顺序、中间必须有停顿）
----------------------------------------
1. **保序**：先产出的先上屏，绝不插队；
2. **当前行更新免费**：同一句的流式译文/部分识别更新立即生效，不受停留限制；
3. **换行要等"定稿 + 停留"**：让位前，当前行既要**内容不再变**（译文流完/回退长回来），
   又要在这个状态下停留够 `dwell_s`；流得太久时由 `max_hold_s` 兜底（默认 dwell 的 1.5 倍），
   保证新句不被积压；
4. **旧句只回填**：已经上过屏的句子再来更新（迟到的定稿/更正），直接回填它自己那行，
   **不重新排队**（否则旧句会挤掉当前行、自己再演一遍）；
5. 队列有上限，积压太多时丢**最旧**的（那些本来就没机会被看清）。
"""
from __future__ import annotations


class DisplayQueue:
    def __init__(self, dwell_ms: float = 900.0, max_wait: int = 4,
                 settle_grace_ms: float | None = None):
        self.dwell_s = max(0.0, float(dwell_ms) / 1000.0)
        self.max_wait = max(1, int(max_wait))
        # 让位的条件不是"上屏够久"，而是"**内容定稿**后再停留够久"。
        # 起因（用户反馈）：续接重译/回退会让同一句的译文"缩回去再重新长"，
        # 而旧口径从上屏那一刻计时 —— 完整译文还没在当前行露过面，就够钟让位、
        # 被下一句挤进历史行（观感是"我还没看完就没了"，而且历史行里留的是残句）。
        # 但流式可以持续很久，不能无限等 → settle_grace_s 是"最多再多等多久"的硬上限，
        # 保证新句不被积压、字幕不落后于话音。默认 = 半个 dwell。
        self.settle_grace_s = (self.dwell_s * 0.5 if settle_grace_ms is None
                               else max(0.0, float(settle_grace_ms) / 1000.0))
        self.max_hold_s = self.dwell_s + self.settle_grace_s
        self.shown_cid = None          # 当前行（已上屏）的 cid
        self.shown_t = 0.0             # 它是什么时候上屏的
        self.changed_t = 0.0           # 它最后一次"内容变化"的时刻（稳定计时起点）
        self.shown_key = None          # 当前行的内容指纹（判定"还在变"）
        self.done: set = set()         # 已经上过屏的 cid（见 push：迟到更新只回填）
        # 量级说明：cid 是短字符串/int，连看几小时也就几千条（几十 KB），无需限长；
        # reset()（暂停/清屏）会整体清空。
        self.wait: list[tuple] = []    # 待上屏（保序）：[(cid, src, dst, lang)]
        self.dropped = 0               # 统计：因积压被丢掉的句数

    # ---------------- 入队 ----------------
    def push(self, cid, src: str, dst, lang: str, now: float):
        """投递一条字幕。

        返回 ("update", item) 表示"这条要立即上屏"（当前行更新、或已上过屏的旧句回填）；
        返回 None 表示已排队，等 ready() 在合适的时机交出。
        """
        item = (cid, src, dst, lang)
        if cid == self.shown_cid:
            key = (src, dst, lang)
            if key != self.shown_key:      # 内容真变了才算"还在变"（同值重复推不算）
                self.shown_key = key
                self.changed_t = now
            return ("update", item)                 # 同一句：立即生效
        if cid in self.done:
            # 已经上过屏的**旧句**的迟到更新（定稿/更正/流式尾巴）：直接回填它自己那行，
            # 绝不能重新排队 —— 那会让它把当前行挤掉、自己再"演"一遍（旧句回放）。
            return ("update", item)
        for i, w in enumerate(self.wait):
            if w[0] == cid:
                self.wait[i] = item                  # 同一句在队列里：覆盖（保序）
                return None
        self.wait.append(item)
        while len(self.wait) > self.max_wait:        # 积压：丢最旧
            self.wait.pop(0)
            self.dropped += 1
        return None

    # ---------------- 出队 ----------------
    def ready(self, now: float):
        """该上屏的下一句（没有就 None）。**每次调用最多交出一条**，保证停顿。"""
        if not self.wait:
            return None
        if self.shown_cid is not None:
            if (now - self.changed_t) < self.dwell_s and (now - self.shown_t) < self.max_hold_s:
                return None        # 内容还在变（译文还在流）→ 等它定稿再让位
        item = self.wait.pop(0)
        self.shown_cid, self.shown_t = item[0], now
        self.changed_t = now                        # 新行的"稳定"从这一刻起算
        self.shown_key = (item[1], item[2], item[3])
        self.done.add(item[0])
        return item

    def pending(self) -> int:
        return len(self.wait)

    def remove(self, cid):
        """撤回一条（管线把它判为幻觉/音乐被丢弃了）。

        当前行被撤回时把 shown_cid 清掉 —— 下一句可以立刻上屏，不用再等停留；
        否则"当前行"是一个已经不存在的内容，下一句得白等 dwell 才轮到。
        """
        if cid == self.shown_cid:
            self.shown_cid = None
            self.shown_t = 0.0
            self.changed_t = 0.0
            self.shown_key = None
        self.wait = [w for w in self.wait if w[0] != cid]

    def reset(self):
        """暂停/清屏后调用：当前行与队列都清空。"""
        self.shown_cid = None
        self.shown_t = 0.0
        self.changed_t = 0.0
        self.shown_key = None
        self.done.clear()
        self.wait.clear()
