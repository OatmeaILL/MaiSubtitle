"""火山引擎「大模型流式语音识别」后端（WebSocket 双向流式）。

协议：SAUC v3（文档 docs.volcengine.com/docs/6561/1354869）。
与 HttpASR 的差别只是"怎么把音频交给服务端"：

  HTTP（HttpASR）  整段 WAV 一次性 POST（OpenAI / FunASR 风格），分句全交给服务端
  WS（本模块）     一段音频按 200ms 分包推流 → 推完发"负包" → 取该段最终文本

为什么值得多这条路：双向流式能开服务端 VAD 分句 + **二遍识别**（enable_nonstream），
标点、数字规范化（ITN）、准确率都更好，而且**不需要本地 GPU**。

协议要点（整数一律大端）：
  header 4B = [ver(4b)|header_size(4b)] [msg_type(4b)|flags(4b)]
              [serialization(4b)|compression(4b)] [保留(8b)]
  msg_type : 0b0001 full client request / 0b0010 audio only
             0b1001 server response      / 0b1111 error
  flags    : 0b0000 无序列号 / 0b0001 正序列号
             0b0010 最后一包(无序列号) / 0b0011 负序列号(最后一包)
  其后依次：[序列号 4B（仅 flags 带序号时）] + payload_size 4B + payload

鉴权（建连的 HTTP 头）：
  新版控制台  X-Api-Key
  旧版控制台  X-Api-App-Key + X-Api-Access-Key
  两者都要    X-Api-Resource-Id（如 volc.bigasr.sauc.duration）
              X-Api-Connect-Id / X-Api-Request-Id（UUID，便于对账）
  响应头里的 X-Tt-Logid 是排错线索，失败时一定要记下来（见 _logid）。

本机实测口径见 docs/dev/HANDOVER.md（识别后端一节）。
"""
import gzip
import json
import re
import ssl
import time
import uuid

import numpy as np

# ---- 协议常量 ----
_PROTOCOL_VERSION = 0b0001
_HEADER_SIZE = 0b0001
_FULL_CLIENT_REQUEST = 0b0001
_AUDIO_ONLY_REQUEST = 0b0010
_FULL_SERVER_RESPONSE = 0b1001
_SERVER_ERROR_RESPONSE = 0b1111
_FLAG_POS_SEQ = 0b0001
_FLAG_LAST_NO_SEQ = 0b0010
_FLAG_NEG_SEQ = 0b0011
_SER_JSON = 0b0001
_SER_NONE = 0b0000
_COMP_NONE = 0b0000
_COMP_GZIP = 0b0001

# 默认端点：双向流式**优化版**（文档推荐；有变化才回包，首字/尾字延迟更低）
DEFAULT_URL = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async"
DEFAULT_RESOURCE_ID = "volc.bigasr.sauc.duration"
# 单包音频 200ms（文档：双向流式 200ms 性能最优；100~200ms 之外会影响性能）
CHUNK_MS = 200
# 云端"成功"的 code：新版文档 0，经典口径 1000（两种都放行，别把成功当失败）
_OK_CODES = (None, 0, 1000)

# 语种 → 云端 language 键（**仅 bigmodel_nostream 支持 audio.language**，见 _request_json）
_LANG_KEYS = {"zh": "zh-CN", "en": "en-US", "ja": "ja-JP", "ko": "ko-KR"}


class VolcAsrError(RuntimeError):
    """云端识别失败（含服务端错误码 / logid，便于对数）。"""


def _header(msg_type: int, flags: int, ser: int = _SER_JSON,
            comp: int = _COMP_NONE) -> bytes:
    """4 字节协议头（按位打包，不涉及大端）。"""
    return bytes([(_PROTOCOL_VERSION << 4) | _HEADER_SIZE,
                  (msg_type << 4) | flags,
                  (ser << 4) | comp,
                  0])


def encode_client_request(payload: dict) -> bytes:
    """full client request：header + payload_size + JSON（不压缩）。"""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return (_header(_FULL_CLIENT_REQUEST, 0)
            + len(body).to_bytes(4, "big") + body)


def encode_audio_frame(pcm: bytes, seq: int, last: bool = False) -> bytes:
    """audio only request：header + 序列号 + payload_size + PCM（不压缩）。

    last=True 发的是"负包"（flags 0b0011 + 负序列号），表示音频到此为止 ——
    服务端收到后会把该段的最终结果回完。负包的 payload 允许为空。
    """
    flags = _FLAG_NEG_SEQ if last else _FLAG_POS_SEQ
    num = (-abs(int(seq)) if last else int(seq)).to_bytes(4, "big", signed=True)
    return _header(_AUDIO_ONLY_REQUEST, flags) + num + len(pcm).to_bytes(4, "big") + pcm


def plan_packets(pcm, chunk_samples: int) -> list[tuple[int, bytes, bool]]:
    """把整段 PCM 排成"要发的包"：[ (序列号, payload, 是否最后一包), … ]。

    **序列号从 2 开始**（不是 1）：full client request 隐式占了 1，服务端按它自己的
    计数器对齐，第一条音频包必须是 2。实锤（2026-09-17 用户真密钥实测，旧实现从 1 开始）：
      code=45000000 `...autoAssignedSequence (2) mismatch sequence in request (1)`
    （该修法待用户用「验证连接」复验 —— 写在这里以免下次又踩。）

    单独抽出来是为了能离线单测（发出去的字节序列错一位，只有在真连时才暴露，
    代价是用户白跑一次；这条实锤就是这么来的）。
    """
    step = max(1, int(chunk_samples))
    out: list[tuple[int, bytes, bool]] = []
    seq = 1                                   # 1 归 full client request
    for off in range(0, len(pcm), step):
        seq += 1
        out.append((seq, pcm[off:off + step].tobytes(), False))
    seq += 1
    out.append((seq, b"", True))              # 负包：告诉服务端音频到此为止
    return out


def _decompress(data: bytes, comp: int) -> bytes:
    """按 header 声明的压缩方式解 payload（没声明就直接用）。"""
    if comp == _COMP_GZIP and data:
        try:
            return gzip.decompress(data)
        except Exception:
            return data          # 不是 gzip：原样返回，交给 JSON 解析去报错
    return data


def decode_frame(data: bytes) -> dict:
    """解析服务端下发的一帧。返回 dict（纯函数，可离线单测）：

      {"type": "response"/"error"/"bad", "seq": int|None, "last": bool,
       "payload": dict|None, "code": int|None, "message": str}

    last=True 表示"识别结果已全部返回"（flags 0b0010/0b0011 或 payload.is_last_package）。
    """
    if not isinstance(data, (bytes, bytearray)) or len(data) < 4:
        return {"type": "bad", "message": f"帧太短（{len(data) if data else 0}B）",
                "seq": None, "last": False, "payload": None, "code": None}
    ver = data[0] >> 4
    hsize = (data[0] & 0x0F) * 4
    mtype = data[1] >> 4
    flags = data[1] & 0x0F
    comp = data[2] & 0x0F
    if ver != _PROTOCOL_VERSION or hsize < 4 or len(data) < hsize:
        return {"type": "bad", "message": f"头不合法（ver={ver} hsize={hsize}）",
                "seq": None, "last": False, "payload": None, "code": None}
    pos = hsize
    out = {"type": "unknown", "seq": None, "last": bool(flags & _FLAG_LAST_NO_SEQ),
           "payload": None, "code": None, "message": ""}
    if flags & _FLAG_POS_SEQ:                       # 带序列号（正 or 负）
        if len(data) < pos + 4:
            out["type"] = "bad"
            out["message"] = "缺序列号字段"
            return out
        out["seq"] = int.from_bytes(data[pos:pos + 4], "big", signed=True)
        pos += 4
    if mtype == _SERVER_ERROR_RESPONSE:
        if len(data) < pos + 8:
            out["type"] = "error"
            out["message"] = "错误帧缺少 code/size"
            return out
        out["code"] = int.from_bytes(data[pos:pos + 4], "big", signed=False)
        size = int.from_bytes(data[pos + 4:pos + 8], "big", signed=False)
        pos += 8
        body = _decompress(bytes(data[pos:pos + size]), comp)
        out["message"] = body.decode("utf-8", "replace")
        out["type"] = "error"
        return out
    if mtype == _FULL_SERVER_RESPONSE:
        if len(data) < pos + 4:
            out["type"] = "bad"
            out["message"] = "缺 payload_size"
            return out
        size = int.from_bytes(data[pos:pos + 4], "big", signed=False)
        pos += 4
        body = _decompress(bytes(data[pos:pos + size]), comp)
        try:
            out["payload"] = json.loads(body.decode("utf-8", "replace")) if body else {}
        except Exception as e:
            out["type"] = "bad"
            out["message"] = f"payload 不是 JSON（{str(e)[:60]}）"
            return out
        out["type"] = "response"
        js = out["payload"] if isinstance(out["payload"], dict) else {}
        if js.get("code") not in _OK_CODES:
            out["code"] = js.get("code")
            out["message"] = str(js.get("message") or "")
        if js.get("is_last_package") or js.get("is_last") or bool(js.get("last_package")):
            out["last"] = True
        return out
    return out


def pick_text(payload: dict | None) -> tuple[str, dict]:
    """从响应 JSON 取 (整段文本, 附加信息)。

    兼容两种形态（文档两份版本各写了一种）：
      A) {"code":0,"result":{"text":"…","utterances":[…]}}
      B) {"code":1000,"payload_msg":{"result":{"text":"…"}}}
    外加更老的 v2 口径（顶层 text / results[0].text）兜底 —— 取不到就返回空串，
    由调用方决定"空"怎么处理（静音段本来就可能是空的）。
    """
    info: dict = {}
    js = payload if isinstance(payload, dict) else {}
    if isinstance(js.get("payload_msg"), dict):
        js = js["payload_msg"]
    res = js.get("result")
    if isinstance(res, list):                       # 文档里 result 标成 list
        res = res[0] if res and isinstance(res[0], dict) else {}
    if not isinstance(res, dict):
        res = {}
    text = str(res.get("text") or js.get("text") or "")
    if not text:
        rs = js.get("results")
        if isinstance(rs, list) and rs and isinstance(rs[0], dict):
            text = str(rs[0].get("text") or "")
    utts = res.get("utterances")
    if isinstance(utts, list) and utts:
        info["utterances"] = len(utts)
        info["definite"] = sum(1 for u in utts
                               if isinstance(u, dict) and u.get("definite"))
    adds = res.get("additions") if isinstance(res.get("additions"), dict) else None
    if adds and adds.get("log_id"):
        info["logid"] = str(adds["log_id"])
    ai = js.get("audio_info") if isinstance(js.get("audio_info"), dict) else None
    if ai and ai.get("duration") is not None:
        info["audio_ms"] = ai["duration"]
    return text.strip(), info


def _words_in(prompt: str, limit: int = 50) -> list[str]:
    """术语提示词（"a, b, c"）→ 热词列表；云端限制 100 token，这里粗砍到 limit 条。"""
    out = []
    for w in re.split(r"[,、，;；]\s*", str(prompt or "")):
        w = w.strip()
        if w and w not in out:
            out.append(w)
        if len(out) >= limit:
            break
    return out


class VolcWsAsr:
    """火山流式识别（WebSocket 双向流式）——整段进、整段出，接口与 HttpASR 一致。

    为什么是"整段进"：本程序的管线是"VAD 切好一段 → 识别 → 翻译"，一次调用只面对
    一段音频。所以这里把整段按 200ms 分包推上去，推完发负包收最终文本 —— **不等
    实时节奏**（按 1:1 节奏推，4s 的段就要 4s 才推完，延迟等于段长，得不偿失）。
    真正的"边说边出字"要等第二阶段的常连接改造（把中间结果喂 on_partial）。
    """

    def __init__(self, url: str = DEFAULT_URL, api_key: str = "",
                 app_key: str = "", access_key: str = "",
                 resource_id: str = DEFAULT_RESOURCE_ID,
                 model_name: str = "bigmodel",
                 connect_timeout: float = 8.0, recv_timeout: float = 12.0,
                 max_wait: float = 20.0, chunk_ms: int = CHUNK_MS):
        self.url = str(url or DEFAULT_URL).strip()
        self.api_key = str(api_key or "").strip()
        self.app_key = str(app_key or "").strip()
        self.access_key = str(access_key or "").strip()
        self.resource_id = str(resource_id or "").strip() or DEFAULT_RESOURCE_ID
        self.model_name = str(model_name or "bigmodel").strip() or "bigmodel"
        self.connect_timeout = float(connect_timeout)
        self.recv_timeout = float(recv_timeout)
        # 整次调用的硬上限：必须小于 gpu_proc.DEFAULT_TIMEOUT(30s)，否则会把
        # 子进程拖到"识别超时"→ 白白重启一次（模型重载几十秒）
        self.max_wait = float(max_wait)
        self.chunk_samples = max(1600, int(16000 * int(chunk_ms) / 1000))
        self.uid = uuid.uuid4().hex
        self.load_s = 0.0
        # 二遍识别只在"双向流式优化版（bigmodel_async）"上支持（文档明说）
        self.nonstream = "bigmodel_async" in self.url
        self._last_logid = ""

    # ---------------- 建连 ----------------
    def _headers(self) -> list:
        """鉴权头。新版控制台只要 X-Api-Key；旧版要 App Key + Access Key。"""
        hs = [f"X-Api-Resource-Id: {self.resource_id}",
              f"X-Api-Connect-Id: {uuid.uuid4()}",
              f"X-Api-Request-Id: {uuid.uuid4()}"]
        if self.api_key:
            hs.append(f"X-Api-Key: {self.api_key}")
        if self.app_key:
            hs.append(f"X-Api-App-Key: {self.app_key}")
        if self.access_key:
            hs.append(f"X-Api-Access-Key: {self.access_key}")
        return hs

    def _connect(self, websocket):
        try:
            ws = websocket.create_connection(
                self.url, header=self._headers(), timeout=self.connect_timeout,
                sslopt={"cert_reqs": ssl.CERT_REQUIRED})
        except Exception as e:
            raise VolcAsrError(
                f"连不上 {self.url}（{str(e)[:120]}）—— 检查网络/地址、"
                f"X-Api-Key（或 App Key + Access Key）、X-Api-Resource-Id") from None
        try:                       # 握手响应头里的 logid 是唯一对账线索
            _hd = ws.getheaders() or {}
            # 头名大小写不保证（实测有的链路回小写）→ 大小写不敏感地找一遍
            self._last_logid = str(
                _hd.get("X-Tt-Logid") or _hd.get("x-tt-logid")
                or next((v for k, v in _hd.items()
                         if str(k).lower() == "x-tt-logid"), "") or "")
        except Exception:
            self._last_logid = ""
        return ws

    # ---------------- 请求体 ----------------
    def _request_json(self, language: str | None, prompt: str | None) -> dict:
        req = {
            "model_name": self.model_name,
            "enable_itn": True,          # 数字/金额规范化："一百二十三" → "123"
            "enable_punc": True,         # 标点（默认就开，写出来是为了明确口径）
            "enable_ddc": False,         # 语义顺滑会**删掉语气词**，字幕要保真 → 关
            "show_utterances": True,     # 分句信息（definite 判句、日志对账用）
            "result_type": "full",       # 每次回包给"整段到目前为止"的文本
        }
        if self.nonstream:
            # 二遍识别：流式先出字，分句用非流式模型重认一遍（更准，且 definite 只在
            # 非流式结果里给）。代价是每段末尾多一次重认 —— 实测值得（标点/准确率）。
            req["enable_nonstream"] = True
        audio = {"format": "pcm", "codec": "raw", "rate": 16000,
                 "bits": 16, "channel": 1}
        # audio.language 只有"流式输入模式（bigmodel_nostream）"支持；双向流式传了
        # 会被忽略（文档："仅流式输入模式支持此参数，二遍不支持"），所以只在
        # nostream 端点上传，免得写出一个"看着有用其实没用"的参数。
        if language and "nostream" in self.url:
            audio["language"] = _LANG_KEYS.get(str(language).lower(), str(language))
        body = {"user": {"uid": self.uid}, "audio": audio, "request": req}
        words = _words_in(prompt or "")
        if words:
            # 术语表 → 热词（corpus.context 需为 **JSON 字符串**）。热词是提示性的，
            # 不保证 100% 生效；超过 100 token 会被服务端截断，这里先砍到 50 条。
            ctx = {"hotwords": [{"word": w} for w in words]}
            req["corpus"] = {"context": json.dumps(ctx, ensure_ascii=False)}
        return body

    # ---------------- 主流程 ----------------
    def transcribe(self, audio: np.ndarray, language: str | None = None,
                   prompt: str | None = None, return_confidence: bool = False,
                   beam_size: int | None = None, without_timestamps: bool = True):
        """audio: int16 16k 单声道。返回 (text, meta)。

        失败一律抛 VolcAsrError（带 logid）：管线层会记 error.log 并继续下一段，
        不会静默吞掉（"字幕不动了却查不到原因"是最难查的一类问题）。
        """
        import websocket          # 延迟导入：不用这个后端就不给启动加依赖
        t0 = time.perf_counter()
        pcm = np.asarray(audio)
        if pcm.dtype != np.int16:
            pcm = pcm.astype(np.int16)
        total_s = len(pcm) / 16000.0
        deadline = t0 + self.max_wait
        text, info, n_resp, seq = "", {}, 0, 0
        ws = self._connect(websocket)
        try:
            ws.send_binary(encode_client_request(self._request_json(language, prompt)))
            for _seq, _payload, _last in plan_packets(pcm, self.chunk_samples):
                seq = _seq
                ws.send_binary(encode_audio_frame(_payload, _seq, last=_last))
            ws.settimeout(self.recv_timeout)
            while True:
                if time.perf_counter() > deadline:
                    raise VolcAsrError(
                        f"云端识别超时（{self.max_wait:.0f}s 内没收到最后一包，"
                        f"已收 {n_resp} 包，logid={self._logid}）")
                # 收包前的 socket 超时要收敛到剩余时间：recv 一旦开始就按 timeout
                # 整额阻塞，否则最坏 max_wait + recv_timeout（20+12=32s）> gpu_proc
                # 的 30s 看门狗 → 子进程被误判卡死重启（§七十六 同类教训）
                ws.settimeout(max(0.2, min(self.recv_timeout,
                                           deadline - time.perf_counter())))
                try:
                    data = ws.recv()
                except websocket.WebSocketTimeoutException:
                    if n_resp == 0:
                        raise VolcAsrError(
                            f"等待识别结果超时（{self.recv_timeout:.0f}s，"
                            f"logid={self._logid}）") from None
                    break                      # 已经收到结果、只是没等到最后一包：用已有的
                if not data:
                    break                      # 连接被服务端关闭
                if isinstance(data, str):
                    continue                   # 协议只有二进制帧；文本帧忽略
                fr = decode_frame(data)
                n_resp += 1
                if fr["type"] == "bad":
                    raise VolcAsrError(f"响应帧解析失败：{fr['message']}"
                                       f"（logid={self._logid}）")
                if fr["type"] == "error":
                    raise VolcAsrError(f"服务端返回错误 code={fr['code']} "
                                       f"{str(fr['message'])[:200]}（logid={self._logid}）")
                t, extra = pick_text(fr.get("payload"))
                if t:
                    text = t                  # result_type=full → 后到的就是全量
                info.update(extra)
                if fr.get("code") is not None:   # payload 里带非成功 code
                    raise VolcAsrError(f"识别失败 code={fr['code']} "
                                       f"{str(fr.get('message'))[:200]}"
                                       f"（logid={info.get('logid') or self._logid}）")
                if fr["last"]:
                    break
        except VolcAsrError:
            raise
        except Exception as e:
            # 收发阶段的原生异常（断链/超时/坏包）统一带上 logid 再抛：
            # error.log 里只有"某行 websocket 异常"是查不出原因的
            raise VolcAsrError(
                f"WebSocket 收发失败（{type(e).__name__}: {str(e)[:120]}）"
                f"（logid={self._logid}）") from None
        finally:
            try:
                ws.close()
            except Exception:
                pass
        if n_resp == 0:
            raise VolcAsrError(f"服务端一包都没回（logid={self._logid}）—— "
                               f"多为鉴权/资源 ID 不对或该服务未开通")
        dt = time.perf_counter() - t0
        meta = {"latency_s": round(dt, 3), "audio_s": round(total_s, 2),
                "rtf": round(dt / max(total_s, 1e-6), 3),
                "ws": "volc", "packets": seq, "responses": n_resp}
        if info.get("logid") or self._logid:
            meta["logid"] = info.get("logid") or self._logid
        if info.get("definite") is not None:
            meta["definite"] = info["definite"]
        if not text:
            meta["empty"] = True
        return text.strip(), meta

    @property
    def _logid(self) -> str:
        return self._last_logid or "-"

    def detect_lang(self, audio: np.ndarray):
        """云端这个接口不做语种检测 → 返回空，由"固定源语言"接管（同 HttpASR）。"""
        return "", 0.0


def _probe_audio(seconds: float = 3.0):
    """给"验证连接"准备一小段音频：有本机测试素材就发**真语音**（顺带验证识别与
    标点），没有就发静音（只验证鉴权/协议/联通）。返回 (int16 数组, 说明)。

    素材只用本机 testdata/（**不进 git**，所以别的机器上自然走静音分支）。
    """
    try:
        import wave

        from .config import TESTDATA_DIR
        p = TESTDATA_DIR / "_clip_zh.wav"
        if p.exists():
            with wave.open(str(p), "rb") as w:
                if w.getframerate() == 16000 and w.getnchannels() == 1:
                    n = min(int(16000 * seconds), w.getnframes())
                    a = np.frombuffer(w.readframes(n), dtype=np.int16)
                    if a.size:
                        return a, "语音素材（testdata/_clip_zh.wav）"
    except Exception:
        pass                                  # 素材读不了 → 退回静音，不影响验证
    return np.zeros(int(16000 * 1.2), dtype=np.int16), "静音"


def probe(url: str = DEFAULT_URL, api_key: str = "", app_key: str = "",
          access_key: str = "", resource_id: str = "", model_name: str = "bigmodel",
          seconds: float = 3.0) -> dict:
    """验证"地址 + 密钥 + 资源 ID"能不能用：真连一次云端，发一小段音频。

    设置界面「验证连接」按钮用它。**不抛异常**（失败也返回结果），返回：
      {"ok": bool, "msg": str, "logid": str, "ms": int, "text": str, "audio": str}
    失败信息里带 logid —— 火山控制台查问题只认这个。
    """
    if not (str(api_key or "").strip()
            or (str(app_key or "").strip() and str(access_key or "").strip())):
        return {"ok": False, "msg": "没填密钥：新版控制台填「API Key」；"
                                    "旧版控制台填「App Key + Access Key」",
                "logid": "", "ms": 0, "text": "", "audio": "-"}
    audio, kind = _probe_audio(seconds)
    asr = VolcWsAsr(url or DEFAULT_URL, api_key=api_key, app_key=app_key,
                    access_key=access_key, resource_id=resource_id,
                    model_name=model_name)
    t0 = time.perf_counter()
    try:
        text, meta = asr.transcribe(audio)
    except Exception as e:
        return {"ok": False, "msg": str(e)[:300], "logid": asr._logid,
                "ms": int((time.perf_counter() - t0) * 1000), "text": "", "audio": kind}
    ms = int((time.perf_counter() - t0) * 1000)
    if kind.startswith("静音") and not text:
        msg = f"鉴权与协议通过（发了 1.2s 静音，所以没有文字），往返 {ms} ms"
    else:
        msg = f"连接成功，往返 {ms} ms"
    return {"ok": True, "msg": msg, "logid": meta.get("logid", "") or asr._logid,
            "ms": ms, "text": text, "audio": kind}