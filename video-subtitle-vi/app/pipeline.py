"""
Pipeline: video -> audio -> ASR (Whisper) -> phân đoạn phụ đề -> dịch tiếng Việt (LLM) -> SRT/VTT -> render video.
"""
import json
import logging
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import requests

log = logging.getLogger("pipeline")

# ---------- Cấu hình ----------
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")  # cuda | cpu
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "float16" if WHISPER_DEVICE == "cuda" else "int8")

TRANSLATOR = os.getenv("TRANSLATOR", "claude")  # claude | ollama | openai
# API chuẩn OpenAI: DeepSeek (mặc định), OpenAI, Qwen DashScope, OpenRouter...
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "deepseek-chat")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-5-5")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
TRANSLATE_BATCH = int(os.getenv("TRANSLATE_BATCH", "30"))

SUB_FONT = os.getenv("SUB_FONT", "Noto Sans")
MAX_LINE_CHARS = 42      # ký tự / dòng phụ đề tiếng Việt
MAX_SEG_SECONDS = 6.5    # thời lượng tối đa 1 đoạn phụ đề
MIN_SEG_SECONDS = 1.0
MAX_GAP_SECONDS = 0.8    # khoảng lặng đủ để ngắt đoạn

_model = None
_model_lock = threading.Lock()


def get_model():
    """Nạp Whisper một lần, dùng lại cho mọi job."""
    global _model
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel
            log.info("Loading Whisper %s on %s (%s)", WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE)
            _model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
        return _model


def _run(cmd, cwd=None):
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Lệnh lỗi: {' '.join(cmd[:3])}...\n{proc.stderr[-1500:]}")
    return proc.stdout


# ---------- 1. Audio ----------
def extract_audio(video: Path, out: Path) -> Path:
    _run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out)])
    return out


def probe_duration(path: Path) -> float:
    out = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=nw=1:nk=1", str(path)])
    try:
        return float(out.strip())
    except ValueError:
        return 0.0


# ---------- 2. Nhận dạng giọng nói ----------
def transcribe(audio: str, language=None, duration=0.0, on_progress=None):
    """Trả về danh sách từ có timestamp + ngôn ngữ phát hiện được."""
    model = get_model()
    segments, info = model.transcribe(
        audio,
        language=language,          # None = tự nhận diện
        # ponytail: tắt VAD vì nó cắt mất lời nói có nhạc nền (mất cả 20s trong video test).
        # Video dài nhiều đoạn nhạc thuần mà bị "bịa chữ" thì bật lại kèm vad_parameters=dict(onset=0.2).
        vad_filter=False,
        condition_on_previous_text=False,  # giảm lặp câu / bịa chữ khi không có VAD
        word_timestamps=True,       # cần để ngắt phụ đề chính xác
        beam_size=5,
    )
    words = []
    for seg in segments:  # generator: xử lý dần trong lúc nhận dạng
        if seg.words:
            words += [{"start": w.start, "end": w.end, "text": w.word, "seg_start": i == 0}
                      for i, w in enumerate(seg.words)]
        else:
            words.append({"start": seg.start, "end": seg.end, "text": seg.text, "seg_start": True})
        if on_progress and duration:
            on_progress(min(seg.end / duration, 1.0))
    return words, info.language, float(info.language_probability)


# ---------- 3. Phân đoạn phụ đề ----------
_SENTENCE_END = re.compile(r"[.!?。！？…]$")
_SOFT_BREAK = re.compile(r"[,，、;；:：]$")


def _split_point(cur, cjk, min_chars):
    """Chỗ ngắt tự nhiên muộn nhất: sau dấu phẩy, hoặc chỗ Whisper chèn khoảng trắng (CJK = ngừng nói).
    Không có thì ngắt ở khoảng nghỉ dài nhất giữa 2 từ, tránh cắt đôi một từ."""
    best, best_gap = len(cur), -1.0
    for k in range(len(cur) - 1, 0, -1):
        if len("".join(w["text"] for w in cur[:k]).strip()) < min_chars:
            break
        if _SOFT_BREAK.search(cur[k - 1]["text"].strip()) or (cjk and cur[k]["text"].startswith(" ")):
            return k
        gap = cur[k]["start"] - cur[k - 1]["end"]
        if gap > best_gap:
            best, best_gap = k, gap
    return best


def build_segments(words, lang):
    """Gộp từ thành các đoạn phụ đề dễ đọc: theo câu, khoảng lặng, độ dài, thời lượng.
    Giới hạn ký tự nguồn tính sao cho bản dịch tiếng Việt vừa ~1 dòng (CJK nở ~3x khi dịch)."""
    char_limit = {"zh": 14, "ja": 16, "ko": 24}.get(lang, 38)
    cjk = lang in ("zh", "ja")
    segs, cur = [], []

    def flush():
        if not cur:
            return
        text = "".join(w["text"] for w in cur).strip()
        if text:
            segs.append({"id": len(segs), "start": round(cur[0]["start"], 3),
                         "end": round(cur[-1]["end"], 3), "src": text, "vi": ""})
        cur.clear()

    for w in words:
        # Ngắt ở ranh giới câu Whisper đã chia (chuẩn ngữ nghĩa hơn cắt theo độ dài) hoặc khoảng lặng
        if cur and (w.get("seg_start") or w["start"] - cur[-1]["end"] > MAX_GAP_SECONDS):
            flush()
        cur.append(w)
        text = "".join(x["text"] for x in cur).strip()
        dur = cur[-1]["end"] - cur[0]["start"]
        sentence_done = bool(_SENTENCE_END.search(w["text"].strip())) and len(text) >= char_limit // 3
        if sentence_done:
            flush()
        elif dur >= MAX_SEG_SECONDS or len(text) >= char_limit:
            k = _split_point(cur, cjk, char_limit // 2)
            rest = cur[k:]
            del cur[k:]
            flush()
            cur.extend(rest)
    flush()

    # Đoạn quá ngắn thì kéo dài để kịp đọc, không đè lên đoạn sau
    for i, s in enumerate(segs):
        if s["end"] - s["start"] < MIN_SEG_SECONDS:
            limit = segs[i + 1]["start"] - 0.05 if i + 1 < len(segs) else s["start"] + MIN_SEG_SECONDS
            s["end"] = round(max(s["end"], min(s["start"] + MIN_SEG_SECONDS, limit)), 3)
    return segs


# ---------- 3b. Đọc phụ đề in sẵn trên hình (OCR) ----------
_ocr = None


def _get_ocr():
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
    return _ocr


def detect_sub_box(video: Path, samples=16):
    """Tìm vùng phụ đề in cứng: hàng chữ nằm giữa ngang, giữ nguyên độ cao nhưng nội dung đổi qua nhiều khung
    (logo/watermark đứng yên nên bị loại). Trả (x, y, w, h) theo tỉ lệ khung hình, None nếu không thấy."""
    import cv2
    ocr = _get_ocr()
    dur = probe_duration(video) or 1.0
    img = video.parent / "detect_frame.png"
    boxes = []  # (x0, y0, x1, y1, text), tỉ lệ 0..1
    for i in range(samples):
        t = dur * (0.05 + 0.9 * i / max(samples - 1, 1))
        _run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1",
              "-vf", "scale=960:-2", str(img)])
        frame = cv2.imread(str(img))
        if frame is None:
            continue
        H, W = frame.shape[:2]
        for pts, text, conf in ocr(frame)[0] or []:
            xs, ys = [q[0] for q in pts], [q[1] for q in pts]
            b = (min(xs) / W, min(ys) / H, max(xs) / W, max(ys) / H, text)
            if conf >= 0.6 and 0.3 < (b[0] + b[2]) / 2 < 0.7:
                boxes.append(b)
    img.unlink(missing_ok=True)

    # ponytail: chỉ lấy 1 hàng chữ; phụ đề 2 dòng thì dòng phụ có thể lọt ra ngoài, gộp hàng kề nếu gặp
    best = set()
    for a in boxes:
        ac, ah = (a[1] + a[3]) / 2, a[3] - a[1]
        row = {b for b in boxes if abs((b[1] + b[3]) / 2 - ac) < ah * 0.6}
        if len({b[4] for b in row}) > len({b[4] for b in best}):
            best = row
    if len({b[4] for b in best}) < 3:
        return None
    pad = sorted(b[3] - b[1] for b in best)[len(best) // 2] * 0.4
    y0, y1 = max(0.0, min(b[1] for b in best) - pad), min(1.0, max(b[3] for b in best) + pad)
    half = min(0.5, max(max(0.5 - b[0], b[2] - 0.5) for b in best) + 0.04)  # đối xứng quanh tâm
    return round(0.5 - half, 3), round(y0, 3), round(2 * half, 3), round(y1 - y0, 3)


def ocr_segments(video: Path, segs, on_progress=None):
    """Mỗi đoạn lấy 2 khung (30% và 70% thời lượng), cắt 35% phía dưới, đọc chữ nằm giữa ngang.
    Kết quả để ở s["ocr"], dùng làm bản tham chiếu khi dịch; thời gian vẫn theo giọng nói."""
    ocr = _get_ocr()
    img = video.parent / "ocr_frame.png"
    for n, s in enumerate(segs):
        texts = []
        for f in (0.3, 0.7):
            t = s["start"] + (s["end"] - s["start"]) * f
            _run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1",
                  "-vf", "crop=iw:ih*0.35:0:ih*0.65,scale=960:-2", str(img)])
            boxes = [b for b in (ocr(str(img))[0] or [])
                     if b[2] >= 0.7 and 0.25 * 960 < (b[0][0][0] + b[0][2][0]) / 2 < 0.75 * 960]
            text = " ".join(b[1] for b in sorted(boxes, key=lambda b: (round(b[0][0][1] / 20), b[0][0][0])))
            if text and text not in texts:
                texts.append(text)
        s["ocr"] = " / ".join(texts)
        if on_progress:
            on_progress((n + 1) / len(segs))
    img.unlink(missing_ok=True)
    return segs


# ---------- 4. Dịch ----------
SYSTEM_PROMPT = """Bạn là biên dịch viên phụ đề video chuyên nghiệp. Dịch từng dòng phụ đề sang tiếng Việt.
Quy tắc:
- Dịch tự nhiên theo văn nói của người Việt, không dịch word-by-word.
- Mỗi id là MỘT dòng phụ đề ngắn, tối đa 42 ký tự. Lược bỏ từ đệm, rút gọn ý để người xem đọc kịp.
- Văn bản gốc do máy nghe tự động nên có thể sai chữ đồng âm/gần âm (vd 柴→茶, 梅电梯→没电梯). Suy ra ý đúng theo ngữ cảnh rồi mới dịch.
- Một câu có thể bị chia qua nhiều id liên tiếp: dịch phần ý của id nào nằm ở id đó, sao cho đọc liền các dòng thành câu trọn vẹn, không lặp, không mất ý.
- Dựa vào ngữ cảnh để chọn xưng hô (anh/em, tôi/bạn, ông/cháu...) và giữ nhất quán.
- Giữ nguyên tên riêng, tên thương hiệu, thuật ngữ kỹ thuật nếu không có bản dịch phổ biến.
- Tuân thủ bảng thuật ngữ nếu được cung cấp.
- Nếu dòng có trường "screen" (phụ đề in sẵn trên hình, đọc bằng OCR, thường chính xác hơn "text" do máy nghe):
  dùng "screen" để sửa chữ nghe nhầm và bổ sung ý bị thiếu. Bỏ qua "screen" nếu nó không khớp lời nói (biển hiệu, logo, chữ của câu khác).
- Mỗi id dịch riêng, không gộp, không tách, không bỏ sót id nào.
- Chỉ trả về JSON hợp lệ dạng {"items":[{"id":0,"vi":"..."}]}, không giải thích, không markdown."""


def _call_claude(system: str, user: str) -> str:
    import anthropic
    client = anthropic.Anthropic()  # đọc ANTHROPIC_API_KEY từ môi trường
    msg = client.messages.create(model=CLAUDE_MODEL, max_tokens=8000, system=system,
                                 messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in msg.content if b.type == "text")


def _call_ollama(system: str, user: str) -> str:
    r = requests.post(f"{OLLAMA_URL}/api/chat", timeout=900, json={
        "model": OLLAMA_MODEL, "stream": False, "format": "json",
        "options": {"temperature": 0.2},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    })
    r.raise_for_status()
    return r.json()["message"]["content"]


def _call_openai(system: str, user: str) -> str:
    r = requests.post(f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions", timeout=300,
                      headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                      json={"model": OPENAI_MODEL, "temperature": 0.2,
                            "response_format": {"type": "json_object"},
                            "messages": [{"role": "system", "content": system},
                                         {"role": "user", "content": user}]})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _parse_translation(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip()
    data = json.loads(text[text.find("{"): text.rfind("}") + 1])
    return {int(x["id"]): str(x["vi"]).strip() for x in data.get("items", [])}


def translate_segments(segs, src_lang, glossary="", on_progress=None):
    call = {"ollama": _call_ollama, "openai": _call_openai}.get(TRANSLATOR, _call_claude)
    total = len(segs)
    for i in range(0, total, TRANSLATE_BATCH):
        chunk = segs[i:i + TRANSLATE_BATCH]
        context = [{"src": s["src"], "vi": s["vi"]} for s in segs[max(0, i - 5):i]]
        parts = [f"Ngôn ngữ nguồn (mã ISO): {src_lang}"]
        if glossary.strip():
            parts.append(f"Bảng thuật ngữ bắt buộc:\n{glossary.strip()}")
        if context:
            parts.append("Ngữ cảnh các dòng trước (chỉ tham khảo, không dịch lại):\n"
                         + json.dumps(context, ensure_ascii=False))
        parts.append("Các dòng cần dịch:\n"
                     + json.dumps([{"id": s["id"], "text": s["src"], **({"screen": s["ocr"]} if s.get("ocr") else {})}
                                 for s in chunk], ensure_ascii=False))
        user = "\n\n".join(parts)

        result = {}
        for attempt in range(3):
            try:
                result.update(_parse_translation(call(SYSTEM_PROMPT, user)))
                missing = [s["id"] for s in chunk if not result.get(s["id"])]
                if not missing:
                    break
                log.warning("Batch %d thiếu %d dòng, thử lại", i, len(missing))
            except Exception as e:  # lỗi mạng, JSON hỏng, rate limit...
                log.warning("Dịch lỗi (lần %d): %s", attempt + 1, e)
                time.sleep(2 * (attempt + 1))
        for s in chunk:
            s["vi"] = result.get(s["id"]) or s["vi"] or s["src"]  # dòng lỗi giữ bản gốc để sửa tay
        if on_progress:
            on_progress(min((i + len(chunk)) / total, 1.0))
    return segs


# ---------- 5. Xuất phụ đề ----------
def wrap_line(text: str, max_len=MAX_LINE_CHARS) -> str:
    """Ngắt thành tối đa 2 dòng cân đối."""
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    words, best = text.split(" "), None
    for k in range(1, len(words)):
        a, b = " ".join(words[:k]), " ".join(words[k:])
        score = max(len(a), len(b))
        if best is None or score < best[0]:
            best = (score, a, b)
    return f"{best[1]}\n{best[2]}" if best else text


def _ts(sec: float, sep=",") -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02}{sep}{ms:03}"


def write_srt(segs, path: Path):
    blocks = [f"{n}\n{_ts(s['start'])} --> {_ts(s['end'])}\n{wrap_line(s['vi'] or s['src'])}\n"
              for n, s in enumerate(segs, 1)]
    path.write_text("\n".join(blocks), encoding="utf-8")


def write_vtt(segs, path: Path):
    blocks = [f"{_ts(s['start'], '.')} --> {_ts(s['end'], '.')}\n{wrap_line(s['vi'] or s['src'])}\n"
              for s in segs]
    path.write_text("WEBVTT\n\n" + "\n".join(blocks), encoding="utf-8")


# ---------- 5b. Giọng đọc tiếng Việt (VieNeu-TTS v3 Turbo, chạy trên máy) ----------
# 25 giọng có sẵn của vieneu==3.8.4 (Vieneu().list_preset_voices()): tên -> (giới tính, vùng, phong cách).
# ponytail: ghi cứng để /api/voices không phải load model; nâng cấp vieneu thì chép lại danh sách.
VOICES = {
    "Hải Đăng": ("Nam", "Bắc", "tự nhiên"), "Ngọc Huyền": ("Nữ", "Bắc", "tự nhiên"),
    "Adam bựa": ("Nam", "Bắc", "tự nhiên"), "Trúc Ly": ("Nữ", "Bắc", "tự nhiên"),
    "Thiện Minh": ("Nam", "Bắc", "kể chuyện"), "Mai Anh": ("Nữ", "Bắc", "tin tức"),
    "Thiền Tâm Đức": ("Nam", "Bắc", "kể chuyện"), "Minh Đức": ("Nam", "Bắc", "tin tức"),
    "Phạm Tuyên": ("Nam", "Bắc", "tự nhiên"), "Xuân Vĩnh": ("Nam", "Bắc", "tự nhiên"),
    "Thanh Bình": ("Nam", "Bắc", "kể chuyện"), "Ngọc Linh": ("Nữ", "Bắc", "kể chuyện"),
    "Đoan Trang": ("Nữ", "Bắc", "tự nhiên"), "Quỳnh Anh": ("Nữ", "Bắc", "đọc truyện"),
    "Quốc Tuấn": ("Nam", "Bắc", "tự nhiên"),
    "Quang Sơn": ("Nam", "Trung", "tự nhiên"), "Ngọc Trân": ("Nữ", "Trung", "tự nhiên"),
    "Thùy Dung": ("Nữ", "Nam", "tin tức"), "Thái Sơn": ("Nam", "Nam", "kể chuyện"),
    "Thục Đoan": ("Nữ", "Nam", "kể chuyện"), "Minh Triết": ("Nam", "Nam", "tin tức"),
    "Mỹ Duyên": ("Nữ", "Nam", "đọc truyện"), "Đức Trí": ("Nam", "Nam", "đọc truyện"),
    "Kim Thanh": ("Nữ", "Nam", "đọc truyện"), "Adam": ("Nam", "Nam", "tự nhiên"),
}
# Job cũ lưu khóa giọng edge-tts -> giọng VieNeu gần nhất
OLD_VOICES = {"nu": "Ngọc Huyền", "nu-cao": "Trúc Ly", "nu-tram": "Ngọc Linh",
              "nam": "Hải Đăng", "nam-tram": "Thanh Bình", "nam-cao": "Adam bựa"}
TTS_BG_VOLUME = round(float(os.getenv("TTS_BG_VOLUME", "0.15")) * 100)  # % tiếng gốc mặc định khi có giọng đọc


def valid_bg_volume(v) -> int:
    """% âm lượng tiếng gốc khi lồng tiếng, 0 = tắt hẳn."""
    try:
        return max(0, min(100, int(v)))
    except (TypeError, ValueError):
        return TTS_BG_VOLUME
# Câu dài hơn khung (tới mốc câu sau) được tăng tốc tối đa chừng này, phần còn thừa thì đẩy câu sau lùi lại.
# Cao quá nghe lúc nhanh lúc chậm; thấp quá thì lời trễ dần so với hình ở đoạn thoại dày.
TTS_MAX_TEMPO = float(os.getenv("TTS_MAX_TEMPO", "1.15"))
TTS_RATE = 48000
TTS_GAP = 0.15  # nghỉ giữa 2 câu khi phải kéo dài video cho kịp đọc
TTS_PRECISION = os.getenv("TTS_PRECISION", "fp32")  # fp32 | int8 (nhanh ~2x, cần CPU có AVX-VNNI)
_RATE_RE = re.compile(r"[+-]\d{1,2}%")
_tts_lock = threading.Lock()  # 1 model dùng chung, chạy lần lượt từng câu


def valid_rate(rate: str) -> str:
    return rate if _RATE_RE.fullmatch(rate or "") else "+0%"


def voice_key(v) -> str:
    """Khóa giọng hợp lệ, "" = không lồng tiếng."""
    v = OLD_VOICES.get(v, v)
    return v if v in VOICES else ""


def rate_factor(rate: str) -> float:
    """"+15%" -> 1.15. VieNeu Turbo không có tham số tốc độ nên tốc độ đọc làm bằng atempo."""
    return 1 + int(valid_rate(rate)[:-1]) / 100


_tts_model = None


def _tts():
    global _tts_model
    if _tts_model is None:
        from vieneu import Vieneu
        log.info("Đang nạp model giọng đọc VieNeu-TTS (%s)...", TTS_PRECISION)
        _tts_model = Vieneu(precision=TTS_PRECISION)
    return _tts_model


def tts_clip(text: str, voice_key: str, out: Path) -> Path:
    """Đọc text bằng giọng voice_key, ghi WAV 48 kHz (tốc độ gốc, chỉnh tốc độ ở khâu ghép)."""
    tmp = out.with_name(out.name + ".part.wav")
    with _tts_lock:  # review và xuất video có thể cùng đọc 1 câu -> kiểm tra lại trong khóa
        if not out.exists():
            m = _tts()
            m.save(m.infer(text, voice=voice_key), str(tmp))
            tmp.replace(out)
    return out


def clip_tempo(dur: float, slot: float, rate: str) -> float:
    """Hệ số atempo: tốc độ người dùng chọn, rồi tăng thêm (tối đa TTS_MAX_TEMPO) nếu câu vẫn tràn khung."""
    base = rate_factor(rate)
    dur /= base
    return base * (min(dur / slot, TTS_MAX_TEMPO) if slot > 0 and dur > slot else 1.0)


def place_clips(clips):
    """clips: [(giây bắt đầu, số mẫu)] -> vị trí mẫu của từng clip. Đặt đúng mốc, nếu câu trước
    còn đọc dở thì lùi câu sau lại cho khỏi đè tiếng."""
    pos, cursor = [], 0
    for start, n in clips:
        p = max(int(round(start * TTS_RATE)), cursor)
        pos.append(p)
        cursor = p + n
    return pos


def timeline_map(t: float, stretches) -> float:
    """Thời điểm t của video gốc -> thời điểm trong video đã kéo dài. stretches: [(a, b, f)] = đoạn
    [a, b) được làm chậm f lần."""
    return t + sum((f - 1) * min(max(t - a, 0.0), b - a) for a, b, f in stretches)


def build_voiceover(job_dir: Path, segs, voice_key: str, rate: str, duration=float("inf")):
    """Đọc từng đoạn phụ đề rồi ghép thành 1 track WAV. Câu nào đọc không kịp (đã tăng tốc tối đa
    TTS_MAX_TEMPO) thì đoạn hình tương ứng được kéo dài cho vừa -> trả (file wav, stretches) để
    stretch_video làm chậm đúng các đoạn đó; track giọng đọc đã theo timeline sau khi kéo dài.
    Clip được cache theo nội dung nên sửa 1 dòng chỉ đọc lại dòng đó."""
    import hashlib
    import wave
    cache = job_dir / "tts"
    cache.mkdir(exist_ok=True)
    rate = valid_rate(rate)
    items = []  # (seg, text, wav)
    for s in sorted(segs, key=lambda s: s["start"]):
        text = " ".join((s.get("vi") or s.get("src") or "").split())
        if re.search(r"\w", text):  # bỏ dòng chỉ có dấu câu
            key = hashlib.sha1(f"vieneu|{voice_key}|{text}".encode()).hexdigest()[:16]
            items.append((s, text, cache / f"{key}.wav"))

    for s, text, src in items:  # CPU chạy tuần tự, song song không nhanh hơn
        if not src.exists():
            tts_clip(text, voice_key, src)

    # Giải mã + tăng tốc câu nào dài hơn khung (tới mốc câu sau) rồi xếp lên timeline
    pcm, stretches = [], []
    for n, (s, text, src) in enumerate(items):
        end = items[n + 1][0]["start"] if n + 1 < len(items) else max(duration, s["start"])
        slot = end - s["start"]
        tempo = clip_tempo(probe_duration(src), slot, rate)
        wav = cache / f"clip.{threading.get_ident()}.wav"  # review và xuất video có thể chạy cùng lúc
        _run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", str(TTS_RATE),
              *(["-af", f"atempo={tempo:.3f}"] if abs(tempo - 1) > 0.01 else []), "-c:a", "pcm_s16le", str(wav)])
        with wave.open(str(wav)) as w:
            pcm.append((s["start"], w.readframes(w.getnframes())))
        wav.unlink()
        need = len(pcm[-1][1]) / 2 / TTS_RATE + TTS_GAP
        if need > slot + 0.05 and 0 < slot < float("inf"):  # 2 câu cùng mốc (slot 0): để place_clips lùi câu sau
            stretches.append((s["start"], end, need / slot))
    pcm = [(timeline_map(t, stretches), d) for t, d in pcm]

    out = "vi_voice.wav"
    tmp = job_dir / f"{out}.{threading.get_ident()}.part"
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TTS_RATE)
        written = 0
        for p, (_, data) in zip(place_clips([(t, len(d) // 2) for t, d in pcm]), pcm):
            w.writeframes(b"\0\0" * (p - written) + data)
            written = p + len(data) // 2
    tmp.replace(job_dir / out)
    return out, stretches


def _atempo(speed: float) -> str:
    """atempo chỉ nhận 0.5..100 -> nối nhiều nấc khi cần chậm hơn 0.5."""
    parts = []
    while speed < 0.5:
        parts.append("atempo=0.5")
        speed /= 0.5
    return ",".join(parts + [f"atempo={speed:.5f}"])


def stretch_video(job_dir: Path, video_name: str, stretches, segs):
    """Làm chậm các đoạn [a, b) f lần (hình + tiếng gốc) và dời phụ đề theo. Trả (video, srt) mới."""
    # ffmpeg không nhận 1 biểu thức quá dài -> nhiều setpts nối nhau, áp từ đoạn cuối về đầu: đoạn sau chỉ
    # dời các mốc nằm sau nó nên mốc a, b của đoạn trước vẫn đúng như video gốc.
    groups = [stretches[i:i + 25] for i in range(0, len(stretches), 25)][::-1]
    fc = "[0:v]" + ",".join("setpts='(T" + "".join(f"+{f - 1:.5f}*clip(T-{a:.3f},0,{b - a:.3f})" for a, b, f in g)
                            + ")/TB'" for g in groups) + "[v]"
    amap = []
    if has_audio(job_dir / video_name):
        # Cắt tiếng gốc thành các khúc liền nhau, khúc nào cần kéo dài thì atempo chậm lại rồi nối
        cuts, t = [], 0.0
        for a, b, f in stretches:
            if a > t:
                cuts.append((t, a, 1.0))
            cuts.append((a, b, f))
            t = b
        cuts.append((t, None, 1.0))
        fc += f";[0:a]asplit={len(cuts)}" + "".join(f"[s{i}]" for i in range(len(cuts)))
        for i, (a, b, f) in enumerate(cuts):
            trim = f"start={a:.3f}" + (f":end={b:.3f}" if b is not None else "")
            fc += f";[s{i}]atrim={trim},asetpts=PTS-STARTPTS" + (f",{_atempo(1 / f)}" if f > 1 else "") + f"[p{i}]"
        fc += ";" + "".join(f"[p{i}]" for i in range(len(cuts))) + f"concat=n={len(cuts)}:v=0:a=1[a]"
        amap = ["-map", "[a]", "-c:a", "aac", "-b:a", "192k"]
    (job_dir / "stretch.filter").write_text(fc)
    out = "input_stretched.mp4"
    # ponytail: encode 2 lần (kéo dài rồi mới in phụ đề), crf 18 nên gần như không mất chất lượng
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", video_name, "-filter_complex_script", "stretch.filter",
          "-map", "[v]", *amap, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", out], cwd=job_dir)
    write_srt([{**s, "start": timeline_map(s["start"], stretches), "end": timeline_map(s["end"], stretches)}
               for s in segs], job_dir / "vi_stretched.srt")
    return out, "vi_stretched.srt"


def has_audio(path: Path) -> bool:
    return bool(_run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                      "-of", "csv=p=0", str(path)]).strip())


def _voice_mix(video: Path, idx: int, bg_volume=TTS_BG_VOLUME):
    """Filter trộn tiếng gốc (bg_volume %) + giọng đọc ở input idx. Trả (filter, nhãn để -map).
    bg_volume=0 / video không có tiếng: chỉ dùng giọng đọc."""
    if not bg_volume or not has_audio(video):
        return "", f"{idx}:a"
    return (f"[0:a]volume={bg_volume / 100}[a0];[a0][{idx}:a]amix=inputs=2:duration=first:normalize=0[aout]",
            "[aout]")


# ---------- 6. Gắn phụ đề vào video ----------
# Vùng dự phòng (tỉ lệ x, y, rộng, cao) khi detect_sub_box không tìm thấy phụ đề gốc.
COVER_BOX = (0.15, 0.81, 0.70, 0.11)
BRAND_FONT = os.getenv("BRAND_FONT", "/usr/share/fonts/truetype/noto/NotoSans-BoldItalic.ttf")
BRAND_LOGO = Path(os.getenv("BRAND_LOGO", Path(__file__).parent / "brand_logo.png"))
BRAND_RED, BRAND_DARK = (139, 10, 20), (28, 28, 28)


def probe_size(path: Path):
    out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                "-of", "csv=p=0", str(path)])
    w, h = out.strip().split(",")[:2]
    return int(w), int(h)


def make_brand_overlay(W: int, H: int, title: str, out: Path) -> Path:
    """Ảnh PNG trong suốt cỡ khung hình: thanh tiêu đề trên cùng (đỏ | đen, chéo ở giữa) + logo tròn góc phải."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    bar = max(24, round(W * 0.045))
    mid, slant = W // 2, bar // 2
    d.rectangle([0, 0, W, bar], fill=BRAND_DARK)
    d.polygon([(0, 0), (mid + slant, 0), (mid - slant, bar), (0, bar)], fill=BRAND_RED)

    # Từ cuối nằm bên đen (chữ đỏ), phần còn lại bên đỏ (chữ đen)
    words = title.upper().split()
    left, right = " ".join(words[:-1] or words), " ".join(words[-1:]) if len(words) > 1 else ""
    font = ImageFont.truetype(BRAND_FONT, round(bar * 0.72))
    cy, gap = bar / 2, bar * 0.6
    d.text((mid - gap, cy), left, font=font, fill=(17, 17, 17), anchor="rm")
    if right:
        d.text((mid + gap, cy), right, font=font, fill=(200, 24, 36), anchor="lm")

    r = round(W * 0.05)
    cx, cyl = W - r - round(W * 0.012), bar + r - round(r * 0.35)
    if BRAND_LOGO.exists():  # logo kênh có sẵn (PNG tròn nền trong suốt)
        logo = Image.open(BRAND_LOGO).convert("RGBA").resize((2 * r, 2 * r), Image.LANCZOS)
        img.alpha_composite(logo, (cx - r, cyl - r))
        img.save(out)
        return out

    # Không có file logo: vẽ logo tròn viền tối, nền đỏ, chữ trắng 2 dòng
    d.ellipse([cx - r, cyl - r, cx + r, cyl + r], fill=BRAND_DARK)
    ri = round(r * 0.86)
    d.ellipse([cx - ri, cyl - ri, cx + ri, cyl + ri], fill=BRAND_RED)
    lines = [left, right] if right else [left]
    size = round(ri * 0.42)
    while size > 6:  # thu nhỏ chữ tới khi vừa lòng tròn
        f = ImageFont.truetype(BRAND_FONT, size)
        if max(d.textlength(l, font=f) for l in lines) <= ri * 1.6:
            break
        size -= 1
    d.multiline_text((cx, cyl), "\n".join(lines), font=f, fill="white", anchor="mm", align="center",
                     spacing=round(size * 0.25))
    img.save(out)
    return out


def render_video(job_dir: Path, video_name: str, mode="hard", font_size=13, cover="", brand="", voice="",
                 srt_name="vi.srt", bg_volume=TTS_BG_VOLUME) -> str:
    """mode=hard: in cứng phụ đề (encode lại). mode=soft: nhúng track phụ đề (nhanh, bật/tắt được).
    voice = file WAV giọng đọc (build_voiceover): trộn với tiếng gốc nhỏ lại; soft giữ thêm track tiếng gốc.
    Chỉ với hard: cover="blur" làm mờ / "black" tô đen dải phụ đề gốc; brand=chữ trên thanh tiêu đề + logo.
    FontSize tính trên chiều cao chuẩn 288 của libass."""
    out = f"output_{mode}.mp4"
    if mode == "hard":
        cover = "blur" if cover is True else (cover or "")
        chain, margin_v, inputs, srt = ["[0:v]null"], 28, ["-i", video_name], srt_name
        if cover:
            x, y, w, h = detect_sub_box(job_dir / video_name) or COVER_BOX
            log.info("Vùng che phụ đề gốc (%s): %s", cover, (x, y, w, h))
            if cover == "black":
                # Dải đen hết chiều ngang; phụ đề gốc ở sát đáy thì tô tới mép dưới như kiểu khung phim
                y1 = 1.0 if y + h > 0.75 else y + h
                chain.append(f"drawbox=x=0:y=ih*{y}:w=iw:h=ih*{y1 - y:.3f}:color=black:t=fill")
                # Neo giữa chữ vào giữa dải đen (\an5\pos) để 1 hay 2 dòng đều nằm gọn trong dải.
                # Hệ tọa độ phụ đề SRT của libass: 384 x 288.
                tag = f"{{\\an5\\pos(192,{(y + y1) / 2 * 288:.1f})}}"
                srt = "vi_render.srt"
                (job_dir / srt).write_text(re.sub(r"(--> [^\n]*\n)", lambda m: m.group(1) + tag,
                                                  (job_dir / srt_name).read_text(encoding="utf-8")), encoding="utf-8")
            else:
                chain = [f"[0:v]split[v][c];[c]crop=iw*{w}:ih*{h}:iw*{x}:ih*{y},"
                         "boxblur=luma_radius='min(h/2-1,20)':luma_power=3:chroma_radius='min(ch/2-1,10)':chroma_power=3[b];"
                         f"[v][b]overlay=W*{x}:H*{y}"]
                margin_v = max(4, int((1 - y - h) * 288))  # đặt phụ đề Việt lên đúng vùng đã làm mờ
        if brand.strip():
            make_brand_overlay(*probe_size(job_dir / video_name), brand.strip(), job_dir / "brand.png")
            inputs += ["-i", "brand.png"]
            chain[-1] += "[pre];[pre][1:v]overlay=0:0"
        style = (f"FontName={SUB_FONT},FontSize={int(font_size)},PrimaryColour=&H00FFFFFF,"
                 f"OutlineColour=&H00000000,BorderStyle=1,Outline=1.5,Shadow=0,MarginV={margin_v}")
        fc = ",".join(chain) + f",subtitles={srt}:force_style='{style}'[out]"
        amap = "0:a?"
        if voice:
            mix, amap = _voice_mix(job_dir / video_name, len(inputs) // 2, bg_volume)
            inputs += ["-i", voice]
            fc += f";{mix}" if mix else ""
        _run(["ffmpeg", "-y", *inputs, "-filter_complex", fc, "-map", "[out]", "-map", amap,
              "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
              "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out], cwd=job_dir)
    elif voice:  # track 1 = lồng tiếng (mặc định), track 2 = tiếng gốc (bỏ khi tắt hẳn tiếng gốc)
        mix, amap = _voice_mix(job_dir / video_name, 2, bg_volume)
        _run(["ffmpeg", "-y", "-i", video_name, "-i", srt_name, "-i", voice,
              *(["-filter_complex", mix] if mix else []),
              "-map", "0:v", "-map", amap, *(["-map", "0:a?"] if bg_volume else []), "-map", "1:0",
              "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-c:s", "mov_text",
              "-metadata:s:a:0", "language=vie",
              "-disposition:a:0", "default", *(["-disposition:a:1", "0"] if mix else []),
              "-metadata:s:s:0", "language=vie", "-disposition:s:0", "default",
              "-movflags", "+faststart", out], cwd=job_dir)
    else:
        _run(["ffmpeg", "-y", "-i", video_name, "-i", "vi.srt",
              "-map", "0:v", "-map", "0:a?", "-map", "1:0",
              "-c:v", "copy", "-c:a", "copy", "-c:s", "mov_text",
              "-metadata:s:s:0", "language=vie", "-disposition:s:0", "default",
              "-movflags", "+faststart", out], cwd=job_dir)
    return out
