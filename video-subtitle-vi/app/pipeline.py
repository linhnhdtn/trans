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


def render_video(job_dir: Path, video_name: str, mode="hard", font_size=13, cover="", brand="") -> str:
    """mode=hard: in cứng phụ đề (encode lại). mode=soft: nhúng track phụ đề (nhanh, bật/tắt được).
    Chỉ với hard: cover="blur" làm mờ / "black" tô đen dải phụ đề gốc; brand=chữ trên thanh tiêu đề + logo.
    FontSize tính trên chiều cao chuẩn 288 của libass."""
    out = f"output_{mode}.mp4"
    if mode == "hard":
        cover = "blur" if cover is True else (cover or "")
        chain, margin_v, inputs, srt = ["[0:v]null"], 28, ["-i", video_name], "vi.srt"
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
                                                  (job_dir / "vi.srt").read_text(encoding="utf-8")), encoding="utf-8")
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
        _run(["ffmpeg", "-y", *inputs, "-filter_complex", fc, "-map", "[out]", "-map", "0:a?",
              "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
              "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out], cwd=job_dir)
    else:
        _run(["ffmpeg", "-y", "-i", video_name, "-i", "vi.srt",
              "-map", "0:v", "-map", "0:a?", "-map", "1:0",
              "-c:v", "copy", "-c:a", "copy", "-c:s", "mov_text",
              "-metadata:s:s:0", "language=vie", "-disposition:s:0", "default",
              "-movflags", "+faststart", out], cwd=job_dir)
    return out
