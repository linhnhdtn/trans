import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import urlparse

import yt_dlp

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
JOBS_DIR = DATA_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "2048"))
ALLOWED_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
DL_DIR = DATA_DIR / "downloads"
# Chỉ nhận link các nền tảng này: yt-dlp có extractor "generic" tải được URL bất kỳ (kể cả mạng nội bộ)
DL_HOSTS = ("tiktok.com", "douyin.com", "iesdouyin.com", "facebook.com", "fb.watch", "fb.com", "youtube.com", "youtu.be")

# 1 worker = xử lý tuần tự, tránh tràn VRAM khi chỉ có 1 GPU
executor = ThreadPoolExecutor(max_workers=int(os.getenv("WORKERS", "1")))
_lock = threading.Lock()

app = FastAPI(title="Video Subtitle VI")


# ---------- Lưu trạng thái job (file JSON, đủ cho MVP) ----------
def job_dir(jid: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", jid):
        raise HTTPException(404, "Không tìm thấy job")
    d = JOBS_DIR / jid
    if not d.exists():
        raise HTTPException(404, "Không tìm thấy job")
    return d


def load_job(jid: str) -> dict:
    return json.loads((job_dir(jid) / "job.json").read_text(encoding="utf-8"))


def save_job(job: dict):
    path = JOBS_DIR / job["id"] / "job.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def update_job(jid: str, **fields) -> dict:
    with _lock:
        job = load_job(jid)
        job.update(fields, updated_at=time.time())
        save_job(job)
        return job


def fail_interrupted_jobs():
    """Hàng đợi nằm trong RAM: job còn dở lúc khởi động là đã chết theo tiến trình cũ, đừng để kẹt mãi."""
    for p in JOBS_DIR.glob("*/job.json"):
        try:
            job = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        fix = {}
        if job.get("status") not in ("ready", "error"):
            fix.update(status="error", step="Xử lý bị gián đoạn",
                       error="Máy chủ khởi động lại khi đang xử lý. Hãy tạo lại video này.")
        if job.get("render_status") == "rendering":
            fix.update(render_status="error", render_error="Máy chủ khởi động lại khi đang xuất. Hãy xuất lại.")
        if fix:
            save_job({**job, **fix})


fail_interrupted_jobs()


def throttled_progress(jid: str, base: float, span: float):
    last = [0.0]

    def cb(p: float):
        now = time.time()
        if now - last[0] >= 1.0 or p >= 1.0:
            last[0] = now
            update_job(jid, progress=round(base + span * p, 3))
    return cb


def write_subs(jid: str, segs):
    d = job_dir(jid)
    pipeline.write_srt(segs, d / "vi.srt")
    pipeline.write_vtt(segs, d / "vi.vtt")


# ---------- Xử lý nền ----------
def process_job(jid: str):
    try:
        job = load_job(jid)
        d = job_dir(jid)
        video = d / job["video"]

        update_job(jid, status="extracting", step="Đang tách âm thanh", progress=0.02)
        audio = pipeline.extract_audio(video, d / "audio.wav")
        duration = pipeline.probe_duration(video)

        update_job(jid, status="transcribing", step="Đang nghe giọng nói trong video",
                   progress=0.05, duration=duration)
        words, lang, prob = pipeline.transcribe(
            str(audio), job.get("source_lang") or None, duration,
            on_progress=throttled_progress(jid, 0.05, 0.55))
        segs = pipeline.build_segments(words, lang)
        if not segs:
            raise RuntimeError("Không phát hiện giọng nói trong video.")

        if job.get("ocr"):
            update_job(jid, status="ocr", step="Đang đọc chữ có sẵn trên hình", progress=0.6,
                       detected_lang=lang, lang_prob=round(prob, 3))
            pipeline.ocr_segments(video, segs, on_progress=throttled_progress(jid, 0.6, 0.15))

        update_job(jid, status="translating", step="Đang dịch sang tiếng Việt", progress=0.75,
                   detected_lang=lang, lang_prob=round(prob, 3), segments=segs)
        if lang == "vi":
            for s in segs:
                s["vi"] = s["src"]
        else:
            pipeline.translate_segments(segs, lang, job.get("glossary", ""),
                                        on_progress=throttled_progress(jid, 0.75, 0.23))

        write_subs(jid, segs)
        (d / "audio.wav").unlink(missing_ok=True)
        auto = job.get("auto_render")
        update_job(jid, status="ready", step="Phụ đề đã sẵn sàng", progress=1.0, segments=segs,
                   **({"render_status": "rendering", "render_mode": auto} if auto else {}))
        if auto:  # chạy luôn trong worker hiện tại, khỏi xếp hàng lại
            render_job(jid, auto, job.get("font_size", 13), job.get("cover", ""), job.get("brand", ""),
                       clean_voice(job.get("voice", "")), job.get("voice_rate", "+0%"),
                       pipeline.valid_bg_volume(job.get("bg_volume")))
    except Exception as e:
        log.exception("Job %s lỗi", jid)
        update_job(jid, status="error", step="Xử lý thất bại", error=str(e)[:2000])


def render_job(jid: str, mode: str, font_size: int, cover="", brand: str = "", voice="", voice_rate="+0%",
               bg_volume=pipeline.TTS_BG_VOLUME):
    try:
        job = load_job(jid)
        d, audio, video, srt = job_dir(jid), "", job["video"], "vi.srt"
        if voice:
            update_job(jid, render_step="Đang tạo giọng đọc tiếng Việt")
            audio, stretches = pipeline.build_voiceover(d, job["segments"], voice, voice_rate,
                                                        job.get("duration") or float("inf"))
            if stretches:
                update_job(jid, render_step=f"Đang kéo dài {len(stretches)} đoạn hình cho kịp lời đọc")
                video, srt = pipeline.stretch_video(d, video, stretches, job["segments"])
        update_job(jid, render_step="Đang xuất video")
        out = pipeline.render_video(d, video, mode, font_size, cover, brand, audio, srt, bg_volume)
        outputs = {**job.get("outputs", {}), mode: out}
        update_job(jid, render_status="done", render_error=None, outputs=outputs)
    except Exception as e:
        log.exception("Render %s lỗi", jid)
        update_job(jid, render_status="error", render_error=str(e)[:2000])


def clean_voice(v) -> str:
    """"" = không lồng tiếng (mặc định). Khóa giọng edge-tts cũ được đổi sang giọng VieNeu."""
    return pipeline.voice_key(v)


def clean_cover(v) -> str:
    """"" = giữ nguyên, "blur" = làm mờ, "black" = tô đen. Nhận cả kiểu cũ true/"1" = làm mờ."""
    return "blur" if v in (True, "1") else v if v in ("blur", "black") else ""


# ---------- API ----------
@app.post("/api/jobs")
async def create_job(file: Optional[UploadFile] = File(None), download_id: str = Form(""),
                     source_lang: str = Form(""), glossary: str = Form(""),
                     ocr: str = Form(""), auto_render: str = Form(""), font_size: int = Form(13),
                     cover: str = Form(""), brand: str = Form(""), voice: str = Form(""),
                     voice_rate: str = Form("+0%"), bg_volume: str = Form("")):
    if download_id:  # video đã tải bằng tab "Tải video": chép trên máy chủ, khỏi upload lại
        meta = load_download(download_id)
        src, filename = DL_DIR / download_id / meta["file"], download_name(meta)
    elif file:
        src, filename = None, file.filename or ""
    else:
        raise HTTPException(400, "Chưa chọn video")
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Định dạng không hỗ trợ. Dùng: {', '.join(sorted(ALLOWED_EXT))}")

    jid = uuid.uuid4().hex
    d = JOBS_DIR / jid
    d.mkdir(parents=True)
    video_name = f"input{ext}"
    if src:
        shutil.copyfile(src, d / video_name)
    else:
        size, limit = 0, MAX_UPLOAD_MB * 1024 * 1024
        with open(d / video_name, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    f.close()
                    shutil.rmtree(d, ignore_errors=True)
                    raise HTTPException(413, f"File vượt quá {MAX_UPLOAD_MB} MB")
                f.write(chunk)

    job = {"id": jid, "filename": filename, "video": video_name, "status": "queued",
           "step": "Đang chờ xử lý", "progress": 0.0, "source_lang": source_lang.strip(),
           "glossary": glossary[:5000], "ocr": ocr == "1",
           "auto_render": auto_render if auto_render in ("hard", "soft") else "",
           "font_size": max(8, min(font_size, 48)), "cover": clean_cover(cover),
           "brand": brand.strip()[:60], "voice": clean_voice(voice), "voice_rate": pipeline.valid_rate(voice_rate),
           "bg_volume": pipeline.valid_bg_volume(bg_volume), "segments": [], "outputs": {}, "render_status": None,
           "created_at": time.time(), "updated_at": time.time()}
    save_job(job)
    executor.submit(process_job, jid)
    return {"id": jid}


@app.get("/api/jobs")
def list_jobs():
    # ponytail: quét thư mục mỗi lần gọi, chuyển sang DB/index nếu số job lên hàng nghìn
    jobs = []
    for p in JOBS_DIR.glob("*/job.json"):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        jobs.append({k: j.get(k) for k in ("id", "filename", "status", "step", "progress", "detected_lang",
                                           "duration", "outputs", "render_status", "created_at")})
    jobs.sort(key=lambda j: j.get("created_at") or 0, reverse=True)
    return jobs


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    return load_job(jid)


@app.delete("/api/jobs/{jid}")
def delete_job(jid: str):
    with _lock:
        job = load_job(jid)
        if job["status"] not in ("ready", "error") or job.get("render_status") == "rendering":
            raise HTTPException(409, "Video đang được xử lý, chờ xong rồi mới xoá được")
        shutil.rmtree(job_dir(jid))
    return {"ok": True}


class SegmentEdit(BaseModel):
    id: int
    vi: str
    start: Optional[float] = None
    end: Optional[float] = None


class SegmentsUpdate(BaseModel):
    segments: List[SegmentEdit]


@app.put("/api/jobs/{jid}/segments")
def update_segments(jid: str, body: SegmentsUpdate):
    with _lock:
        job = load_job(jid)
        if job["status"] != "ready":
            raise HTTPException(409, "Phụ đề chưa sẵn sàng để chỉnh sửa")
        by_id = {s["id"]: s for s in job["segments"]}
        for e in body.segments:
            s = by_id.get(e.id)
            if not s:
                continue
            s["vi"] = e.vi.strip()
            if e.start is not None and e.end is not None and 0 <= e.start < e.end:
                s["start"], s["end"] = round(e.start, 3), round(e.end, 3)
        job["outputs"] = {}  # phụ đề đổi thì video đã xuất không còn đúng
        job["updated_at"] = time.time()
        save_job(job)
    write_subs(jid, job["segments"])
    return {"ok": True}


class RenderRequest(BaseModel):
    mode: str = "hard"
    font_size: int = 13
    cover: Union[bool, str] = ""
    brand: str = ""
    voice: str = ""
    voice_rate: str = "+0%"
    bg_volume: int = pipeline.TTS_BG_VOLUME  # % tiếng gốc khi lồng tiếng, 0 = chỉ còn giọng đọc AI


@app.post("/api/jobs/{jid}/render")
def render(jid: str, body: RenderRequest):
    job = load_job(jid)
    if job["status"] != "ready":
        raise HTTPException(409, "Phụ đề chưa sẵn sàng")
    if job.get("render_status") == "rendering":
        raise HTTPException(409, "Video đang được xuất")
    if body.mode not in ("hard", "soft"):
        raise HTTPException(400, "mode phải là hard hoặc soft")
    voice, rate = clean_voice(body.voice), pipeline.valid_rate(body.voice_rate)
    update_job(jid, render_status="rendering", render_mode=body.mode, render_error=None,
               voice=voice, voice_rate=rate, bg_volume=pipeline.valid_bg_volume(body.bg_volume))  # nhớ lựa chọn cho lần mở sau
    executor.submit(render_job, jid, body.mode, max(8, min(body.font_size, 48)),
                    clean_cover(body.cover), body.brand.strip()[:60], voice, rate, pipeline.valid_bg_volume(body.bg_volume))
    return {"ok": True}


class VoiceoverRequest(BaseModel):
    voice: str
    voice_rate: str = "+0%"


@app.post("/api/jobs/{jid}/voiceover")
def voiceover(jid: str, body: VoiceoverRequest):
    """Tạo track lồng tiếng để nghe thử ngay khi review (không xuất video). Câu đã đọc lấy từ cache."""
    job = load_job(jid)
    voice = clean_voice(body.voice)
    if job["status"] != "ready" or not voice:
        raise HTTPException(409, "Chưa có phụ đề hoặc chưa chọn giọng")
    # ponytail: chạy đồng bộ trong request, video dài (vài trăm câu) chờ vài phút; chuyển sang job nền nếu cần
    _, stretches = pipeline.build_voiceover(job_dir(jid), job["segments"], voice, pipeline.valid_rate(body.voice_rate),
                                            job.get("duration") or float("inf"))
    return {"stretches": stretches}  # trình phát làm chậm video đúng các đoạn này cho khớp giọng


@app.get("/api/jobs/{jid}/files/{kind}")
def download(jid: str, kind: str):
    job = load_job(jid)
    d = job_dir(jid)
    if kind == "thumb":
        thumb = d / "thumb.jpg"
        if not thumb.exists():  # tạo lần đầu khi được xem, áp dụng cả cho job cũ
            tmp = d / f"thumb.{uuid.uuid4().hex}.jpg"
            t = min(1.0, (job.get("duration") or 2) / 2)
            try:
                pipeline._run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(d / job["video"]),
                               "-frames:v", "1", "-vf", "scale=480:-2", "-q:v", "4", str(tmp)])
                tmp.replace(thumb)
            except Exception:
                tmp.unlink(missing_ok=True)
                raise HTTPException(404, "Không tạo được ảnh xem trước")
        return FileResponse(thumb)
    stem = Path(job["filename"]).stem or "video"
    files = {
        "video": (job["video"], None),
        "srt": ("vi.srt", f"{stem}.vi.srt"),
        "vtt": ("vi.vtt", None),
        "voice": ("vi_voice.wav", None),
        "hard": (job.get("outputs", {}).get("hard"), f"{stem}.vi.hardsub.mp4"),
        "soft": (job.get("outputs", {}).get("soft"), f"{stem}.vi.softsub.mp4"),
    }
    name, download_name = files.get(kind, (None, None))
    if not name or not (d / name).exists():
        raise HTTPException(404, "File chưa có")
    return FileResponse(d / name, filename=download_name)


# ---------- Tải video từ TikTok / Douyin / Facebook / YouTube ----------
class DownloadRequest(BaseModel):
    url: str


@app.post("/api/downloads")
def create_download(body: DownloadRequest):
    url = body.url.strip()
    u = urlparse(url)
    host = (u.hostname or "").lower()
    if u.scheme not in ("http", "https") or not any(host == h or host.endswith("." + h) for h in DL_HOSTS):
        raise HTTPException(400, "Chỉ hỗ trợ link TikTok, Douyin, Facebook, YouTube")

    did = uuid.uuid4().hex
    d = DL_DIR / did
    d.mkdir(parents=True)
    opts = {
        "outtmpl": str(d / "video.%(ext)s"),
        # Ưu tiên H.264 + AAC để xem được trên mọi trình duyệt/điện thoại
        "format": "bv*[vcodec~='^(avc|h264)']+ba[ext=m4a]/b[vcodec~='^(avc|h264)']/b[ext=mp4]/bv*+ba/b",
        "merge_output_format": "mp4",
        "noplaylist": True, "quiet": True, "no_warnings": True,
        "max_filesize": MAX_UPLOAD_MB * 1024 * 1024,
    }
    if cookies := os.getenv("YTDLP_COOKIES"):  # Douyin/Facebook riêng tư thường cần cookie trình duyệt
        opts["cookiefile"] = cookies
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url)
        video = next(p for p in d.glob("video.*") if p.suffix not in (".part", ".ytdl"))
    except Exception as e:
        shutil.rmtree(d, ignore_errors=True)
        log.warning("Tải %s lỗi: %s", url, e)
        msg = re.sub(r"\x1b\[[0-9;]*m", "", str(e)).removeprefix("ERROR: ")
        raise HTTPException(422, f"Không tải được video: {msg[:500]}")

    title = (info.get("title") or info.get("description") or "video").strip()
    meta = {"id": did, "url": url, "title": title[:200], "uploader": info.get("uploader"),
            "duration": info.get("duration"), "file": video.name, "ext": video.suffix, "created_at": time.time()}
    (d / "info.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return meta


def load_download(did: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{32}", did) or not (DL_DIR / did / "info.json").exists():
        raise HTTPException(404, "Không tìm thấy video")
    return json.loads((DL_DIR / did / "info.json").read_text(encoding="utf-8"))


def download_name(meta: dict) -> str:
    return (re.sub(r'[\\/:*?"<>|\s]+', " ", meta["title"])[:80].strip(" .") or "video") + meta["ext"]


@app.get("/api/downloads/{did}/file")
def download_file(did: str):
    meta = load_download(did)
    return FileResponse(DL_DIR / did / meta["file"], filename=download_name(meta))


# ---------- Giọng đọc ----------
@app.get("/api/voices")
def voices():
    return [{"key": k, "label": f"{k} – {g}, {style}", "group": f"Miền {r}"}
            for k, (g, r, style) in pipeline.VOICES.items()]


PREVIEW_TEXT = "Xin chào, đây là giọng đọc tiếng Việt dùng để lồng tiếng cho video của bạn."


@app.get("/api/tts/preview")
def tts_preview(voice: str, rate: str = "+0%"):
    voice = pipeline.voice_key(voice)
    if not voice:
        raise HTTPException(400, "Giọng không hợp lệ")
    rate = pipeline.valid_rate(rate)
    d = DATA_DIR / "tts_preview"
    d.mkdir(exist_ok=True)
    out = d / f"vieneu-{voice}{rate.replace('%', 'pct')}.wav"
    if not out.exists():
        raw = d / f"vieneu-{voice}.wav"
        try:
            if not raw.exists():
                pipeline.tts_clip(PREVIEW_TEXT, voice, raw)
            tempo = pipeline.rate_factor(rate)
            pipeline._run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
                           *(["-af", f"atempo={tempo:.3f}"] if tempo != 1 else []), str(out)])
        except Exception as e:
            raise HTTPException(502, str(e)[:300])
    return FileResponse(out, media_type="audio/wav")


@app.get("/api/health")
def health():
    return {"ok": True, "translator": pipeline.TRANSLATOR, "whisper": pipeline.WHISPER_MODEL}


@app.get("/brand_logo.png")
def brand_logo():
    if not pipeline.BRAND_LOGO.exists():
        raise HTTPException(404, "Chưa có logo")
    return FileResponse(pipeline.BRAND_LOGO)


@app.get("/download")
def download_page():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/jobs/{jid}")
def job_page(jid: str):
    job_dir(jid)  # 404 nếu id sai
    return FileResponse(Path(__file__).parent / "static" / "index.html")


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
