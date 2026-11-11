# CÀI ĐẶT ỨNG DỤNG "LỜI VIỆT" – PHỤ ĐỀ TIẾNG VIỆT TỰ ĐỘNG CHO VIDEO

> **Dành cho Claude (AI agent: Claude Code / Claude Cowork) đang chạy trên máy của người dùng.**
> Hãy đọc toàn bộ file này, rồi thực hiện lần lượt các bước bên dưới để tạo mã nguồn, cài đặt môi trường, chạy và kiểm tra ứng dụng.
> Người dùng nói tiếng Việt: mọi trao đổi, câu hỏi và báo cáo đều viết bằng tiếng Việt.

## 0. Ứng dụng này là gì

Web app chạy local: người dùng upload video → Whisper nhận dạng giọng nói (tự phát hiện ngôn ngữ) → LLM (Claude hoặc Qwen qua Ollama) dịch sang tiếng Việt → người dùng chỉnh sửa phụ đề trên web → xuất file SRT, video in cứng phụ đề hoặc video nhúng phụ đề.

| Thành phần | Công nghệ |
|---|---|
| Backend | Python 3.11, FastAPI, chạy job nền bằng ThreadPoolExecutor |
| Nhận dạng giọng nói | faster-whisper (large-v3), VAD, timestamp cấp từ |
| Dịch | Claude API (mặc định) hoặc Ollama + Qwen (dữ liệu không ra ngoài) |
| Xử lý video | FFmpeg, font Noto Sans (đủ dấu tiếng Việt) |
| Frontend | 1 file HTML thuần, không cần build |
| Triển khai | Docker Compose, cổng 8000 |

## 1. Quy tắc bắt buộc cho agent

1. **Tạo file đúng nguyên văn** nội dung trong mục 6. Không tự ý sửa logic, đổi tên biến hay "cải tiến" code khi chưa được yêu cầu. Chỉ sửa khi có lỗi thực tế, và báo người dùng đã sửa gì.
2. **Hỏi người dùng trước khi** cài phần mềm hệ thống, chạy lệnh cần `sudo`/quyền Administrator, khởi động lại máy, hoặc tải dữ liệu lớn (image Docker CUDA khoảng 5–8 GB, model Whisper khoảng 3 GB).
3. **API key:** hỏi người dùng nhập `ANTHROPIC_API_KEY`, chỉ ghi vào file `.env`. Không in key ra màn hình, không ghi vào file khác, không commit lên git (tạo `.gitignore` như mục 6).
4. Không có quyền hoặc không làm được bước nào thì **dừng lại, giải thích ngắn gọn và hướng dẫn người dùng tự làm**, không đoán mò.
5. Kết thúc bằng một **báo cáo ngắn** theo mẫu ở mục 8.

## 2. Bước 1 – Kiểm tra môi trường

Chạy các lệnh phù hợp với hệ điều hành rồi tóm tắt kết quả cho người dùng.

| Kiểm tra | Lệnh (Linux/macOS/WSL) | Lệnh (Windows PowerShell) |
|---|---|---|
| Hệ điều hành | `uname -a`, `cat /etc/os-release` | `systeminfo \| findstr /B /C:"OS"` |
| Docker | `docker --version && docker compose version` | như bên trái |
| GPU NVIDIA | `nvidia-smi` | `nvidia-smi` |
| Docker dùng được GPU | `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` | như bên trái |
| Dung lượng trống | `df -h .` | `Get-PSDrive C` |

Yêu cầu: còn trống ít nhất **20 GB**; GPU NVIDIA **8 GB VRAM** trở lên nếu chạy chế độ GPU.

Chọn chế độ cài đặt:

| Tình huống | Chế độ |
|---|---|
| Có GPU NVIDIA và lệnh kiểm tra Docker GPU chạy được | **A – GPU** (khuyến nghị) |
| Có GPU NVIDIA nhưng Docker chưa nhận GPU | Cài thêm theo mục 3, rồi dùng **A** |
| Không có GPU NVIDIA (máy Mac, laptop văn phòng, card AMD/Intel) | **B – CPU** (chậm hơn nhiều, báo trước cho người dùng) |

Báo người dùng chế độ đã chọn và lý do trước khi tiếp tục.

## 3. Bước 2 – Cài phần mềm còn thiếu (hỏi người dùng trước)

**Ubuntu / Debian**
```bash
# Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER      # cần đăng xuất/đăng nhập lại để có hiệu lực

# NVIDIA Container Toolkit (chỉ khi có GPU NVIDIA, driver đã cài và nvidia-smi chạy được)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
Nếu `nvidia-smi` không chạy được trên Ubuntu: driver NVIDIA chưa cài. Đề xuất `sudo ubuntu-drivers autoinstall` rồi khởi động lại máy (hỏi người dùng trước).

**Windows 10/11**
- Cài/cập nhật driver NVIDIA mới nhất từ trang của NVIDIA.
- Cài Docker Desktop, bật "Use the WSL 2 based engine". Docker Desktop hỗ trợ GPU qua WSL2, không cần cài NVIDIA Container Toolkit riêng.
- Agent thường không tự cài được Docker Desktop: hướng dẫn người dùng tải từ docker.com, cài xong mở Docker Desktop rồi báo lại.
- Chạy các lệnh của tài liệu này trong PowerShell hoặc terminal WSL. Với PowerShell, thay `cp` bằng `Copy-Item`, `mkdir -p` bằng `New-Item -ItemType Directory -Force`.

**macOS**
- Không có GPU NVIDIA → dùng chế độ **B – CPU**. Cài Docker Desktop for Mac.
- Lưu ý: Docker trên Mac không dùng được GPU Apple. Chế độ CPU chạy được nhưng chậm.

Sau khi cài, chạy lại các lệnh kiểm tra ở mục 2.

## 4. Bước 3 – Tạo mã nguồn

1. Hỏi người dùng muốn đặt dự án ở đâu (mặc định: `~/video-subtitle-vi` hoặc `C:\Projects\video-subtitle-vi`).
2. Tạo thư mục, rồi tạo **từng file trong mục 6 đúng đường dẫn và đúng nguyên văn nội dung** (nội dung nằm giữa các dòng ````` ` `````).
3. Chế độ **B – CPU**: tạo thêm `Dockerfile.cpu` và `docker-compose.cpu.yml` ở mục 6.2.
4. Kiểm tra cấu trúc thư mục:

```
video-subtitle-vi/
├── .env.example
├── .gitignore
├── Dockerfile
├── Dockerfile.cpu            (chỉ chế độ B)
├── docker-compose.yml
├── docker-compose.cpu.yml    (chỉ chế độ B)
├── requirements.txt
└── app/
    ├── __init__.py
    ├── main.py
    ├── pipeline.py
    └── static/
        └── index.html
```

5. Kiểm tra cú pháp Python nếu máy có python3: `python3 -m py_compile app/main.py app/pipeline.py`.

## 5. Bước 4 – Cấu hình, chạy và kiểm tra

**Cấu hình**
```bash
cp .env.example .env
mkdir -p data models
```
Hỏi người dùng:
1. **Dịch bằng gì?** Claude (chất lượng tốt nhất, cần API key, có phí theo lượng dùng) hay Ollama + Qwen (dữ liệu không ra ngoài, cần GPU khoảng 24 GB VRAM nếu chạy chung Whisper).
2. Nếu Claude: xin `ANTHROPIC_API_KEY` (lấy tại console.anthropic.com) và ghi vào `.env`.

Chỉnh `.env` theo chế độ:

| Biến | Chế độ A – GPU | Chế độ B – CPU |
|---|---|---|
| `WHISPER_DEVICE` | `cuda` | `cpu` |
| `WHISPER_MODEL` | `large-v3` (GPU < 8 GB: `medium`) | `small` |
| `WHISPER_COMPUTE` | để mặc định | `int8` |
| `TRANSLATOR` | `claude` hoặc `ollama` | `claude` |

**Chạy**
```bash
# Chế độ A – GPU
docker compose up -d --build

# Chế độ A + dịch nội bộ bằng Ollama
docker compose --profile local-llm up -d --build
docker compose exec ollama ollama pull qwen2.5:14b

# Chế độ B – CPU
docker compose -f docker-compose.cpu.yml up -d --build
```
Lần build đầu mất 5–20 phút tùy mạng. Theo dõi: `docker compose logs -f app` (chế độ B thêm `-f docker-compose.cpu.yml`).

**Kiểm tra**
1. `curl http://localhost:8000/api/health` phải trả về `{"ok":true,...}`.
2. Tạo video thử 10 giây có giọng nói. Nếu người dùng không có sẵn video, hỏi họ cung cấp một video ngắn có lời nói tiếng Anh. (Video sinh bằng FFmpeg chỉ có tiếng bíp, không kiểm tra được phần nhận dạng.)
3. Gửi video thử qua API và theo dõi đến khi `status` là `ready`:
```bash
curl -s -F "file=@test.mp4" http://localhost:8000/api/jobs
# trả về {"id":"<JOB_ID>"}
curl -s http://localhost:8000/api/jobs/<JOB_ID>
```
   Lần đầu sẽ chậm vì tải model Whisper (khoảng 3 GB) vào thư mục `models/`.
4. Khi `ready`: tải `http://localhost:8000/api/jobs/<JOB_ID>/files/srt` và kiểm tra phụ đề tiếng Việt có dấu, đúng nghĩa.
5. Mời người dùng mở **http://localhost:8000** trên trình duyệt để dùng giao diện.

**Xử lý lỗi thường gặp**

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `could not select device driver "nvidia"` | Chưa cài/cấu hình NVIDIA Container Toolkit | Làm lại phần NVIDIA ở mục 3, `sudo systemctl restart docker` |
| `libcudnn` / `cudnn` not found | Sai phiên bản CUDA/cuDNN | Giữ base image `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`; cập nhật driver NVIDIA |
| `CUDA out of memory` | GPU không đủ VRAM | `WHISPER_MODEL=medium` hoặc `small`, hoặc tắt Ollama |
| Job lỗi khi dịch, log có `authentication` / `401` | API key sai hoặc hết hạn mức | Kiểm tra `ANTHROPIC_API_KEY` trong `.env`, rồi `docker compose up -d` lại |
| Job lỗi `model not found` khi dùng Claude | Tên model không còn hợp lệ | Hỏi người dùng tên model Claude hiện có trong tài khoản, sửa `CLAUDE_MODEL` |
| Không tải được model Whisper | Mạng chặn huggingface.co | Báo người dùng mở mạng hoặc dùng proxy |
| `permission denied ... docker.sock` | User chưa thuộc nhóm docker | `sudo usermod -aG docker $USER`, đăng xuất rồi đăng nhập lại |
| Cổng 8000 đã bị dùng | Ứng dụng khác chiếm cổng | Đổi `"8000:8000"` thành `"8080:8000"` trong compose |
| Phụ đề hiện ô vuông thay vì chữ có dấu | Thiếu font | Kiểm tra image đã cài `fonts-noto-core` |
| Xuất video nhúng phụ đề lỗi với file AVI | Codec gốc không tương thích MP4 | Dùng chế độ in cứng |

Sửa xong thì chạy lại bước kiểm tra. Lỗi lạ không có trong bảng: đọc `docker compose logs app`, tóm tắt nguyên nhân cho người dùng trước khi sửa.

## 6. Mã nguồn

### 6.1 File bắt buộc

**`.gitignore`**

`````text
.env
data/
models/
ollama/
__pycache__/
*.pyc
`````

**`requirements.txt`**

`````text
fastapi==0.115.6
uvicorn[standard]==0.32.1
python-multipart==0.0.19
faster-whisper==1.1.0
anthropic>=0.40.0
requests==2.32.3
pydantic>=2.9
`````

**`Dockerfile`**

`````dockerfile
# GPU NVIDIA (khuyến nghị). Chạy CPU: đổi base sang python:3.11-slim và đặt WHISPER_DEVICE=cpu
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    DATA_DIR=/data HF_HOME=/models

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip ffmpeg fonts-noto-core fontconfig \
 && fc-cache -f && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY app ./app

EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
`````

**`docker-compose.yml`**

`````yaml
services:
  app:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    volumes:
      - ./data:/data        # video, phụ đề, kết quả
      - ./models:/models    # cache model Whisper (~3 GB với large-v3)
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  # Chỉ bật khi TRANSLATOR=ollama (dịch nội bộ, dữ liệu không ra ngoài)
  # docker compose --profile local-llm up -d
  ollama:
    image: ollama/ollama:latest
    profiles: ["local-llm"]
    volumes:
      - ./ollama:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
`````

**`.env.example`**

`````bash
# ---- Nhận dạng giọng nói ----
WHISPER_MODEL=large-v3        # large-v3 | medium | small (nhỏ hơn = nhanh hơn, kém chính xác hơn)
WHISPER_DEVICE=cuda           # cuda | cpu
# WHISPER_COMPUTE=float16     # GPU: float16 | CPU: int8

# ---- Dịch ----
TRANSLATOR=claude             # claude | ollama
ANTHROPIC_API_KEY=sk-ant-xxx
CLAUDE_MODEL=claude-haiku-5-5 # đổi sang claude-sonnet-5-5 nếu cần chất lượng cao hơn
OLLAMA_URL=http://ollama:11434
OLLAMA_MODEL=qwen2.5:14b
TRANSLATE_BATCH=30

# ---- Hệ thống ----
SUB_FONT=Noto Sans
MAX_UPLOAD_MB=2048
WORKERS=1
`````

**`app/__init__.py`** (file rỗng)

`````python

`````

**`app/pipeline.py`**

`````python
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

TRANSLATOR = os.getenv("TRANSLATOR", "claude")  # claude | ollama
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
        vad_filter=True,            # bỏ đoạn im lặng / nhạc
        word_timestamps=True,       # cần để ngắt phụ đề chính xác
        beam_size=5,
    )
    words = []
    for seg in segments:  # generator: xử lý dần trong lúc nhận dạng
        if seg.words:
            words += [{"start": w.start, "end": w.end, "text": w.word} for w in seg.words]
        else:
            words.append({"start": seg.start, "end": seg.end, "text": seg.text})
        if on_progress and duration:
            on_progress(min(seg.end / duration, 1.0))
    return words, info.language, float(info.language_probability)


# ---------- 3. Phân đoạn phụ đề ----------
_SENTENCE_END = re.compile(r"[.!?。！？…]$")


def build_segments(words, lang):
    """Gộp từ thành các đoạn phụ đề dễ đọc: theo câu, khoảng lặng, độ dài, thời lượng."""
    char_limit = 32 if lang in ("zh", "ja") else 84
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
        if cur and w["start"] - cur[-1]["end"] > MAX_GAP_SECONDS:
            flush()
        cur.append(w)
        text = "".join(x["text"] for x in cur).strip()
        dur = cur[-1]["end"] - cur[0]["start"]
        sentence_done = bool(_SENTENCE_END.search(w["text"].strip())) and len(text) >= 12
        if sentence_done or dur >= MAX_SEG_SECONDS or len(text) >= char_limit:
            flush()
    flush()

    # Đoạn quá ngắn thì kéo dài để kịp đọc, không đè lên đoạn sau
    for i, s in enumerate(segs):
        if s["end"] - s["start"] < MIN_SEG_SECONDS:
            limit = segs[i + 1]["start"] - 0.05 if i + 1 < len(segs) else s["start"] + MIN_SEG_SECONDS
            s["end"] = round(max(s["end"], min(s["start"] + MIN_SEG_SECONDS, limit)), 3)
    return segs


# ---------- 4. Dịch ----------
SYSTEM_PROMPT = """Bạn là biên dịch viên phụ đề video chuyên nghiệp. Dịch từng dòng phụ đề sang tiếng Việt.
Quy tắc:
- Dịch tự nhiên theo văn nói của người Việt, không dịch word-by-word.
- Ngắn gọn để người xem đọc kịp, tối đa khoảng 80 ký tự mỗi dòng.
- Dựa vào ngữ cảnh để chọn xưng hô (anh/em, tôi/bạn, ông/cháu...) và giữ nhất quán.
- Giữ nguyên tên riêng, tên thương hiệu, thuật ngữ kỹ thuật nếu không có bản dịch phổ biến.
- Tuân thủ bảng thuật ngữ nếu được cung cấp.
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


def _parse_translation(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip()
    data = json.loads(text[text.find("{"): text.rfind("}") + 1])
    return {int(x["id"]): str(x["vi"]).strip() for x in data.get("items", [])}


def translate_segments(segs, src_lang, glossary="", on_progress=None):
    call = _call_ollama if TRANSLATOR == "ollama" else _call_claude
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
                     + json.dumps([{"id": s["id"], "text": s["src"]} for s in chunk], ensure_ascii=False))
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
def render_video(job_dir: Path, video_name: str, mode="hard", font_size=20) -> str:
    """mode=hard: in cứng phụ đề (encode lại). mode=soft: nhúng track phụ đề (nhanh, bật/tắt được)."""
    out = f"output_{mode}.mp4"
    if mode == "hard":
        style = (f"FontName={SUB_FONT},FontSize={int(font_size)},PrimaryColour=&H00FFFFFF,"
                 "OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0,MarginV=28")
        _run(["ffmpeg", "-y", "-i", video_name, "-vf", f"subtitles=vi.srt:force_style='{style}'",
              "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
              "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out], cwd=job_dir)
    else:
        _run(["ffmpeg", "-y", "-i", video_name, "-i", "vi.srt",
              "-map", "0:v", "-map", "0:a?", "-map", "1:0",
              "-c:v", "copy", "-c:a", "copy", "-c:s", "mov_text",
              "-metadata:s:s:0", "language=vie", "-disposition:s:0", "default",
              "-movflags", "+faststart", out], cwd=job_dir)
    return out
`````

**`app/main.py`**

`````python
import json
import logging
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

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

        update_job(jid, status="translating", step="Đang dịch sang tiếng Việt", progress=0.6,
                   detected_lang=lang, lang_prob=round(prob, 3), segments=segs)
        if lang == "vi":
            for s in segs:
                s["vi"] = s["src"]
        else:
            pipeline.translate_segments(segs, lang, job.get("glossary", ""),
                                        on_progress=throttled_progress(jid, 0.6, 0.38))

        write_subs(jid, segs)
        (d / "audio.wav").unlink(missing_ok=True)
        update_job(jid, status="ready", step="Phụ đề đã sẵn sàng", progress=1.0, segments=segs)
    except Exception as e:
        log.exception("Job %s lỗi", jid)
        update_job(jid, status="error", step="Xử lý thất bại", error=str(e)[:2000])


def render_job(jid: str, mode: str, font_size: int):
    try:
        job = load_job(jid)
        out = pipeline.render_video(job_dir(jid), job["video"], mode, font_size)
        outputs = {**job.get("outputs", {}), mode: out}
        update_job(jid, render_status="done", render_error=None, outputs=outputs)
    except Exception as e:
        log.exception("Render %s lỗi", jid)
        update_job(jid, render_status="error", render_error=str(e)[:2000])


# ---------- API ----------
@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), source_lang: str = Form(""), glossary: str = Form("")):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Định dạng không hỗ trợ. Dùng: {', '.join(sorted(ALLOWED_EXT))}")

    jid = uuid.uuid4().hex
    d = JOBS_DIR / jid
    d.mkdir(parents=True)
    video_name = f"input{ext}"
    size, limit = 0, MAX_UPLOAD_MB * 1024 * 1024
    with open(d / video_name, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                f.close()
                for p in d.iterdir():
                    p.unlink()
                d.rmdir()
                raise HTTPException(413, f"File vượt quá {MAX_UPLOAD_MB} MB")
            f.write(chunk)

    job = {"id": jid, "filename": file.filename, "video": video_name, "status": "queued",
           "step": "Đang chờ xử lý", "progress": 0.0, "source_lang": source_lang.strip(),
           "glossary": glossary[:5000], "segments": [], "outputs": {}, "render_status": None,
           "created_at": time.time(), "updated_at": time.time()}
    save_job(job)
    executor.submit(process_job, jid)
    return {"id": jid}


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    return load_job(jid)


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
    font_size: int = 20


@app.post("/api/jobs/{jid}/render")
def render(jid: str, body: RenderRequest):
    job = load_job(jid)
    if job["status"] != "ready":
        raise HTTPException(409, "Phụ đề chưa sẵn sàng")
    if job.get("render_status") == "rendering":
        raise HTTPException(409, "Video đang được xuất")
    if body.mode not in ("hard", "soft"):
        raise HTTPException(400, "mode phải là hard hoặc soft")
    update_job(jid, render_status="rendering", render_mode=body.mode, render_error=None)
    executor.submit(render_job, jid, body.mode, max(12, min(body.font_size, 48)))
    return {"ok": True}


@app.get("/api/jobs/{jid}/files/{kind}")
def download(jid: str, kind: str):
    job = load_job(jid)
    d = job_dir(jid)
    stem = Path(job["filename"]).stem or "video"
    files = {
        "video": (job["video"], None),
        "srt": ("vi.srt", f"{stem}.vi.srt"),
        "vtt": ("vi.vtt", None),
        "hard": (job.get("outputs", {}).get("hard"), f"{stem}.vi.hardsub.mp4"),
        "soft": (job.get("outputs", {}).get("soft"), f"{stem}.vi.softsub.mp4"),
    }
    name, download_name = files.get(kind, (None, None))
    if not name or not (d / name).exists():
        raise HTTPException(404, "File chưa có")
    return FileResponse(d / name, filename=download_name)


@app.get("/api/health")
def health():
    return {"ok": True, "translator": pipeline.TRANSLATOR, "whisper": pipeline.WHISPER_MODEL}


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
`````

**`app/static/index.html`**

`````html
<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lời Việt – Phụ đề tiếng Việt tự động</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
  color-scheme: light;
  --bg: #EEF1F6; --surface: #FFFFFF; --ink: #1C2740; --muted: #5B6680;
  --line: #D5DBE6; --accent: #0E7A73; --accent-ink: #FFFFFF;
  --cue: #FFE27A; --cue-ink: #1C2740; --danger: #B3261E;
  --radius-lg: 14px; --radius-sm: 8px;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --bg: #141A26; --surface: #1C2433; --ink: #E6EAF2; --muted: #9AA4B8;
    --line: #2E384A; --accent: #3CC2B6; --accent-ink: #0B1F1D;
    --cue: #4A3F12; --cue-ink: #FFE27A; --danger: #FF8A80;
  }
}
* { box-sizing: border-box; }
html, body { margin: 0; }
body {
  background: var(--bg); color: var(--ink);
  font: 400 15px/1.55 "Be Vietnam Pro", "Segoe UI", system-ui, sans-serif;
}
header {
  display: flex; align-items: baseline; gap: 14px;
  padding: 18px clamp(16px, 4vw, 40px); border-bottom: 1px solid var(--line);
}
header h1 { margin: 0; font-size: 20px; font-weight: 700; letter-spacing: -0.01em; }
header p { margin: 0; color: var(--muted); font-size: 14px; }
main { padding: clamp(16px, 4vw, 40px); max-width: 1320px; margin: 0 auto; }
button, select, textarea, input { font: inherit; color: inherit; }
:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; }

/* Upload */
.upload { max-width: 640px; }
.upload h2 { font-size: 30px; line-height: 1.2; margin: 8px 0 10px; letter-spacing: -0.02em; }
.upload > p { color: var(--muted); margin: 0 0 28px; max-width: 54ch; }
.drop {
  display: grid; place-items: center; text-align: center; gap: 6px;
  min-height: 190px; padding: 24px; cursor: pointer;
  border: 2px dashed var(--line); border-radius: var(--radius-lg); background: var(--surface);
}
.drop.over, .drop:hover { border-color: var(--accent); }
.drop strong { font-size: 17px; }
.drop span { color: var(--muted); font-size: 14px; }
.field { margin-top: 18px; display: grid; gap: 6px; }
.field label { font-weight: 600; font-size: 14px; }
.field small { color: var(--muted); }
select, textarea {
  width: 100%; padding: 10px 12px; background: var(--surface);
  border: 1px solid var(--line); border-radius: var(--radius-sm);
}
textarea { resize: vertical; }
.btn {
  display: inline-flex; align-items: center; justify-content: center; gap: 8px;
  padding: 10px 18px; border-radius: var(--radius-sm); border: 1px solid var(--line);
  background: var(--surface); cursor: pointer; font-weight: 600; text-decoration: none;
}
.btn.primary { background: var(--accent); border-color: var(--accent); color: var(--accent-ink); }
.btn:disabled { opacity: .5; cursor: not-allowed; }
.upload .btn.primary { margin-top: 22px; width: 100%; padding: 13px; }

/* Progress */
.progress-wrap { max-width: 640px; }
.progress-wrap h2 { font-size: 22px; margin: 0 0 4px; }
.bar { height: 10px; background: var(--line); border-radius: 99px; overflow: hidden; margin: 18px 0 8px; }
.bar > div { height: 100%; width: 0; background: var(--accent); transition: width .4s ease; }
.muted { color: var(--muted); }
.error { color: var(--danger); white-space: pre-wrap; }

/* Editor */
.editor { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr); gap: 28px; align-items: start; }
.stage { position: sticky; top: 16px; }
.screen { position: relative; background: #000; border-radius: var(--radius-lg); overflow: hidden; }
.screen video { display: block; width: 100%; max-height: 62vh; background: #000; }
.caption {
  position: absolute; left: 50%; bottom: 7%; transform: translateX(-50%);
  max-width: 88%; text-align: center; white-space: pre-line; pointer-events: none;
  font-weight: 600; font-size: clamp(15px, 2.3vw, 24px); line-height: 1.35; color: #fff;
  text-shadow: 0 0 4px #000, 0 0 4px #000, 0 2px 2px #000;
}
.meta { display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 14px 0; font-size: 14px; color: var(--muted); }
.meta b { color: var(--ink); font-weight: 600; }
.actions { display: flex; flex-wrap: wrap; gap: 10px; }
.export { margin-top: 18px; padding: 16px; background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius-lg); }
.export h3 { margin: 0 0 4px; font-size: 16px; }
.export p { margin: 0 0 12px; font-size: 14px; }
.export .row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
.export select { width: auto; }

.list-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 10px; }
.list-head h2 { margin: 0; font-size: 18px; }
.cues { display: grid; gap: 8px; max-height: calc(100vh - 160px); overflow-y: auto; padding-right: 4px; }
.cue {
  display: grid; grid-template-columns: 74px 1fr; gap: 4px 12px;
  padding: 10px 12px; background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius-sm);
}
.cue.active { background: var(--cue); border-color: transparent; }
.cue.active .src, .cue.active .time { color: var(--cue-ink); }
.time {
  grid-row: span 2; background: none; border: 0; padding: 2px 0; text-align: left;
  color: var(--accent); font-weight: 600; font-size: 13px; cursor: pointer; font-variant-numeric: tabular-nums;
}
.src { margin: 0; font-size: 13px; color: var(--muted); }
.cue textarea {
  border: 1px solid transparent; background: transparent; padding: 4px 6px; margin-left: -6px;
  font-weight: 500; min-height: 34px; resize: vertical;
}
.cue textarea:hover { border-color: var(--line); }
.cue textarea:focus { border-color: var(--accent); background: var(--surface); color: var(--ink); }
.cue.dirty { box-shadow: inset 3px 0 0 var(--accent); }
.hidden { display: none !important; }

@media (max-width: 900px) {
  .editor { grid-template-columns: 1fr; }
  .stage { position: static; }
  .cues { max-height: none; }
}
@media (prefers-reduced-motion: reduce) { .bar > div { transition: none; } }
</style>
</head>
<body>
<header>
  <h1>Lời Việt</h1>
  <p>Nghe giọng nói trong video và tạo phụ đề tiếng Việt</p>
</header>

<main>
  <!-- Bước 1: Upload -->
  <section id="view-upload" class="upload">
    <h2>Tạo phụ đề tiếng Việt cho video của bạn</h2>
    <p>Tải video lên, hệ thống sẽ nhận dạng lời nói, dịch sang tiếng Việt và cho bạn chỉnh sửa trước khi xuất video.</p>

    <label id="drop" class="drop" tabindex="0">
      <input id="file" type="file" accept="video/*,.mkv" hidden>
      <strong id="drop-title">Kéo video vào đây hoặc bấm để chọn</strong>
      <span id="drop-sub">MP4, MOV, MKV, WEBM, AVI</span>
    </label>

    <div class="field">
      <label for="lang">Ngôn ngữ trong video</label>
      <select id="lang">
        <option value="">Tự nhận diện</option>
        <option value="en">Tiếng Anh</option>
        <option value="zh">Tiếng Trung</option>
        <option value="ja">Tiếng Nhật</option>
        <option value="ko">Tiếng Hàn</option>
        <option value="th">Tiếng Thái</option>
        <option value="fr">Tiếng Pháp</option>
        <option value="de">Tiếng Đức</option>
        <option value="es">Tiếng Tây Ban Nha</option>
        <option value="ru">Tiếng Nga</option>
        <option value="id">Tiếng Indonesia</option>
      </select>
    </div>

    <div class="field">
      <label for="glossary">Thuật ngữ cần giữ đúng (không bắt buộc)</label>
      <textarea id="glossary" rows="3" placeholder="Mỗi dòng một cặp, ví dụ:&#10;checkout = thanh toán&#10;Magento = Magento"></textarea>
      <small>Dùng cho tên thương hiệu, tên người, thuật ngữ chuyên ngành.</small>
    </div>

    <button id="start" class="btn primary" disabled>Tạo phụ đề</button>
    <p id="upload-error" class="error"></p>
  </section>

  <!-- Bước 2: Tiến trình -->
  <section id="view-progress" class="progress-wrap hidden">
    <h2 id="p-file">Đang xử lý</h2>
    <p id="p-step" class="muted">Đang tải lên</p>
    <div class="bar"><div id="p-bar"></div></div>
    <p id="p-pct" class="muted">0%</p>
    <p id="p-error" class="error"></p>
    <button id="p-new" class="btn hidden">Thử video khác</button>
  </section>

  <!-- Bước 3: Chỉnh sửa và xuất -->
  <section id="view-editor" class="editor hidden">
    <div class="stage">
      <div class="screen">
        <video id="video" controls preload="metadata" playsinline></video>
        <div id="caption" class="caption"></div>
      </div>
      <div class="meta">
        <span>Video: <b id="m-file"></b></span>
        <span>Ngôn ngữ gốc: <b id="m-lang"></b></span>
        <span><b id="m-count"></b> đoạn phụ đề</span>
      </div>
      <div class="actions">
        <a id="dl-srt" class="btn" href="#">Tải file SRT</a>
        <button id="new-job" class="btn">Video mới</button>
      </div>
      <div class="export">
        <h3>Xuất video có phụ đề</h3>
        <p class="muted">In cứng: phụ đề luôn hiển thị, hợp với Facebook, TikTok. Nhúng: nhanh hơn, người xem bật tắt được.</p>
        <div class="row">
          <select id="font-size" aria-label="Cỡ chữ phụ đề">
            <option value="16">Chữ nhỏ</option>
            <option value="20" selected>Chữ vừa</option>
            <option value="26">Chữ lớn</option>
          </select>
          <button id="render-hard" class="btn primary">Xuất video in cứng</button>
          <button id="render-soft" class="btn">Xuất video nhúng phụ đề</button>
        </div>
        <p id="render-status" class="muted" style="margin:12px 0 0"></p>
        <div class="row" id="render-links" style="margin-top:8px"></div>
      </div>
    </div>

    <div>
      <div class="list-head">
        <h2>Phụ đề tiếng Việt</h2>
        <button id="save" class="btn primary" disabled>Lưu thay đổi</button>
      </div>
      <div id="cues" class="cues"></div>
    </div>
  </section>
</main>

<script>
const $ = (id) => document.getElementById(id);
const LANGS = { en: "Tiếng Anh", zh: "Tiếng Trung", ja: "Tiếng Nhật", ko: "Tiếng Hàn", th: "Tiếng Thái",
  fr: "Tiếng Pháp", de: "Tiếng Đức", es: "Tiếng Tây Ban Nha", ru: "Tiếng Nga", id: "Tiếng Indonesia", vi: "Tiếng Việt" };
let jobId = null, job = null, pollTimer = null, selectedFile = null;
const dirty = new Map();

function show(view) {
  for (const v of ["upload", "progress", "editor"]) $("view-" + v).classList.toggle("hidden", v !== view);
}
function fmt(sec) {
  const m = Math.floor(sec / 60), s = Math.floor(sec % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

// ---------- Upload ----------
const drop = $("drop");
function pickFile(f) {
  if (!f) return;
  selectedFile = f;
  $("drop-title").textContent = f.name;
  $("drop-sub").textContent = `${(f.size / 1048576).toFixed(1)} MB – bấm để đổi video`;
  $("start").disabled = false;
}
$("file").addEventListener("change", (e) => pickFile(e.target.files[0]));
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); } });
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => pickFile(e.dataTransfer.files[0]));

$("start").addEventListener("click", () => {
  if (!selectedFile) return;
  $("upload-error").textContent = "";
  const fd = new FormData();
  fd.append("file", selectedFile);
  fd.append("source_lang", $("lang").value);
  fd.append("glossary", $("glossary").value);

  show("progress");
  $("p-file").textContent = selectedFile.name;
  $("p-step").textContent = "Đang tải video lên";
  $("p-error").textContent = "";
  $("p-new").classList.add("hidden");

  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/jobs");
  xhr.upload.onprogress = (e) => { if (e.lengthComputable) setBar(e.loaded / e.total * 0.02); };
  xhr.onload = () => {
    if (xhr.status >= 200 && xhr.status < 300) {
      jobId = JSON.parse(xhr.responseText).id;
      location.hash = jobId;
      poll();
    } else {
      show("upload");
      $("upload-error").textContent = (safeJson(xhr.responseText).detail) || "Tải lên thất bại. Kiểm tra kết nối rồi thử lại.";
    }
  };
  xhr.onerror = () => { show("upload"); $("upload-error").textContent = "Không kết nối được máy chủ."; };
  xhr.send(fd);
});
function safeJson(t) { try { return JSON.parse(t); } catch { return {}; } }
function setBar(p) {
  $("p-bar").style.width = `${Math.round(p * 100)}%`;
  $("p-pct").textContent = `${Math.round(p * 100)}%`;
}

// ---------- Poll ----------
async function poll() {
  clearTimeout(pollTimer);
  try {
    const r = await fetch(`/api/jobs/${jobId}`);
    if (!r.ok) { location.hash = ""; return reset(); }
    job = await r.json();
  } catch { pollTimer = setTimeout(poll, 3000); return; }

  if (job.status === "ready") {
    if ($("view-editor").classList.contains("hidden")) openEditor();
    updateRender();
    if (job.render_status === "rendering") pollTimer = setTimeout(poll, 2000);
    return;
  }
  show("progress");
  $("p-file").textContent = job.filename;
  $("p-step").textContent = job.step;
  setBar(job.progress || 0);
  if (job.status === "error") {
    $("p-error").textContent = job.error;
    $("p-new").classList.remove("hidden");
    return;
  }
  pollTimer = setTimeout(poll, 1500);
}

// ---------- Editor ----------
function openEditor() {
  show("editor");
  dirty.clear();
  $("save").disabled = true;
  const v = $("video");
  v.src = `/api/jobs/${jobId}/files/video`;
  $("m-file").textContent = job.filename;
  $("m-lang").textContent = LANGS[job.detected_lang] || job.detected_lang;
  $("m-count").textContent = job.segments.length;
  $("dl-srt").href = `/api/jobs/${jobId}/files/srt`;
  renderCues();
}

function renderCues() {
  const box = $("cues");
  box.innerHTML = "";
  for (const s of job.segments) {
    const row = document.createElement("div");
    row.className = "cue";
    row.dataset.id = s.id;

    const t = document.createElement("button");
    t.className = "time";
    t.textContent = fmt(s.start);
    t.title = "Phát từ đoạn này";
    t.addEventListener("click", () => { $("video").currentTime = s.start + 0.01; $("video").play(); });

    const src = document.createElement("p");
    src.className = "src";
    src.textContent = s.src;

    const ta = document.createElement("textarea");
    ta.rows = 1;
    ta.value = s.vi;
    ta.setAttribute("aria-label", `Phụ đề tại ${fmt(s.start)}`);
    ta.addEventListener("input", () => {
      s.vi = ta.value;
      dirty.set(s.id, ta.value);
      row.classList.add("dirty");
      $("save").disabled = false;
      updateCaption(true);
    });

    row.append(t, src, ta);
    box.append(row);
  }
}

let activeId = null;
function updateCaption(force) {
  const now = $("video").currentTime;
  const s = job.segments.find((x) => now >= x.start && now <= x.end);
  const id = s ? s.id : null;
  if (id === activeId && !force) return;
  activeId = id;
  $("caption").textContent = s ? s.vi : "";
  document.querySelectorAll(".cue.active").forEach((el) => el.classList.remove("active"));
  if (s) {
    const el = document.querySelector(`.cue[data-id="${s.id}"]`);
    if (el) {
      el.classList.add("active");
      if (!$("video").paused && !el.contains(document.activeElement)) el.scrollIntoView({ block: "nearest" });
    }
  }
}
$("video").addEventListener("timeupdate", () => updateCaption(false));
$("video").addEventListener("seeked", () => updateCaption(true));

$("save").addEventListener("click", async () => {
  $("save").disabled = true;
  $("save").textContent = "Đang lưu";
  const segments = [...dirty].map(([id, vi]) => ({ id, vi }));
  const r = await fetch(`/api/jobs/${jobId}/segments`, {
    method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ segments }),
  });
  $("save").textContent = "Lưu thay đổi";
  if (r.ok) {
    dirty.clear();
    document.querySelectorAll(".cue.dirty").forEach((el) => el.classList.remove("dirty"));
    job.outputs = {};
    updateRender();
  } else {
    $("save").disabled = false;
    alert("Lưu thất bại: " + (safeJson(await r.text()).detail || r.status));
  }
});

// ---------- Render ----------
async function startRender(mode) {
  if (dirty.size && !confirm("Có thay đổi chưa lưu. Lưu trước rồi mới xuất để video dùng bản mới nhất. Vẫn xuất?")) return;
  const r = await fetch(`/api/jobs/${jobId}/render`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode, font_size: Number($("font-size").value) }),
  });
  if (!r.ok) return alert(safeJson(await r.text()).detail || "Không xuất được video");
  job.render_status = "rendering";
  updateRender();
  poll();
}
$("render-hard").addEventListener("click", () => startRender("hard"));
$("render-soft").addEventListener("click", () => startRender("soft"));

function updateRender() {
  const busy = job.render_status === "rendering";
  $("render-hard").disabled = busy;
  $("render-soft").disabled = busy;
  const st = $("render-status");
  st.className = job.render_status === "error" ? "error" : "muted";
  st.textContent = busy ? "Đang xuất video, có thể mất vài phút với video dài."
    : job.render_status === "error" ? "Xuất video thất bại: " + job.render_error : "";
  const links = $("render-links");
  links.innerHTML = "";
  const outs = job.outputs || {};
  if (outs.hard) links.append(link("hard", "Tải video in cứng"));
  if (outs.soft) links.append(link("soft", "Tải video nhúng phụ đề"));
}
function link(kind, text) {
  const a = document.createElement("a");
  a.className = "btn";
  a.href = `/api/jobs/${jobId}/files/${kind}`;
  a.textContent = text;
  return a;
}

// ---------- Điều hướng ----------
function reset() {
  clearTimeout(pollTimer);
  jobId = null; job = null; selectedFile = null;
  history.replaceState(null, "", location.pathname);
  $("file").value = "";
  $("drop-title").textContent = "Kéo video vào đây hoặc bấm để chọn";
  $("drop-sub").textContent = "MP4, MOV, MKV, WEBM, AVI";
  $("start").disabled = true;
  $("video").removeAttribute("src");
  show("upload");
}
$("new-job").addEventListener("click", () => {
  if (dirty.size && !confirm("Bỏ các thay đổi chưa lưu?")) return;
  reset();
});
$("p-new").addEventListener("click", reset);
window.addEventListener("beforeunload", (e) => { if (dirty.size) e.preventDefault(); });

const h = location.hash.slice(1);
if (/^[0-9a-f]{32}$/.test(h)) { jobId = h; poll(); }
</script>
</body>
</html>
`````

### 6.2 Chỉ dùng cho chế độ B – CPU

**`Dockerfile.cpu`**

`````dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 DATA_DIR=/data HF_HOME=/models \
    WHISPER_DEVICE=cpu WHISPER_COMPUTE=int8

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg fonts-noto-core fontconfig \
 && fc-cache -f && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
`````

**`docker-compose.cpu.yml`**

`````yaml
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.cpu
    ports:
      - "8000:8000"
    env_file: .env
    volumes:
      - ./data:/data
      - ./models:/models
    restart: unless-stopped
`````


## 7. Lệnh vận hành thường dùng

| Việc | Lệnh |
|---|---|
| Dừng ứng dụng | `docker compose down` |
| Khởi động lại sau khi sửa `.env` | `docker compose up -d` |
| Cập nhật sau khi sửa code | `docker compose up -d --build` |
| Xem log | `docker compose logs -f app` |
| Dọn dữ liệu job cũ | xóa các thư mục trong `data/jobs/` |

## 8. Mẫu báo cáo cuối cùng cho người dùng

```
✅ Đã cài xong Lời Việt
- Chế độ: GPU (RTX ...) / CPU
- Dịch bằng: Claude (model ...) / Ollama (qwen2.5:14b)
- Thư mục dự án: ...
- Mở ứng dụng: http://localhost:8000
- Đã test: video ... giây, ngôn ngữ ..., ra ... đoạn phụ đề, thời gian xử lý ...
- Đã sửa so với tài liệu gốc: (không có / liệt kê)
- Lưu ý: ...
```
