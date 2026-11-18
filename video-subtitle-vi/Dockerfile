# GPU NVIDIA (khuyến nghị). Chạy CPU: đổi base sang python:3.11-slim và đặt WHISPER_DEVICE=cpu
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    DATA_DIR=/data HF_HOME=/models TORCH_HOME=/models/torch

# gcc/g++/python3-dev: audio-separator cần build diffq
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev gcc g++ ffmpeg fonts-noto-core fontconfig libgl1 libglib2.0-0 \
 && fc-cache -f && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# PyTorch CUDA 12.6 cho BS-Roformer/Demucs chạy GPU (driver NVIDIA >= 525, RTX 40xx dùng tốt)
RUN pip3 install --no-cache-dir torch==2.14.1 torchaudio torchvision --index-url https://download.pytorch.org/whl/cu126
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY app ./app

EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
