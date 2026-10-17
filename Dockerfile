# ---- 1. build frontend React ----
FROM node:22-slim AS web
WORKDIR /web
COPY web/frontend/package.json web/frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/frontend ./
RUN npm run build

# ---- 2. runtime: API + worker (cùng image, khác lệnh chạy) ----
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg redis-server libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY docker/requirements-runtime.txt .
# PyTorch bản CPU: nhẹ hơn bản CUDA ~2 GB, đủ cho worker CPU
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-runtime.txt

COPY src src
COPY scripts scripts
COPY configs configs
COPY web/backend web/backend
COPY --from=web /web/dist web/frontend/dist
COPY docker/start.sh /start.sh
COPY docker/fetch_assets.py docker/fetch_assets.py
# HF Spaces chạy container bằng user uid 1000 (không phải root) -> các thư mục ghi được phải mở quyền
RUN chmod +x /start.sh && mkdir -p /data/clips /data/jobs /data/hf /weights && chmod -R 777 /data /weights

ENV PYTHONUNBUFFERED=1 \
    YOLO_AUTOINSTALL=False \
    CLIPS_DIR=/data/clips \
    JOBS_DIR=/data/jobs \
    WEIGHTS_DIR=/weights \
    HF_HOME=/data/hf \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    PORT=7860
EXPOSE 7860
# mặc định: 1 container chạy cả redis + worker + api (HF Spaces); docker-compose ghi đè lệnh cho từng service
CMD ["/start.sh"]
