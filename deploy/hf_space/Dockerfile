# Dockerfile cho Hugging Face Space: Space chỉ chứa file này + README.md, code clone từ GitHub.
# Giữ đồng bộ với Dockerfile ở gốc repo (khác ở chỗ lấy source từ stage "src").
FROM python:3.11-slim AS src
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
ARG REPO=https://github.com/thaisanh1602/football_cv_tracking.git
ARG REF=main
# nội dung đổi khi có commit mới trên GitHub -> Docker không dùng lại layer clone cũ
ADD https://api.github.com/repos/thaisanh1602/football_cv_tracking/git/refs/heads/main /tmp/ref.json
RUN git clone --depth 1 --branch ${REF} ${REPO} /src

FROM node:22-slim AS web
WORKDIR /web
COPY --from=src /src/web/frontend ./
RUN npm ci --no-audit --no-fund && npm run build

FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg redis-server libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=src /src/docker/requirements-runtime.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-runtime.txt
COPY --from=src /src/src src
COPY --from=src /src/scripts scripts
COPY --from=src /src/configs configs
COPY --from=src /src/web/backend web/backend
COPY --from=src /src/docker/fetch_assets.py docker/fetch_assets.py
COPY --from=src /src/docker/start.sh /start.sh
COPY --from=web /web/dist web/frontend/dist
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
CMD ["/start.sh"]
