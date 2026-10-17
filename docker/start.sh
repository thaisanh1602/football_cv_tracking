#!/bin/sh
# Chế độ 1 container (HF Spaces chỉ cho 1 container): tải assets -> redis + 1 RQ worker + API.
set -e
python /app/docker/fetch_assets.py
redis-server --daemonize yes --save "" --appendonly no --dir /tmp
cd /app/web/backend
rq worker clips --url redis://localhost:6379/0 &
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-7860}"
