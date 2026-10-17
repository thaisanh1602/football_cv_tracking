#!/bin/sh
# Chế độ 1 container (HF Spaces chỉ cho 1 container): redis + 1 RQ worker + API.
set -e
redis-server --daemonize yes --save "" --appendonly no
cd /app/web/backend
rq worker clips --url redis://localhost:6379/0 &
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-7860}"
