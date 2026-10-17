"""API cho demo web: phục vụ video gốc + stream annotation JSON qua WebSocket. Trình duyệt tự vẽ overlay.

Mỗi clip trong CLIPS_DIR gồm <id>.mp4 (video gốc) và <id>.json (annotation, schema ở src/export/annotations.py).

    uvicorn app.main:app --reload --port 8000        (chạy trong web/backend)
"""
import asyncio
import json
import os
import re
import time
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import jobs

ROOT = Path(__file__).resolve().parents[3]
CLIPS_DIR = Path(os.getenv("CLIPS_DIR", ROOT / "outputs" / "web_clips"))
FRONTEND_DIST = Path(os.getenv("FRONTEND_DIST", ROOT / "web" / "frontend" / "dist"))
MAX_RANGE = 500  # số frame tối đa mỗi lần xin, tránh client gửi khoảng khổng lồ

app = FastAPI(title="football-cv-tracking demo")


def _clip_path(clip_id: str, ext: str) -> Path:
    """Clip mẫu: CLIPS_DIR/<id>.<ext>. Clip upload: id "job-<hex>" -> JOBS_DIR/<hex>/clip.<ext>.
    Chỉ nhận tên đơn giản để chặn path traversal (../)."""
    if m := re.fullmatch(r"job-([0-9a-f]{32})", clip_id):
        p = jobs.JOBS_DIR / m.group(1) / f"clip.{ext}"
    elif re.fullmatch(r"[A-Za-z0-9_-]+", clip_id):
        p = CLIPS_DIR / f"{clip_id}.{ext}"
    else:
        raise HTTPException(404, "clip không tồn tại")
    if not p.exists():
        raise HTTPException(404, "clip không tồn tại")
    return p


@lru_cache(maxsize=16)
def load_annotations(clip_id: str) -> dict:
    return json.loads(_clip_path(clip_id, "json").read_text(encoding="utf-8"))


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/clips")
def list_clips():
    clips = []
    for j in sorted(CLIPS_DIR.glob("*.json")):
        if not j.with_suffix(".mp4").exists():
            continue
        a = load_annotations(j.stem)
        v = a["video"]
        clips.append({"id": j.stem, "title": a.get("title", j.stem), "fps": v["fps"], "frames": v["frames"], "duration": round(v["frames"] / v["fps"], 1)})
    return clips


@app.get("/api/clips/{clip_id}/meta")
def clip_meta(clip_id: str):
    a = load_annotations(clip_id)
    return {k: v for k, v in a.items() if k != "frames"}


@app.get("/api/clips/{clip_id}/video")
def clip_video(clip_id: str):
    # FileResponse hỗ trợ HTTP Range -> trình duyệt tua được
    return FileResponse(_clip_path(clip_id, "mp4"), media_type="video/mp4")


@app.websocket("/ws/clips/{clip_id}")
async def stream_annotations(ws: WebSocket, clip_id: str):
    """Client gửi {"type": "range", "from": a, "to": b} (frame, nửa mở [a, b)), server trả {"type": "frames", "frames": [...]}.
    Client giữ bộ đệm đi trước thời điểm phát và xin tiếp khi gần hết; tua thì xin khoảng mới."""
    await ws.accept()
    try:
        frames = load_annotations(clip_id)["frames"]
    except HTTPException:
        await ws.close(code=4404)
        return
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") != "range":
                continue
            a = max(int(msg.get("from", 0)), 0)
            b = min(int(msg.get("to", a)), a + MAX_RANGE, len(frames))
            await ws.send_json({"type": "frames", "from": a, "to": b, "frames": frames[a:b]})
    except WebSocketDisconnect:
        pass


def _job_status(job_id: str) -> dict:
    from rq.exceptions import NoSuchJobError
    from rq.job import Job

    try:
        job = Job.fetch(job_id, connection=jobs.get_queue().connection)
    except NoSuchJobError:
        raise HTTPException(404, "job không tồn tại")
    state = job.get_status(refresh=True).value
    out = {"id": job_id, "state": state, "stage": job.meta.get("stage"), "progress": job.meta.get("progress")}
    if state == "queued":
        out["position"] = job.get_position()
    if state == "finished":
        out["clip_id"] = f"job-{job_id}"
    if state == "failed":
        out["error"] = "Xử lý thất bại (video không đọc được hoặc lỗi pipeline)"
    return out


_redis_ok = {"value": False, "checked": 0.0}


@app.get("/api/config")
def config():
    # cache 30 s: không có Redis thì mỗi lần ping chờ hết timeout kết nối
    if time.time() - _redis_ok["checked"] > 30:
        try:
            jobs.get_queue().connection.ping()
            _redis_ok["value"] = True
        except Exception:
            _redis_ok["value"] = False  # không có Redis -> chỉ xem clip mẫu
        _redis_ok["checked"] = time.time()
    return {"upload_enabled": _redis_ok["value"], "max_seconds": jobs.MAX_SECONDS, "max_upload_mb": jobs.MAX_UPLOAD_MB}


@app.post("/api/jobs", status_code=202)
async def create_job(file: UploadFile):
    q = jobs.get_queue()
    try:
        queued = q.count
    except Exception:
        raise HTTPException(503, "Hàng đợi xử lý đang tắt, chỉ xem được clip mẫu")
    if queued >= jobs.MAX_QUEUE:
        raise HTTPException(429, "Hàng đợi đang đầy, thử lại sau ít phút")
    jobs.cleanup_old_jobs()
    job_id = uuid.uuid4().hex
    d = jobs.JOBS_DIR / job_id
    d.mkdir(parents=True)
    size, limit = 0, jobs.MAX_UPLOAD_MB * 1e6
    with open(d / "upload", "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > limit:
                f.close()
                (d / "upload").unlink()
                d.rmdir()
                raise HTTPException(413, f"File lớn hơn {jobs.MAX_UPLOAD_MB:.0f} MB")
            f.write(chunk)
    q.enqueue(jobs.process_upload, job_id, job_id=job_id, job_timeout=1800, result_ttl=int(jobs.JOB_TTL_HOURS * 3600))
    return _job_status(job_id)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise HTTPException(404, "job không tồn tại")
    return _job_status(job_id)


@app.websocket("/ws/jobs/{job_id}")
async def job_updates(ws: WebSocket, job_id: str):
    """Đẩy trạng thái job mỗi giây tới khi xong hoặc lỗi."""
    await ws.accept()
    try:
        while True:
            try:
                st = await asyncio.to_thread(job_status, job_id)
            except HTTPException:
                await ws.close(code=4404)
                return
            await ws.send_json(st)
            if st["state"] in ("finished", "failed", "canceled", "stopped"):
                await ws.close()
                return
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass


# frontend đã build (npm run build) -> phục vụ cùng cổng, 1 container duy nhất khi deploy
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
