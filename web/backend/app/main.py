"""API cho demo web: phục vụ video gốc + stream annotation JSON qua WebSocket. Trình duyệt tự vẽ overlay.

Mỗi clip trong CLIPS_DIR gồm <id>.mp4 (video gốc) và <id>.json (annotation, schema ở src/export/annotations.py).

    uvicorn app.main:app --reload --port 8000        (chạy trong web/backend)
"""
import json
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[3]
CLIPS_DIR = Path(os.getenv("CLIPS_DIR", ROOT / "outputs" / "web_clips"))
FRONTEND_DIST = Path(os.getenv("FRONTEND_DIST", ROOT / "web" / "frontend" / "dist"))
MAX_RANGE = 500  # số frame tối đa mỗi lần xin, tránh client gửi khoảng khổng lồ

app = FastAPI(title="football-cv-tracking demo")


def _clip_path(clip_id: str, ext: str) -> Path:
    # chỉ cho tên file đơn giản, chặn path traversal (../)
    if not clip_id.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(404, "clip không tồn tại")
    p = CLIPS_DIR / f"{clip_id}.{ext}"
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


# frontend đã build (npm run build) -> phục vụ cùng cổng, 1 container duy nhất khi deploy
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
