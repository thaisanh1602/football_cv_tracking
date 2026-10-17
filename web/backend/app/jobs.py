"""Job queue xử lý clip người dùng upload (Redis + RQ). Worker chạy trên CPU nên giới hạn độ dài clip.

Luồng: API lưu file -> enqueue process_upload(job_id) -> worker: ffmpeg chuẩn hoá (cắt <= MAX_SECONDS, H.264,
faststart để trình duyệt tua được) -> chạy scripts/infer_video.py --no-video --json-out -> clip.mp4 + clip.json.
Tiến độ đọc từ file --progress-file và ghi vào job.meta để API đẩy qua WebSocket.

    rq worker clips --url redis://localhost:6379/0      (Linux / Docker; RQ không hỗ trợ Windows)
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
JOBS_DIR = Path(os.getenv("JOBS_DIR", ROOT / "outputs" / "jobs"))
WEIGHTS_DIR = Path(os.getenv("WEIGHTS_DIR", ROOT / "outputs"))
MAX_SECONDS = float(os.getenv("MAX_UPLOAD_SECONDS", "10"))
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "60"))
MAX_QUEUE = int(os.getenv("MAX_QUEUE", "5"))
JOB_TTL_HOURS = float(os.getenv("JOB_TTL_HOURS", "24"))
DEVICE = os.getenv("DEVICE", "cpu")
# CPU ~1 frame/s cho detect: keypoint sân và SigLIP chỉ chạy thưa, giữa các lần dùng kết quả gần nhất
PITCH_EVERY = os.getenv("PITCH_EVERY", "5")
TEAM_EVERY = os.getenv("TEAM_EVERY", "5")
QUEUE_NAME = "clips"


def get_queue():
    from redis import Redis
    from rq import Queue

    return Queue(QUEUE_NAME, connection=Redis.from_url(REDIS_URL, socket_connect_timeout=2, socket_timeout=5))


def cleanup_old_jobs():
    """Xoá thư mục job cũ hơn JOB_TTL_HOURS để ổ đĩa của Space không đầy."""
    if not JOBS_DIR.exists():
        return
    cutoff = time.time() - JOB_TTL_HOURS * 3600
    for d in JOBS_DIR.iterdir():
        if d.is_dir() and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)


def process_upload(job_id: str):
    """Chạy trong worker."""
    from rq import get_current_job

    job = get_current_job()
    d = JOBS_DIR / job_id

    def status(stage: str, progress: float | None = None):
        job.meta.update(stage=stage, progress=progress)
        job.save_meta()

    status("transcode")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(d / "upload"), "-t", str(MAX_SECONDS),
         "-vf", "scale='min(1920,iw)':-2", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-preset", "veryfast", "-crf", "23", "-movflags", "+faststart", str(d / "clip.mp4")],
        check=True,
    )
    (d / "upload").unlink(missing_ok=True)

    status("analyze", 0.0)
    cmd = [
        sys.executable, str(ROOT / "scripts" / "infer_video.py"),
        "--source", str(d / "clip.mp4"),
        "--weights", str(WEIGHTS_DIR / "best.pt"),
        "--pitch-weights", str(WEIGHTS_DIR / "pitch_best.pt"),
        "--team", "--team-stride", "10", "--team-every", TEAM_EVERY, "--pitch-every", PITCH_EVERY,
        "--no-video", "--json-out", str(d / "clip.json"), "--progress-file", str(d / "progress"),
        "--device", DEVICE,
    ]
    with open(d / "log.txt", "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        while proc.poll() is None:
            time.sleep(1)
            try:
                done, total = (d / "progress").read_text().split("/")
                status("analyze", min(int(done) / max(int(total), 1), 1.0))
            except (FileNotFoundError, ValueError):
                pass
    if proc.returncode != 0:
        tail = (d / "log.txt").read_text(encoding="utf-8", errors="replace")[-1500:]
        raise RuntimeError(f"infer_video thất bại:\n{tail}")
    status("done", 1.0)
