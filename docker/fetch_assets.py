"""Tải weights + clip demo từ một HF Hub repo (khuyên để private) lúc container khởi động.

Không đưa weights (hàng trăm MB) và footage SoccerNet vào Git / image. Repo assets có dạng:
    best.pt, pitch_best.pt, clips/<id>.mp4, clips/<id>.json
Biến môi trường: ASSETS_REPO (vd "user/football-cv-assets"), HF_TOKEN (secret của Space, cần nếu repo private).
Không đặt ASSETS_REPO thì bỏ qua (chạy local với volume mount).
"""
import os
import shutil
from pathlib import Path

repo = os.getenv("ASSETS_REPO")
if not repo:
    print("ASSETS_REPO chưa đặt: bỏ qua tải assets")
    raise SystemExit(0)

from huggingface_hub import snapshot_download  # noqa: E402

local = Path(snapshot_download(repo, repo_type="model", token=os.getenv("HF_TOKEN"), allow_patterns=["*.pt", "clips/*"]))
weights, clips = Path(os.getenv("WEIGHTS_DIR", "/weights")), Path(os.getenv("CLIPS_DIR", "/data/clips"))
weights.mkdir(parents=True, exist_ok=True)
clips.mkdir(parents=True, exist_ok=True)
for f in local.glob("*.pt"):
    shutil.copy2(f, weights / f.name)
for f in (local / "clips").glob("*"):
    shutil.copy2(f, clips / f.name)
print(f"Đã tải từ {repo}: {sorted(p.name for p in weights.glob('*.pt'))}, {len(list(clips.glob('*.json')))} clip")
