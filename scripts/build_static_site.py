"""Đóng gói demo thành site tĩnh (không backend) để deploy lên HF Static Space / GitHub Pages / Vercel.

Frontend tự chuyển sang chế độ static khi không gọi được api/config: đọc clips/index.json, tải trọn
clips/<id>.json và phát clips/<id>.mp4. Upload / job queue chỉ có ở bản chạy bằng Docker.

    cd web/frontend && npm run build && cd ../..
    python scripts/build_static_site.py --clips outputs/web_clips --out outputs/static_site
    python scripts/build_static_site.py --clips outputs/web_clips --out outputs/static_site --push <user>/football-cv-demo
(--push cần `huggingface-cli login` với token quyền write.)
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SPACE_README = """---
title: Football CV Tracking
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: static
pinned: false
short_description: Tracking, team classification, 2D radar with real speeds
---

Detection, tracking, team classification and a 2D radar with real player speeds on broadcast football video.
Overlays are drawn in the browser (Canvas) from per-frame annotation JSON; this static build plays pre-processed
clips, the full version (FastAPI + WebSocket streaming + upload job queue) runs with Docker.

Code, evaluation and write-ups: https://github.com/thaisanh1602/football_cv_tracking

Research demo. Footage from the SoccerNet dataset, used for non-commercial research purposes only.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="outputs/web_clips", help="thư mục chứa <id>.mp4 + <id>.json")
    ap.add_argument("--dist", default=str(ROOT / "web" / "frontend" / "dist"))
    ap.add_argument("--out", default="outputs/static_site")
    ap.add_argument("--push", default=None, help="id HF Space (repo_type=space) để upload, vd user/football-cv-demo")
    args = ap.parse_args()

    dist, out, clips = Path(args.dist), Path(args.out), Path(args.clips)
    if not (dist / "index.html").exists():
        raise SystemExit(f"Chưa build frontend: chạy `npm run build` trong web/frontend (thiếu {dist / 'index.html'})")
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(dist, out)
    (out / "clips").mkdir()

    index = []
    for j in sorted(clips.glob("*.json")):
        mp4 = j.with_suffix(".mp4")
        if not mp4.exists():
            continue
        doc = json.loads(j.read_text(encoding="utf-8"))
        v = doc["video"]
        index.append({"id": j.stem, "title": doc.get("title", j.stem), "fps": v["fps"], "frames": v["frames"], "duration": round(v["frames"] / v["fps"], 1)})
        shutil.copy2(j, out / "clips" / j.name)
        shutil.copy2(mp4, out / "clips" / mp4.name)
    if not index:
        raise SystemExit(f"Không có cặp <id>.mp4 + <id>.json nào trong {clips}")
    (out / "clips" / "index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    (out / "README.md").write_text(SPACE_README, encoding="utf-8")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"Đã đóng gói {len(index)} clip vào {out} ({size:.0f} MB): {[c['id'] for c in index]}")

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push, repo_type="space", space_sdk="static", exist_ok=True)
        api.upload_folder(folder_path=out, repo_id=args.push, repo_type="space")
        print(f"Đã upload: https://huggingface.co/spaces/{args.push}")


if __name__ == "__main__":
    main()
