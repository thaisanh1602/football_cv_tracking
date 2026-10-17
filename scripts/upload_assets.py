"""Đưa weights + clip demo lên một HF Hub model repo (mặc định private) để Space tải lúc khởi động.

Cần đăng nhập trước: `huggingface-cli login` (token quyền write, tạo ở https://huggingface.co/settings/tokens).

    python scripts/upload_assets.py --repo <user>/football-cv-assets \\
        --weights outputs/best.pt outputs/pitch_best.pt --clips outputs/web_clips
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="vd thaisanh1602/football-cv-assets")
    ap.add_argument("--weights", nargs="+", default=["outputs/best.pt", "outputs/pitch_best.pt"])
    ap.add_argument("--clips", default="outputs/web_clips", help="thư mục chứa <id>.mp4 + <id>.json")
    ap.add_argument("--public", action="store_true", help="mặc định private: footage SoccerNet chỉ dùng nghiên cứu")
    args = ap.parse_args()

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=not args.public, exist_ok=True)
    for w in args.weights:
        api.upload_file(path_or_fileobj=w, path_in_repo=Path(w).name, repo_id=args.repo)
        print("đã upload", w)
    clips = Path(args.clips)
    pairs = [p for p in clips.glob("*.json") if p.with_suffix(".mp4").exists()]
    api.upload_folder(folder_path=clips, path_in_repo="clips", repo_id=args.repo, allow_patterns=["*.mp4", "*.json"])
    print(f"đã upload {len(pairs)} clip: {[p.stem for p in pairs]}")
    print(f"https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
