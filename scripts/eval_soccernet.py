"""Chạy tracker trên subset SoccerNet-Tracking và tính HOTA / IDF1 / IDSW.

Tải dữ liệu (giải nén các file zip vào data/SoccerNet/tracking/test/):
    from SoccerNet.Downloader import SoccerNetDownloader
    SoccerNetDownloader(LocalDirectory="data/SoccerNet").downloadDataTask(task="tracking", split=["test"])

Chạy:
    python scripts/eval_soccernet.py --data-root data/SoccerNet/tracking/test --weights outputs/best.pt \
        --tracker configs/bytetrack.yaml --name bytetrack --limit 5
"""
import argparse
import configparser
import json
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.detection.ball_sahi import SlicedBallDetector  # noqa: E402
from src.detection.detector import Detector, split_people_ball  # noqa: E402
from src.eval.export_mot import MOTWriter  # noqa: E402
from src.eval.run_trackeval import run_trackeval  # noqa: E402
from src.tracking.ball_tracker import BallTracker  # noqa: E402
from src.tracking.trackers import PeopleTracker  # noqa: E402


def track_sequence(seq_dir: Path, out_file: Path, detector, ball_detector, tracker_cfg, include_ball, device):
    info = configparser.ConfigParser()
    info.read(seq_dir / "seqinfo.ini")
    seq = info["Sequence"]
    frames = sorted((seq_dir / seq.get("imDir", "img1")).glob(f"*{seq.get('imExt', '.jpg')}"))

    tracker = PeopleTracker(tracker_cfg, device=device)  # tracker mới cho mỗi sequence
    ball_tracker = BallTracker()
    with MOTWriter(out_file) as mot:
        for i, path in enumerate(tqdm(frames, desc=seq_dir.name, leave=False)):
            frame = cv2.imread(str(path))
            people, ball = split_people_ball(detector(frame))
            mot.write(i + 1, tracker.update(people, frame))
            if include_ball:
                mot.write(i + 1, ball_tracker.update(ball_detector(frame) if ball_detector else ball))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="thư mục chứa các SNMOT-xxx")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tracker", default=str(ROOT / "configs" / "bytetrack.yaml"))
    ap.add_argument("--name", default=None, help="tên thí nghiệm, mặc định = tên file config")
    ap.add_argument("--seqs", nargs="*", default=None, help="danh sách sequence, mặc định tất cả")
    ap.add_argument("--limit", type=int, default=None, help="chỉ lấy N sequence đầu (subset)")
    ap.add_argument("--out", default="outputs/trackeval")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--include-ball", action="store_true", help="GT SoccerNet có cả bóng; bật để đánh giá cả bóng")
    ap.add_argument("--sahi", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    root = Path(args.data_root)
    seqs = args.seqs or sorted(p.name for p in root.iterdir() if (p / "gt" / "gt.txt").exists())
    seqs = seqs[: args.limit] if args.limit else seqs
    name = args.name or Path(args.tracker).stem

    detector = Detector(args.weights, imgsz=args.imgsz, conf=args.conf, device=args.device)
    ball_detector = SlicedBallDetector(detector) if args.sahi else None
    for seq in tqdm(seqs, desc=name):
        out_file = Path(args.out) / name / "data" / f"{seq}.txt"
        track_sequence(root / seq, out_file, detector, ball_detector, args.tracker, args.include_ball, args.device)

    metrics = run_trackeval(root, args.out, name, seqs)
    print(json.dumps({"tracker": name, "seqs": len(seqs), **metrics}, indent=2))
    (Path(args.out) / name / "summary.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
