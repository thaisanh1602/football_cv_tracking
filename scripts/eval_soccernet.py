"""Chạy một hoặc nhiều tracker trên subset SoccerNet-Tracking và tính HOTA / IDF1 / IDSW.

Detection được cache theo sequence (outputs/det_cache) nên so sánh nhiều tracker chỉ tốn detect 1 lần,
và mọi tracker nhận đúng cùng một bộ detection -> so sánh công bằng.

Tải dữ liệu (giải nén các file zip vào data/SoccerNet/tracking/test/):
    from SoccerNet.Downloader import SoccerNetDownloader
    SoccerNetDownloader(LocalDirectory="data/SoccerNet").downloadDataTask(task="tracking", split=["test"])

Chạy:
    python scripts/eval_soccernet.py --data-root data/SoccerNet/tracking/test --weights outputs/best.pt \
        --trackers configs/bytetrack.yaml configs/botsort.yaml configs/botsort_reid.yaml --limit 5
"""
import argparse
import configparser
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.detection.ball_sahi import SlicedBallDetector  # noqa: E402
from src.detection.detector import Detector, split_people_ball  # noqa: E402
from src.eval.export_mot import MOTWriter  # noqa: E402
from src.eval.run_trackeval import run_trackeval  # noqa: E402
from src.tracking.ball_tracker import BallTracker  # noqa: E402
from src.tracking.trackers import PeopleTracker  # noqa: E402
from src.utils.label_mapping import BALL_ID  # noqa: E402


def list_frames(seq_dir: Path) -> list[Path]:
    info = configparser.ConfigParser()
    info.read(seq_dir / "seqinfo.ini")
    seq = info["Sequence"]
    return sorted((seq_dir / seq.get("imDir", "img1")).glob(f"*{seq.get('imExt', '.jpg')}"))


def detect_sequence(frames: list[Path], cache_file: Path, detector, ball_detector) -> list[sv.Detections]:
    """Detect cả sequence, lưu cache dạng mảng (frame, x1, y1, x2, y2, conf, cls)."""
    if cache_file.exists():
        data = np.load(cache_file)["dets"]
    else:
        rows = []
        for i, path in enumerate(tqdm(frames, desc=f"Detect {cache_file.stem}", leave=False)):
            frame = cv2.imread(str(path))
            dets = detector(frame)
            if ball_detector:
                people, _ = split_people_ball(dets)
                dets = sv.Detections.merge([people, ball_detector(frame)])
            rows.append(np.column_stack([np.full(len(dets), i), dets.xyxy, dets.confidence, dets.class_id]))
        data = np.concatenate(rows) if rows else np.empty((0, 7))
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, dets=data)

    out = []
    for i in range(len(frames)):
        d = data[data[:, 0] == i]
        out.append(sv.Detections(xyxy=d[:, 1:5], confidence=d[:, 5], class_id=d[:, 6].astype(int)))
    return out


def track_sequence(frames, dets_per_frame, out_file: Path, tracker_cfg, include_ball, device) -> tuple[float, int]:
    """Trả về tổng thời gian chạy tracker (không tính đọc ảnh) và số frame."""
    tracker = PeopleTracker(tracker_cfg, device=device)  # tracker mới cho mỗi sequence
    ball_tracker = BallTracker()
    seconds = 0.0
    with MOTWriter(out_file) as mot:
        for i, (path, dets) in enumerate(zip(frames, dets_per_frame)):
            frame = cv2.imread(str(path))  # BoT-SORT cần ảnh cho GMC và ReID
            people, ball = dets[dets.class_id != BALL_ID], dets[dets.class_id == BALL_ID]
            t0 = time.perf_counter()
            tracks = tracker.update(people, frame)
            seconds += time.perf_counter() - t0
            mot.write(i + 1, tracks)
            if include_ball:
                mot.write(i + 1, ball_tracker.update(ball))
    return seconds, len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="thư mục chứa các SNMOT-xxx")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--trackers", nargs="+", default=[str(ROOT / "configs" / "bytetrack.yaml")])
    ap.add_argument("--seqs", nargs="*", default=None, help="danh sách sequence, mặc định tất cả")
    ap.add_argument("--limit", type=int, default=None, help="chỉ lấy N sequence đầu (subset)")
    ap.add_argument("--out", default="outputs/trackeval")
    ap.add_argument("--cache", default="outputs/det_cache")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--ball-conf", type=float, default=0.1)
    ap.add_argument("--include-ball", action="store_true", help="GT SoccerNet có cả bóng; bật để đánh giá cả bóng")
    ap.add_argument("--sahi", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    root = Path(args.data_root)
    seqs = args.seqs or sorted(p.name for p in root.iterdir() if (p / "gt" / "gt.txt").exists() and (p / "img1").is_dir())
    seqs = seqs[: args.limit] if args.limit else seqs

    # cache theo cấu hình detect: đổi weights / imgsz / conf / sahi thì cache mới
    cache_dir = Path(args.cache) / f"{Path(args.weights).stem}_{args.imgsz}_{args.conf}_{args.ball_conf}{'_sahi' if args.sahi else ''}"
    detector = Detector(args.weights, imgsz=args.imgsz, conf=args.conf, ball_conf=args.ball_conf, device=args.device)
    ball_detector = SlicedBallDetector(detector) if args.sahi else None
    detections = {}
    for seq in tqdm(seqs, desc="Detect"):
        frames = list_frames(root / seq)
        detections[seq] = (frames, detect_sequence(frames, cache_dir / f"{seq}.npz", detector, ball_detector))

    results = {}
    for cfg in args.trackers:
        name = Path(cfg).stem
        seconds = n_frames = 0
        for seq in tqdm(seqs, desc=f"Track {name}"):
            frames, dets = detections[seq]
            s, n = track_sequence(frames, dets, Path(args.out) / name / "data" / f"{seq}.txt", cfg, args.include_ball, args.device)
            seconds += s
            n_frames += n
        metrics = run_trackeval(root, args.out, name, seqs)
        metrics["tracker_fps"] = round(n_frames / seconds, 1) if seconds else None
        results[name] = metrics
        (Path(args.out) / name / "summary.json").write_text(json.dumps({"seqs": seqs, **metrics}, indent=2))

    print(f"\n{len(seqs)} sequence: {', '.join(seqs)}\n")
    print("| Tracker | HOTA | IDF1 | IDSW | MOTA | FPS tracker |")
    print("|---|---|---|---|---|---|")
    for name, m in results.items():
        print(f"| {name} | {m['HOTA']:.1f} | {m['IDF1']:.1f} | {m['IDSW']} | {m['MOTA']:.1f} | {m['tracker_fps']} |")


if __name__ == "__main__":
    main()
