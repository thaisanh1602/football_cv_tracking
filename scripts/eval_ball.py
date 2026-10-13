"""Đo hiệu quả detect bóng trên SoccerNet-Tracking: full-frame vs SAHI vs kết hợp, có/không BallTracker.

Bóng chỉ ~14x12 px nên match theo khoảng cách tâm (mặc định <= 15 px) thay vì IoU.

    python scripts/eval_ball.py --data-root data/SoccerNet/tracking/train --weights outputs/best.pt --limit 5 --stride 5
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
from src.tracking.ball_tracker import BallTracker  # noqa: E402

MODES = ["full", "sahi", "full+sahi"]


def load_ball_gt(seq_dir: Path) -> dict[int, np.ndarray]:
    """frame -> mảng tâm bóng GT (Nx2). ID bóng lấy từ gameinfo.ini (trackletID_x= ball;...)."""
    info = configparser.ConfigParser()
    info.read(seq_dir / "gameinfo.ini")
    ball_ids = [
        int(k.split("_")[1]) for k, v in info["Sequence"].items() if k.startswith("trackletid_") and v.strip().startswith("ball")
    ]
    gt = np.loadtxt(seq_dir / "gt" / "gt.txt", delimiter=",", ndmin=2)
    gt = gt[np.isin(gt[:, 1], ball_ids)]
    centers = np.column_stack([gt[:, 2] + gt[:, 4] / 2, gt[:, 3] + gt[:, 5] / 2])
    return {int(f): centers[gt[:, 0] == f] for f in np.unique(gt[:, 0])}


def match(pred: np.ndarray, gt: np.ndarray, thresh: float) -> tuple[int, list[float]]:
    """Greedy match 1-1 theo khoảng cách tâm. Trả về số TP và sai số tâm của các TP."""
    if len(pred) == 0 or len(gt) == 0:
        return 0, []
    dist = np.linalg.norm(pred[:, None] - gt[None], axis=2)
    tp, errors = 0, []
    while dist.size and dist.min() <= thresh:
        i, j = np.unravel_index(dist.argmin(), dist.shape)
        tp += 1
        errors.append(float(dist[i, j]))
        dist[i, :] = np.inf
        dist[:, j] = np.inf
    return tp, errors


class Stats:
    def __init__(self):
        self.tp = self.n_pred = self.n_gt = 0
        self.errors = []
        self.seconds = 0.0
        self.frames = 0

    def add(self, pred: sv.Detections, gt: np.ndarray, thresh: float):
        centers = pred.get_anchors_coordinates(sv.Position.CENTER) if len(pred) else np.empty((0, 2))
        tp, errors = match(centers, gt, thresh)
        self.tp += tp
        self.n_pred += len(pred)
        self.n_gt += len(gt)
        self.errors += errors

    def summary(self) -> dict:
        p = self.tp / self.n_pred if self.n_pred else 0.0
        r = self.tp / self.n_gt if self.n_gt else 0.0
        return {
            "recall": round(r, 3),
            "precision": round(p, 3),
            "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
            "center_err_px": round(float(np.mean(self.errors)), 1) if self.errors else None,
            "fps": round(self.frames / self.seconds, 2) if self.seconds else None,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="thư mục chứa các SNMOT-xxx")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stride", type=int, default=5, help="lấy mỗi N frame để chạy nhanh hơn")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25, help="conf của người")
    ap.add_argument("--ball-conf", type=float, default=0.1, help="conf của bóng (full-frame và SAHI)")
    ap.add_argument("--slice", type=int, default=960)
    ap.add_argument("--tile-imgsz", type=int, default=640, help="tile-imgsz/slice ~ imgsz/1920 (vd 640/960) để bóng cùng scale lúc train")
    ap.add_argument("--overlap", type=int, default=192)
    ap.add_argument("--dist-thresh", type=float, default=15)
    ap.add_argument("--out", default="outputs/ball_eval")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    root = Path(args.data_root)
    seqs = args.seqs or sorted(p.name for p in root.iterdir() if (p / "img1").is_dir() and (p / "gameinfo.ini").exists())
    seqs = seqs[: args.limit] if args.limit else seqs

    detector = Detector(args.weights, imgsz=args.imgsz, conf=args.conf, ball_conf=args.ball_conf, device=args.device)
    sahi = SlicedBallDetector(detector, slice_wh=args.slice, overlap_wh=args.overlap, tile_imgsz=args.tile_imgsz, conf=args.ball_conf)
    stats = {f"{m}{t}": Stats() for m in MODES for t in ("", " + tracker")}

    for seq in tqdm(seqs, desc="Sequences"):
        gt_by_frame = load_ball_gt(root / seq)
        frames = sorted((root / seq / "img1").glob("*.jpg"))[:: args.stride]
        # bóng di chuyển xa hơn khi bỏ frame -> nới ngưỡng nhảy theo stride
        trackers = {m: BallTracker(max_jump_px=120 * args.stride, max_lost=max(25 // args.stride, 3)) for m in MODES}

        for path in tqdm(frames, desc=seq, leave=False):
            frame = cv2.imread(str(path))
            gt = gt_by_frame.get(int(path.stem), np.empty((0, 2)))

            t0 = time.perf_counter()
            _, ball_full = split_people_ball(detector(frame))
            t1 = time.perf_counter()
            ball_sahi = sahi(frame)
            t2 = time.perf_counter()

            preds = {
                "full": (ball_full, t1 - t0),
                "sahi": (ball_sahi, (t1 - t0) + (t2 - t1)),  # pipeline vẫn cần full-frame cho người
                "full+sahi": (sv.Detections.merge([ball_full, ball_sahi]).with_nms(threshold=0.3), t2 - t0),
            }
            for mode, (pred, seconds) in preds.items():
                for key, p in ((mode, pred), (f"{mode} + tracker", trackers[mode].update(pred))):
                    stats[key].add(p, gt, args.dist_thresh)
                    stats[key].seconds += seconds
                    stats[key].frames += 1

    results = {k: s.summary() for k, s in stats.items()}
    print(f"\n{len(seqs)} sequence, stride {args.stride}, match tâm <= {args.dist_thresh}px\n")
    print("| Mode | Recall | Precision | F1 | Sai số tâm (px) | FPS |")
    print("|---|---|---|---|---|---|")
    for k, r in results.items():
        print(f"| {k} | {r['recall']} | {r['precision']} | {r['f1']} | {r['center_err_px']} | {r['fps']} |")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"seqs": seqs, "args": vars(args), "results": results}, indent=2))


if __name__ == "__main__":
    main()
