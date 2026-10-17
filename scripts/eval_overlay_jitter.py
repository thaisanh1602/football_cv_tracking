"""Đo rung (jitter) và sai số của overlay trên SoccerNet-Tracking.

Điểm neo overlay = chân cầu thủ (đáy giữa box), nơi vẽ ellipse. So sánh:
- Detection thô: box của detector mỗi frame
- Kalman của BoT-SORT: box mà tracker trả về (đã qua Kalman filter của tracker)
- GT: box chú thích tay của SoccerNet (tham chiếu: người thật chuyển động mượt cỡ nào)

Chỉ đọc detection đã cache (outputs/det_cache, tạo bởi eval_soccernet.py) và chạy optical flow trên ảnh
thu nhỏ: chạy bằng CPU, không cần GPU. Kết quả từng sequence được cache ở outputs/jitter_cache để thử bộ lọc
ở tuần sau mà không đọc lại ảnh.

    python scripts/eval_overlay_jitter.py --data-root <SoccerNet tracking split> --limit 10
"""
import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_idsw import ball_ids, iou_matrix, load_mot  # noqa: E402
from scripts.eval_soccernet import detect_sequence, list_frames  # noqa: E402
from src.stabilize.camera_motion import CameraMotion  # noqa: E402
from src.stabilize.metrics import jitter_values, summarize_jitter  # noqa: E402
from src.tracking.trackers import PeopleTracker  # noqa: E402
from src.utils.label_mapping import BALL_ID  # noqa: E402


def foot(xyxy: np.ndarray) -> np.ndarray:
    """(N,4) xyxy -> (N,3): x chân, y chân, chiều cao box."""
    xyxy = np.asarray(xyxy, dtype=float).reshape(-1, 4)
    return np.column_stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3], xyxy[:, 3] - xyxy[:, 1]])


def run_sequence(seq_dir: Path, det_cache: Path, tracker_cfg: str, cache_file: Path) -> dict:
    """Trả về {"motions": (n,3,3), "rows": (m, 10) = frame, track id, det xyxy, tracker xyxy}."""
    if cache_file.exists():
        return pickle.loads(cache_file.read_bytes())
    if not det_cache.exists():
        raise SystemExit(f"Thiếu {det_cache}: chạy eval_soccernet.py trước (trên Kaggle nếu máy yếu) để có detection cache")
    frames = list_frames(seq_dir)
    dets_all = detect_sequence(frames, det_cache, None, None)
    tracker, cam = PeopleTracker(tracker_cfg), CameraMotion()
    motions, rows = [], []
    for i, (path, dets) in enumerate(tqdm(zip(frames, dets_all), total=len(frames), desc=seq_dir.name, leave=False)):
        frame = cv2.imread(str(path))
        people = dets[dets.class_id != BALL_ID]
        motions.append(cam.update(frame, people.xyxy))
        tracks = tracker.update(people, frame)
        for tid, d, t in zip(tracks.tracker_id if len(tracks) else [], tracks.data.get("det_xyxy", []), tracks.xyxy):
            rows.append([i + 1, tid, *d, *t])
    out = {"motions": np.array(motions), "rows": np.array(rows, dtype=float).reshape(-1, 10)}
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(pickle.dumps(out))
    return out


def tracks_from(frames, ids, anchors) -> dict:
    """Gom theo track id: id -> list (frame, x, y, h), sắp theo frame."""
    tracks = defaultdict(list)
    for f, tid, a in sorted(zip(frames, ids, anchors), key=lambda r: (r[1], r[0])):
        tracks[int(tid)].append((int(f), *a))
    return tracks


def errors_vs_gt(rows: np.ndarray, anchors: dict[str, np.ndarray], gt: dict) -> dict[str, list]:
    """Ghép box detection với GT mỗi frame (IoU >= 0.5, cùng cặp cho mọi biến thể -> công bằng),
    sai số = khoảng cách điểm neo tới chân GT / chiều cao box GT * 100."""
    errs = {k: [] for k in anchors}
    for f in np.unique(rows[:, 0]):
        g = gt.get(int(f))
        sel = np.flatnonzero(rows[:, 0] == f)
        if g is None or len(sel) == 0:
            continue
        d = rows[sel, 2:6]
        iou = iou_matrix(np.column_stack([d[:, :2], d[:, 2:] - d[:, :2]]), g[:, 1:5])
        for r, c in zip(*linear_sum_assignment(-iou)):
            if iou[r, c] < 0.5:
                continue
            gx, gy, gh = g[c, 1] + g[c, 3] / 2, g[c, 2] + g[c, 4], g[c, 4]
            for k, a in anchors.items():
                errs[k].append(np.hypot(a[sel[r], 0] - gx, a[sel[r], 1] - gy) / gh * 100)
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="thư mục chứa các SNMOT-xxx (có img1/, gt/)")
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--det-cache", default="outputs/det_cache/best_1280_0.25_0.1")
    ap.add_argument("--tracker", default=str(ROOT / "configs" / "botsort_tuned.yaml"))
    ap.add_argument("--cache", default="outputs/jitter_cache")
    ap.add_argument("--out", default="outputs/jitter_eval")
    args = ap.parse_args()

    root = Path(args.data_root)
    seqs = args.seqs or sorted(p.name for p in root.iterdir() if (p / "img1").is_dir() and (p / "gt" / "gt.txt").exists())
    seqs = seqs[: args.limit] if args.limit else seqs

    jit = defaultdict(list)
    errs = defaultdict(list)
    for seq in tqdm(seqs, desc="Sequences"):
        r = run_sequence(root / seq, Path(args.det_cache) / f"{seq}.npz", args.tracker, Path(args.cache) / f"{seq}.pkl")
        motions = {i + 1: M for i, M in enumerate(r["motions"])}
        rows = r["rows"]
        anchors = {"Detection thô": foot(rows[:, 2:6]), "Kalman của BoT-SORT": foot(rows[:, 6:10])}

        gt = load_mot(root / seq / "gt" / "gt.txt", exclude_ids=ball_ids(root / seq))
        g_rows = np.array([[f, *b] for f, arr in gt.items() for b in arr])
        g_xyxy = np.column_stack([g_rows[:, 2], g_rows[:, 3], g_rows[:, 2] + g_rows[:, 4], g_rows[:, 3] + g_rows[:, 5]])
        gt_tracks = tracks_from(g_rows[:, 0], g_rows[:, 1], foot(g_xyxy))

        for k, a in anchors.items():
            tr = tracks_from(rows[:, 0], rows[:, 1], a)
            jit[k].append(jitter_values(tr, motions))
            jit[k + " (không bù camera)"].append(jitter_values(tr, motions, compensate=False))
        jit["GT"].append(jitter_values(gt_tracks, motions))
        jit["GT (không bù camera)"].append(jitter_values(gt_tracks, motions, compensate=False))
        for k, v in errors_vs_gt(rows, anchors, gt).items():
            errs[k] += v

    results = {}
    for k, parts in jit.items():
        results[k] = summarize_jitter(np.concatenate(parts))
        if k in errs:
            e = np.array(errs[k])
            results[k] |= {"err_median": round(float(np.median(e)), 2), "err_p90": round(float(np.percentile(e, 90)), 2)}

    print(f"\n{len(seqs)} sequence: {', '.join(seqs)}")
    print("Jitter = |gia tốc chân| sau bù camera, % chiều cao box / frame². Sai số = |chân - chân GT|, % chiều cao box.\n")
    print("| Overlay | Jitter trung vị | Jitter p90 | Sai số trung vị | Sai số p90 |")
    print("|---|---|---|---|---|")
    for k, r in results.items():
        print(f"| {k} | {r['jitter_median']} | {r['jitter_p90']} | {r.get('err_median', '')} | {r.get('err_p90', '')} |")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"seqs": seqs, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
