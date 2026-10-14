"""Đo sai số radar (mét) và độ rung trên SoccerNet Game State Reconstruction (GSR).

GSR có vị trí chân thật của từng người trên sân (bbox_pitch, mét, gốc ở tâm sân). Để đo riêng homography
(không lẫn lỗi detector), lấy điểm chân GT trên ảnh (đáy giữa bbox_image) chiếu qua H của mình rồi so với
bbox_pitch. So sánh: H thô từng frame vs H đã làm mượt.

- sai số: trung bình / trung vị khoảng cách (m), % người lệch <= 1 m, <= 2 m
- độ phủ: % frame có H dùng được
- độ rung: |gia tốc rời rạc| của từng track trên radar (cm/frame^2), so với GT. Rung do H nhảy làm số này
  lớn hơn GT nhiều; người thật chạy mượt nên GT nhỏ.

Tải vài clip GSR (mỗi clip ~200 MB) bằng HTTP range, không cần tải cả file zip 9 GB:
    pip install remotezip
    python -c "from remotezip import RemoteZip; z=RemoteZip('https://huggingface.co/datasets/SoccerNet/SN-GSR-2025/resolve/main/test.zip'); [z.extract(n,'data/SoccerNet/gsr/test') for n in z.namelist() if n.startswith('SNGS-116/') and not n.endswith('/')]"

    python scripts/eval_pitch.py --data-root data/SoccerNet/gsr/test --pitch-weights outputs/pitch_rf.pt --pitch-imgsz 640
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pitch.config import PITCH  # noqa: E402
from src.pitch.homography import HomographySmoother, estimate_homography, project  # noqa: E402
from src.pitch.keypoints import PitchKeypointDetector  # noqa: E402

# GSR: gốc ở tâm sân. Chiều trục y chưa chắc trùng với config -> thử 4 tổ hợp dấu, chọn tổ hợp sai số nhỏ nhất
AXIS_SIGNS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def load_gsr(clip_dir: Path):
    """frame index (0-based) -> list (track_id, điểm chân trên ảnh (x, y), vị trí GT trên sân (x, y) gốc tâm sân)."""
    d = json.load(open(clip_dir / "Labels-GameState.json", encoding="utf-8"))
    frame_of = {img["image_id"]: int(Path(img["file_name"]).stem) - 1 for img in d["images"]}
    people = defaultdict(list)
    for a in d["annotations"]:
        if a.get("supercategory") != "object" or not a.get("bbox_pitch") or a["attributes"].get("role") == "ball":
            continue
        b, p = a["bbox_image"], a["bbox_pitch"]
        people[frame_of[a["image_id"]]].append((a["track_id"], (b["x"] + b["w"] / 2, b["y"] + b["h"]), (p["x_bottom_middle"], p["y_bottom_middle"])))
    return people, len(d["images"])


def detect_keypoints(clip_dir: Path, n_frames: int, detector, cache_file: Path):
    if cache_file.exists():
        c = np.load(cache_file)
        return c["xy"], c["conf"]
    xy, conf = np.zeros((n_frames, 32, 2)), np.zeros((n_frames, 32))
    for i in tqdm(range(n_frames), desc=f"Keypoint {clip_dir.name}", leave=False):
        xy[i], conf[i] = detector(cv2.imread(str(clip_dir / "img1" / f"{i + 1:06d}.jpg")))
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, xy=xy, conf=conf)
    return xy, conf


def run_variant(xy, conf, people, n_frames, smoother, signs, min_conf):
    """Trả về: sai số từng người (m), số frame có H, dict track -> chuỗi (frame, dự đoán, GT) để đo rung."""
    errors, covered, tracks = [], 0, defaultdict(list)
    for i in range(n_frames):
        raw = estimate_homography(xy[i], conf[i], min_conf=min_conf)
        H = smoother.update(raw) if smoother else (raw.H if raw else None)
        if H is None or not people.get(i):
            continue
        covered += 1
        tids, img_pts, gt = zip(*people[i])
        # config: gốc ở góc sân -> đổi sang gốc tâm sân, rồi áp dấu trục của GSR
        pred = (project(H, np.array(img_pts)) - [PITCH.length / 2, PITCH.width / 2]) * signs
        gt = np.array(gt)
        errors += list(np.linalg.norm(pred - gt, axis=1))
        for t, p, g in zip(tids, pred, gt):
            tracks[t].append((i, p, g))
    return np.array(errors), covered, tracks


def jitter_cm(tracks) -> tuple[float, float]:
    """|x[t] - 2x[t-1] + x[t-2]| trung bình trên các đoạn 3 frame liên tiếp, của dự đoán và của GT (cm)."""
    pred_acc, gt_acc = [], []
    for seq in tracks.values():
        for (f0, p0, g0), (f1, p1, g1), (f2, p2, g2) in zip(seq, seq[1:], seq[2:]):
            if f2 - f0 == 2:
                pred_acc.append(np.linalg.norm(p2 - 2 * p1 + p0))
                gt_acc.append(np.linalg.norm(g2 - 2 * g1 + g0))
    return float(np.mean(pred_acc)) * 100, float(np.mean(gt_acc)) * 100


def summarize(errors, covered, n_frames, jit):
    return {
        "coverage": round(covered / n_frames, 3),
        "mean_m": round(float(np.mean(errors)), 2),
        "median_m": round(float(np.median(errors)), 2),
        "within_1m": round(float((errors <= 1).mean()), 3),
        "within_2m": round(float((errors <= 2).mean()), 3),
        "jitter_cm": round(jit[0], 1),
        "jitter_gt_cm": round(jit[1], 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="thư mục chứa các SNGS-xxx")
    ap.add_argument("--pitch-weights", required=True)
    ap.add_argument("--pitch-imgsz", type=int, default=1280)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--out", default="outputs/pitch_eval")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    root = Path(args.data_root)
    clips = args.clips or sorted(p.name for p in root.iterdir() if (p / "Labels-GameState.json").exists())
    detector = PitchKeypointDetector(args.pitch_weights, imgsz=args.pitch_imgsz, device=args.device)
    cache_dir = Path(args.out) / "kpt_cache" / f"{Path(args.pitch_weights).stem}_{args.pitch_imgsz}"

    data = {}
    for clip in clips:
        people, n = load_gsr(root / clip)
        xy, conf = detect_keypoints(root / clip, n, detector, cache_dir / f"{clip}.npz")
        data[clip] = (xy, conf, people, n)

    # xác định chiều trục của GSR trên clip đầu bằng H thô
    xy, conf, people, n = data[clips[0]]
    sign_err = {s: np.median(run_variant(xy, conf, people, n, None, np.array(s), args.min_conf)[0]) for s in AXIS_SIGNS}
    signs = np.array(min(sign_err, key=sign_err.get))
    print("Sai số trung vị theo chiều trục (m):", {str(k): round(float(v), 2) for k, v in sign_err.items()}, "-> chọn", tuple(signs))

    variants = {
        "H thô": None,
        "One-Euro min_cutoff 0.5": dict(min_cutoff=0.5, beta=0.02),
        "One-Euro min_cutoff 0.2": dict(min_cutoff=0.2, beta=0.02),
        "One-Euro min_cutoff 1.0": dict(min_cutoff=1.0, beta=0.02),
    }
    results = {}
    for name, params in variants.items():
        all_err, all_cov, all_n, all_tracks = [], 0, 0, {}
        for clip, (xy, conf, people, n) in data.items():
            sm = HomographySmoother(**params) if params else None
            err, cov, tracks = run_variant(xy, conf, people, n, sm, signs, args.min_conf)
            all_err.append(err)
            all_cov += cov
            all_n += n
            all_tracks.update({(clip, t): v for t, v in tracks.items()})
        results[name] = summarize(np.concatenate(all_err), all_cov, all_n, jitter_cm(all_tracks))

    print(f"\n{len(clips)} clip GSR: {', '.join(clips)}\n")
    print("| Homography | Phủ frame | Sai số TB (m) | Trung vị (m) | ≤ 1 m | ≤ 2 m | Rung (cm/frame²) |")
    print("|---|---|---|---|---|---|---|")
    for name, r in results.items():
        print(f"| {name} | {r['coverage']:.0%} | {r['mean_m']} | {r['median_m']} | {r['within_1m']:.0%} | {r['within_2m']:.0%} | {r['jitter_cm']} |")
    print(f"| *GT (tham chiếu)* | | | | | | *{results['H thô']['jitter_gt_cm']}* |")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"clips": clips, "axis_signs": signs.tolist(), "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
