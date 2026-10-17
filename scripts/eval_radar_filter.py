"""Lọc vị trí từng cầu thủ trên sân (sau homography): đường đánh đổi rung - sai số trên SoccerNet GSR.

Dùng track ID và điểm chân GT trên ảnh (như eval_pitch.py) để chỉ đo homography + bộ lọc. Bộ lọc chạy nhân
quả nên sai số đã gồm độ trễ. Rung = |gia tốc rời rạc| trên radar (cm/frame²), GT ~0.5.

    python scripts/eval_radar_filter.py --data-root data/SoccerNet/gsr/test --pitch-weights outputs/pitch_best.pt
"""
import argparse
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_pitch import detect_keypoints, jitter_cm, load_gsr, run_variant  # noqa: E402
from src.pitch.homography import HomographySmoother  # noqa: E402
from src.pitch.keypoints import PitchKeypointDetector  # noqa: E402
from src.stabilize.overlay_filter import EMA, KalmanCV, OneEuro2D, TrackFilterBank  # noqa: E402

FILTERS = {
    "không lọc": None,
    "EMA α=0.3": partial(EMA, alpha=0.3),
    "EMA α=0.1": partial(EMA, alpha=0.1),
    "One-Euro (0.3, β 0.1)": partial(OneEuro2D, min_cutoff=0.3, beta=0.1),
    "One-Euro (0.1, β 0.3)": partial(OneEuro2D, min_cutoff=0.1, beta=0.3),
    "Kalman σa=4": partial(KalmanCV, accel_std=4.0),
    "Kalman σa=2": partial(KalmanCV, accel_std=2.0),
    "Kalman σa=1": partial(KalmanCV, accel_std=1.0),
    "Kalman σa=0.5": partial(KalmanCV, accel_std=0.5),
}


def apply_filter(tracks: dict, make) -> dict:
    """tracks: key -> list (frame, dự đoán, GT). Lọc dự đoán theo từng track, giữ nguyên GT."""
    if make is None:
        return tracks
    out = {}
    for key, seq in tracks.items():
        bank = TrackFilterBank(make)
        out[key] = [(f, bank.update(f, [0], [p])[0], g) for f, p, g in seq]
    return out


def summarize(tracks: dict) -> dict:
    err = np.concatenate([[np.linalg.norm(p - g) for _, p, g in seq] for seq in tracks.values() if seq])
    jit = jitter_cm(tracks)
    return {
        "median_m": round(float(np.median(err)), 2),
        "mean_m": round(float(np.mean(err)), 2),
        "within_2m": round(float((err <= 2).mean()), 3),
        "jitter_median_cm": round(jit[1], 1),
        "jitter_mean_cm": round(jit[0], 1),
        "jitter_gt_cm": round(jit[2], 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--pitch-weights", required=True)
    ap.add_argument("--pitch-imgsz", type=int, default=1280)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--kpt-cache", default="outputs/pitch_eval/kpt_cache")
    ap.add_argument("--out", default="outputs/radar_filter_eval")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    root = Path(args.data_root)
    clips = args.clips or sorted(p.name for p in root.iterdir() if (p / "Labels-GameState.json").exists())
    cache_dir = Path(args.kpt_cache) / f"{Path(args.pitch_weights).stem}_{args.pitch_imgsz}"
    detector = None if all((cache_dir / f"{c}.npz").exists() for c in clips) else PitchKeypointDetector(args.pitch_weights, imgsz=args.pitch_imgsz, device=args.device)

    tracks = {"H thô": {}, "H mượt": {}}
    for clip in clips:
        people, n = load_gsr(root / clip)
        xy, conf = detect_keypoints(root / clip, n, detector, cache_dir / f"{clip}.npz")
        for name, sm in (("H thô", None), ("H mượt", HomographySmoother())):
            _, _, tr = run_variant(xy, conf, people, n, sm, np.array([1, 1]), 0.5)
            tracks[name].update({(clip, t): v for t, v in tr.items()})

    results = {f"{h} + {f}": summarize(apply_filter(tracks[h], make)) for h in tracks for f, make in FILTERS.items()}

    gt_jit = next(iter(results.values()))["jitter_gt_cm"]
    print(f"\n{len(clips)} clip GSR: {', '.join(clips)}. Bộ lọc nhân quả, sai số đã gồm độ trễ. Rung GT = {gt_jit} cm/frame².\n")
    print("| Homography + lọc vị trí | Sai số trung vị (m) | ≤ 2 m | Rung trung vị (cm/frame²) | Rung TB |")
    print("|---|---|---|---|---|")
    for name, r in results.items():
        print(f"| {name} | {r['median_m']} | {r['within_2m']:.0%} | {r['jitter_median_cm']} | {r['jitter_mean_cm']} |")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"clips": clips, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
