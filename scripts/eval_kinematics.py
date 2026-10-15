"""Đo sai số tốc độ (km/h) và quãng đường (%) trên SoccerNet GSR.

Dùng track ID và điểm chân GT trên ảnh, chiếu qua H của mình -> chỉ đo lỗi của homography + cách tính
động học, không lẫn lỗi detector/tracker. GT: vị trí thật trên sân, xử lý bằng cùng cách tính.

So sánh:
- "2 frame liên tiếp": tốc độ = |Δvị trí| / Δt, quãng đường = tổng |Δvị trí| (cách ngây thơ)
- Savitzky-Golay 2 s + cắt đoạn (src/pitch/kinematics.py), trên H thô và H đã làm mượt

    python scripts/eval_kinematics.py --data-root data/SoccerNet/gsr/test --pitch-weights outputs/pitch_best.pt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_pitch import detect_keypoints, load_gsr, run_variant  # noqa: E402
from src.pitch.homography import HomographySmoother  # noqa: E402
from src.pitch.keypoints import PitchKeypointDetector  # noqa: E402
from src.pitch.kinematics import split_segments, track_kinematics  # noqa: E402

FPS = 25


def naive(frames, xy):
    """Tốc độ từ 2 frame liên tiếp (chỉ khi frame liền nhau), quãng đường cộng thẳng."""
    speed, dist = {}, 0.0
    for i in range(1, len(frames)):
        if frames[i] - frames[i - 1] == 1:
            step = float(np.linalg.norm(xy[i] - xy[i - 1]))
            speed[int(frames[i])] = step * FPS
            dist += step
    return {"speed": speed, "distance": dist}


def evaluate(tracks, method) -> dict:
    """tracks: (clip, track) -> list (frame, dự đoán, GT). So từng track trên đúng các frame có dự đoán."""
    sp_err, pred_max, dist_pred, dist_gt, n_segments = [], [], 0.0, 0.0, 0
    for seq in tracks.values():
        if len(seq) < 2 * FPS:  # bỏ track quá ngắn (< 2 s)
            continue
        f = np.array([s[0] for s in seq])
        pred = np.array([s[1] for s in seq])
        gt = np.array([s[2] for s in seq])
        kp = method(f, pred)
        # GT tính bằng cùng cách, trên cùng các đoạn mà dự đoán dùng (cắt đoạn theo dự đoán)
        kg = {"speed": {}, "distance": 0.0}
        for seg in split_segments(f, pred, FPS) if method is not naive else [slice(None)]:
            r = method(f[seg], gt[seg])
            kg["speed"].update(r["speed"])
            kg["distance"] += r["distance"]
        n_segments += kp.get("segments", 1)
        common = sorted(set(kp["speed"]) & set(kg["speed"]))
        if not common:
            continue
        sp_err += [abs(kp["speed"][c] - kg["speed"][c]) * 3.6 for c in common]
        pred_max.append(max(kp["speed"].values()) * 3.6)
        dist_pred += kp["distance"]
        dist_gt += kg["distance"]
    sp_err, pred_max = np.array(sp_err), np.array(pred_max)
    return {
        "speed_mae_kmh": round(float(sp_err.mean()), 2),
        "speed_median_err_kmh": round(float(np.median(sp_err)), 2),
        "tracks_max_over_36kmh": round(float((pred_max > 36).mean()), 3),
        "distance_err_pct": round(100 * (dist_pred - dist_gt) / dist_gt, 1),
        "segments_per_track": round(n_segments / max(len(pred_max), 1), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--pitch-weights", required=True)
    ap.add_argument("--pitch-imgsz", type=int, default=1280)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--out", default="outputs/kinematics_eval")
    ap.add_argument("--kpt-cache", default="outputs/pitch_eval/kpt_cache", help="dùng chung cache với eval_pitch.py")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    root = Path(args.data_root)
    clips = args.clips or sorted(p.name for p in root.iterdir() if (p / "Labels-GameState.json").exists())
    detector = PitchKeypointDetector(args.pitch_weights, imgsz=args.pitch_imgsz, device=args.device)
    cache_dir = Path(args.kpt_cache) / f"{Path(args.pitch_weights).stem}_{args.pitch_imgsz}"

    tracks = {"H thô": {}, "H mượt": {}}
    for clip in clips:
        people, n = load_gsr(root / clip)
        xy, conf = detect_keypoints(root / clip, n, detector, cache_dir / f"{clip}.npz")
        for name, sm in (("H thô", None), ("H mượt", HomographySmoother())):
            _, _, tr = run_variant(xy, conf, people, n, sm, np.array([1, 1]), 0.5)
            tracks[name].update({(clip, t): v for t, v in tr.items()})

    variants = {
        "2 frame liên tiếp, H thô": (naive, "H thô"),
        "2 frame liên tiếp, H mượt": (naive, "H mượt"),
        "Savitzky-Golay 2 s + cắt đoạn, H thô": (track_kinematics, "H thô"),
        "Savitzky-Golay 2 s + cắt đoạn, H mượt": (track_kinematics, "H mượt"),
    }
    results = {name: evaluate(tracks[h], m) for name, (m, h) in variants.items()}

    print(f"\n{len(clips)} clip GSR: {', '.join(clips)}. So với GT trên cùng frame, track >= 2 s.\n")
    print("| Cách tính | Sai số tốc độ TB (km/h) | Trung vị (km/h) | Track có max > 36 km/h | Sai số quãng đường | Đoạn / track |")
    print("|---|---|---|---|---|---|")
    for name, r in results.items():
        print(f"| {name} | {r['speed_mae_kmh']} | {r['speed_median_err_kmh']} | {r['tracks_max_over_36kmh']:.0%} | {r['distance_err_pct']:+.1f}% | {r['segments_per_track']} |")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"clips": clips, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
