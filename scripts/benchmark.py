"""Đo FPS thật ở 1080p: độ trễ detector theo định dạng (PyTorch / ONNX / TensorRT) và kích thước ảnh, kèm mức
khớp với model gốc; rồi thời gian từng bước của pipeline đầy đủ (không gồm vẽ / ghi video).

Mức khớp: coi detection của model đầu tiên trong --det (vd best.pt @1280) là tham chiếu, ghép IoU >= 0.5 ->
recall (tìm lại được bao nhiêu % detection tham chiếu) và precision. Export đúng thì ~100%; giảm imgsz thì
người ở xa và bóng bị mất trước.

imgsz của file ONNX / engine lấy từ tên file (best_960.onnx -> 960, do export_models.py đặt).

    python scripts/benchmark.py --source outputs/SNGS-118.mp4 --det outputs/best.pt outputs/export/best_1280.onnx \\
        outputs/export/best_640.onnx --pitch outputs/export/pitch_best_1280.onnx --device cpu --frames 100
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import supervision as sv
from tqdm import tqdm

# Ultralytics tự `pip install onnxruntime-gpu` khi thấy CUDA, kể cả lúc đo CPU, và có thể treo hàng chục phút
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.detection.detector import Detector, split_people_ball  # noqa: E402
from src.tracking.trackers import PeopleTracker  # noqa: E402
from src.utils.label_mapping import BALL_ID  # noqa: E402


def imgsz_of(path: str, default: int) -> int:
    m = re.search(r"_(\d{3,4})\.(onnx|engine)$", path)
    return int(m.group(1)) if m else default


def match_rate(ref: sv.Detections, det: sv.Detections) -> tuple[int, int, int]:
    """(số khớp, số tham chiếu, số dự đoán), ghép tham lam theo IoU >= 0.5 cùng class."""
    if len(ref) == 0 or len(det) == 0:
        return 0, len(ref), len(det)
    iou = sv.box_iou_batch(ref.xyxy, det.xyxy)
    iou[ref.class_id[:, None] != det.class_id[None, :]] = 0
    matched = 0
    while iou.size and iou.max() >= 0.5:
        r, c = np.unravel_index(iou.argmax(), iou.shape)
        matched += 1
        iou[r, :] = 0
        iou[:, c] = 0
    return matched, len(ref), len(det)


def bench_detector(weights, frames, imgsz, device, ref=None, warmup=5):
    det = Detector(weights, imgsz=imgsz, device=device)
    for f in frames[:warmup]:
        det(f)
    times, outs = [], []
    for f in tqdm(frames, desc=Path(weights).name, leave=False):
        t0 = time.perf_counter()
        outs.append(det(f))
        times.append(time.perf_counter() - t0)
    r = {"weights": Path(weights).name, "imgsz": imgsz, "ms": round(1000 * float(np.median(times)), 1)}
    r["fps"] = round(1000 / r["ms"], 1)
    if ref is not None:
        for name, mask in (("people", lambda d: d.class_id != BALL_ID), ("ball", lambda d: d.class_id == BALL_ID)):
            m = np.sum([match_rate(a[mask(a)], b[mask(b)]) for a, b in zip(ref, outs)], axis=0)
            r[f"{name}_recall"] = round(m[0] / max(m[1], 1), 3)
            r[f"{name}_precision"] = round(m[0] / max(m[2], 1), 3)
    return r, outs


def bench_pipeline(weights, imgsz, pitch, pitch_imgsz, frames_iter, n, device):
    """Thời gian từng bước, ms/frame (trung vị)."""
    from src.pitch.homography import PitchMapper
    from src.pitch.keypoints import PitchKeypointDetector
    from src.stabilize.overlay_filter import KalmanCV, TrackFilterBank

    det = Detector(weights, imgsz=imgsz, device=device)
    tracker = PeopleTracker(str(ROOT / "configs" / "botsort_tuned.yaml"))
    mapper = PitchMapper(PitchKeypointDetector(pitch, imgsz=pitch_imgsz, device=device)) if pitch else None
    kalman = TrackFilterBank(lambda: KalmanCV(accel_std=2.0))
    t = {k: [] for k in ("decode", "detect", "track", "pitch", "radar_filter", "total")}
    it = iter(frames_iter)
    for i in tqdm(range(n), desc="Pipeline", leave=False):
        t0 = time.perf_counter()
        frame = next(it)
        t1 = time.perf_counter()
        people, _ = split_people_ball(det(frame))
        t2 = time.perf_counter()
        people = tracker.update(people, frame)
        t3 = time.perf_counter()
        feet = None
        if mapper and mapper.update(frame) is not None and len(people):
            feet = mapper.to_pitch(people.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))
        t4 = time.perf_counter()
        if feet is not None:
            kalman.update(i, people.tracker_id, feet)
        t5 = time.perf_counter()
        for k, a, b in (("decode", t0, t1), ("detect", t1, t2), ("track", t2, t3), ("pitch", t3, t4), ("radar_filter", t4, t5), ("total", t0, t5)):
            t[k].append(b - a)
    res = {k: round(1000 * float(np.median(v[3:])), 1) for k, v in t.items()}  # bỏ vài frame đầu (khởi động)
    res["fps"] = round(1000 / res["total"], 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video 1080p")
    ap.add_argument("--det", nargs="+", required=True, help="model đầu tiên = tham chiếu cho mức khớp")
    ap.add_argument("--imgsz", type=int, default=1280, help="imgsz cho file .pt (ONNX/engine lấy từ tên file)")
    ap.add_argument("--pitch", default=None, help="model keypoint sân cho đo pipeline")
    ap.add_argument("--pipeline-det", default=None, help="detector dùng cho đo pipeline (mặc định: model cuối của --det)")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--device", default=None, help="cpu hoặc 0")
    ap.add_argument("--out", default="outputs/benchmark")
    args = ap.parse_args()

    frames = list(sv.get_video_frames_generator(args.source, end=args.frames))
    print(f"{len(frames)} frame {frames[0].shape[1]}x{frames[0].shape[0]}, device={args.device or 'auto'}")

    det_results, ref = [], None
    for w in args.det:
        r, outs = bench_detector(w, frames, imgsz_of(w, args.imgsz), args.device, ref)
        ref = ref or outs
        det_results.append(r)

    print("\n| Detector | imgsz | ms/frame | FPS | Người: recall / precision | Bóng: recall / precision |")
    print("|---|---|---|---|---|---|")
    for r in det_results:
        p = f"{r['people_recall']:.1%} / {r['people_precision']:.1%}" if "people_recall" in r else "tham chiếu"
        b = f"{r['ball_recall']:.1%} / {r['ball_precision']:.1%}" if "ball_recall" in r else "tham chiếu"
        print(f"| {r['weights']} | {r['imgsz']} | {r['ms']} | {r['fps']} | {p} | {b} |")

    pipe = None
    pdet = args.pipeline_det or args.det[-1]
    if args.pitch:
        pipe = bench_pipeline(pdet, imgsz_of(pdet, args.imgsz), args.pitch, imgsz_of(args.pitch, 1280),
                              sv.get_video_frames_generator(args.source), min(args.frames, 300), args.device)
        print(f"\nPipeline ({Path(pdet).name} + {Path(args.pitch).name}), ms/frame:")
        print("| Đọc frame | Detect | Track | Keypoint sân + H | Kalman radar | Tổng | FPS |")
        print("|---|---|---|---|---|---|---|")
        print(f"| {pipe['decode']} | {pipe['detect']} | {pipe['track']} | {pipe['pitch']} | {pipe['radar_filter']} | {pipe['total']} | **{pipe['fps']}** |")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = (args.device or "auto").replace(":", "")
    (out / f"summary_{tag}.json").write_text(json.dumps({"source": args.source, "device": args.device, "detector": det_results, "pipeline": pipe}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
