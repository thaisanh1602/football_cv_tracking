"""Đo nhấp nháy (flicker) của mask người trên SoccerNet GSR: mask từng frame vs mask đã ổn định.

Mask lấy từ YOLO-seg pretrained (COCO person), gán cho cầu thủ theo box GT của GSR. Dùng box GT để chỉ đo
nhấp nháy của mask, không lẫn rung của detector / tracker. Bước YOLO-seg cần GPU: chạy trên Kaggle
(notebooks/kaggle_eval.ipynb); trên máy local chỉ nên thử --max-frames nhỏ với --device cpu.

    python scripts/eval_mask_flicker.py --data-root data/SoccerNet/gsr/test --seg-weights yolo11m-seg.pt
"""
import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.stabilize.masks import MaskStabilizer, PersonSegmenter, iou, match_masks, to_canvas  # noqa: E402

ALPHAS = [1.0, 0.7, 0.5, 0.3]  # 1.0 = mask gốc từng frame


def load_gsr_boxes(clip_dir: Path) -> tuple[dict, int]:
    """frame (0-based) -> (track ids, box xyxy) của người (bỏ bóng)."""
    d = json.load(open(clip_dir / "Labels-GameState.json", encoding="utf-8"))
    frame_of = {img["image_id"]: int(Path(img["file_name"]).stem) - 1 for img in d["images"]}
    per = defaultdict(list)
    for a in d["annotations"]:
        if a.get("supercategory") == "object" and a["attributes"].get("role") != "ball" and a.get("bbox_image"):
            b = a["bbox_image"]
            per[frame_of[a["image_id"]]].append((a["track_id"], b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]))
    out = {f: (np.array([r[0] for r in rows]), np.array([r[1:] for r in rows], float)) for f, rows in per.items()}
    return out, len(d["images"])


def collect(clip_dir: Path, seg, cache_file: Path, max_frames: int | None) -> dict:
    """track id -> list (frame, canvas uint8). Cache để thử nhiều mức ổn định mà không chạy lại YOLO-seg."""
    if cache_file.exists():
        return pickle.loads(cache_file.read_bytes())
    boxes, n = load_gsr_boxes(clip_dir)
    tracks = defaultdict(list)
    for i in tqdm(range(min(n, max_frames or n)), desc=clip_dir.name, leave=False):
        if i not in boxes:
            continue
        frame = cv2.imread(str(clip_dir / "img1" / f"{i + 1:06d}.jpg"))
        ids, xyxy = boxes[i]
        sb, sm = seg(frame)
        for tid, box, m in zip(ids, xyxy, match_masks(sb, sm, xyxy)):
            if m is not None:
                tracks[int(tid)].append((i, (to_canvas(m, box) * 255).astype(np.uint8)))
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(pickle.dumps(dict(tracks)))
    return dict(tracks)


def evaluate(all_tracks: list[dict], alpha: float) -> dict:
    flick, fid = [], []
    for tracks in all_tracks:
        for tid, seq in tracks.items():
            stab = MaskStabilizer(alpha=alpha)
            prev = None
            for f, c in seq:
                raw = c.astype(np.float32) / 255
                s = stab.update_canvas(f, tid, raw)
                fid.append(iou(s, raw))
                if prev is not None and f - prev[0] == 1:
                    flick.append(1 - iou(s, prev[1]))
                prev = (f, s)
    flick, fid = np.array(flick), np.array(fid)
    return {
        "flicker_median": round(float(np.median(flick)), 3),
        "flicker_p90": round(float(np.percentile(flick, 90)), 3),
        "fidelity_median": round(float(np.median(fid)), 3),
        "n_pairs": int(len(flick)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="thư mục chứa các SNGS-xxx")
    ap.add_argument("--seg-weights", default="yolo11m-seg.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--max-frames", type=int, default=None, help="chỉ lấy N frame đầu mỗi clip (thử nhanh)")
    ap.add_argument("--cache", default="outputs/mask_cache")
    ap.add_argument("--out", default="outputs/mask_eval")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    root = Path(args.data_root)
    clips = args.clips or sorted(p.name for p in root.iterdir() if (p / "Labels-GameState.json").exists())
    cache_dir = Path(args.cache) / f"{Path(args.seg_weights).stem}_{args.imgsz}_{args.max_frames or 'all'}"
    seg = None if all((cache_dir / f"{c}.pkl").exists() for c in clips) else PersonSegmenter(args.seg_weights, args.imgsz, device=args.device)
    all_tracks = [collect(root / c, seg, cache_dir / f"{c}.pkl", args.max_frames) for c in clips]

    results = {("mask gốc từng frame" if a == 1.0 else f"ổn định EMA α={a}"): evaluate(all_tracks, a) for a in ALPHAS}
    print(f"\n{len(clips)} clip GSR: {', '.join(clips)}. Flicker = 1 - IoU giữa 2 frame liên tiếp (căn theo box GT).\n")
    print("| Mask | Flicker trung vị | Flicker p90 | Độ trung thành với mask gốc (IoU) |")
    print("|---|---|---|---|")
    for name, r in results.items():
        print(f"| {name} | {r['flicker_median']} | {r['flicker_p90']} | {r['fidelity_median']} |")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"clips": clips, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
