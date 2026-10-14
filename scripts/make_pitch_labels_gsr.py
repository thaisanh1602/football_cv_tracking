"""Sinh nhãn 32 keypoint sân (YOLO-pose) từ SoccerNet GSR, để train keypoint trên dữ liệu cùng domain.

Mỗi frame GSR có: vị trí chân thật của từng người (ảnh <-> sân, mét) và các vạch sân vẽ tay trên ảnh.
Fit homography ảnh -> sân bằng DLT tuyến tính gộp 2 loại ràng buộc:
  - chân người: điểm ảnh (u, v) <-> điểm sân (X, Y)               -> 2 phương trình
  - điểm nằm trên vạch đã biết (vd biên dọc Y = 0, vòng cấm X = 16.5) -> 1 phương trình
Chỉ dùng chân người thì phải ngoại suy ra góc sân xa (cầu thủ tụ một vùng); vạch sân phủ đúng các vùng đó.
Rồi chiếu 32 điểm của src/pitch/config.py vào ảnh: điểm nằm trong khung hình -> nhãn visible.
Frame có sai số khớp lớn (nhãn GSR không nhất quán) bị bỏ.

    python scripts/download_gsr.py --split train --clips SNGS-060 SNGS-061 --out data/SoccerNet/gsr/train
    python scripts/make_pitch_labels_gsr.py --gsr-root data/SoccerNet/gsr/train --out data/pitch_gsr --stride 10
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pitch.config import PITCH  # noqa: E402
from src.pitch.homography import project  # noqa: E402

L, W = PITCH.length, PITCH.width
PB_TOP, PB_BOT = (W - PITCH.penalty_box_width) / 2, (W + PITCH.penalty_box_width) / 2
GB_TOP, GB_BOT = (W - PITCH.goal_box_width) / 2, (W + PITCH.goal_box_width) / 2
# vạch GSR -> (trục, giá trị) trên sân (gốc ở góc). GSR: "top" là biên xa camera (y âm khi gốc ở tâm).
# Không dùng vòng tròn (phi tuyến) và khung thành (không nằm trên mặt sân).
LINES = {
    "Side line top": ("Y", 0.0),
    "Side line bottom": ("Y", W),
    "Side line left": ("X", 0.0),
    "Side line right": ("X", L),
    "Middle line": ("X", L / 2),
    "Big rect. left main": ("X", PITCH.penalty_box_length),
    "Big rect. left top": ("Y", PB_TOP),
    "Big rect. left bottom": ("Y", PB_BOT),
    "Big rect. right main": ("X", L - PITCH.penalty_box_length),
    "Big rect. right top": ("Y", PB_TOP),
    "Big rect. right bottom": ("Y", PB_BOT),
    "Small rect. left main": ("X", PITCH.goal_box_length),
    "Small rect. left top": ("Y", GB_TOP),
    "Small rect. left bottom": ("Y", GB_BOT),
    "Small rect. right main": ("X", L - PITCH.goal_box_length),
    "Small rect. right top": ("Y", GB_TOP),
    "Small rect. right bottom": ("Y", GB_BOT),
}


def normalizer(pts: np.ndarray) -> np.ndarray:
    """Chuẩn hoá Hartley: dời tâm về 0, khoảng cách TB sqrt(2) -> DLT ổn định số học."""
    c = pts.mean(axis=0)
    s = np.sqrt(2) / max(np.linalg.norm(pts - c, axis=1).mean(), 1e-9)
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])


def fit_homography(point_pairs, line_points) -> np.ndarray | None:
    """point_pairs: list ((u, v), (X, Y)); line_points: list ((u, v), "X"|"Y", giá trị). Trả về H ảnh -> sân."""
    if len(point_pairs) + len(line_points) < 8:
        return None
    img = np.array([p for p, _ in point_pairs] + [p for p, _, _ in line_points], dtype=np.float64)
    Ti = normalizer(img)
    Tp = normalizer(np.array([[L / 2, W / 2], [0, 0], [L, W]]))  # chuẩn hoá cố định cho toạ độ sân
    to_h = lambda p, T: T @ np.array([p[0], p[1], 1.0])  # noqa: E731

    rows = []
    for (uv, XY) in point_pairs:
        u, v, w = to_h(uv, Ti)
        X, Y, _ = to_h(XY, Tp)
        rows.append([u, v, w, 0, 0, 0, -X * u, -X * v, -X * w])
        rows.append([0, 0, 0, u, v, w, -Y * u, -Y * v, -Y * w])
    for uv, axis, val in line_points:
        u, v, w = to_h(uv, Ti)
        # vạch X = val (hoặc Y = val) sau chuẩn hoá vẫn là X' = val' vì Tp chỉ co giãn + tịnh tiến
        val_n = to_h((val, 0) if axis == "X" else (0, val), Tp)[0 if axis == "X" else 1]
        r = [0.0] * 9
        k = 0 if axis == "X" else 3
        r[k : k + 3] = [u, v, w]
        r[6:9] = [-val_n * u, -val_n * v, -val_n * w]
        rows.append(r)
    _, _, vt = np.linalg.svd(np.array(rows))
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Tp) @ Hn @ Ti
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else None


def frame_constraints(anns, width, height):
    pairs, lines = [], []
    for a in anns:
        if a.get("supercategory") == "object" and a.get("bbox_pitch") and a["attributes"].get("role") != "ball":
            b, p = a["bbox_image"], a["bbox_pitch"]
            pairs.append(((b["x"] + b["w"] / 2, b["y"] + b["h"]), (p["x_bottom_middle"] + L / 2, p["y_bottom_middle"] + W / 2)))
        elif a.get("supercategory") == "pitch":
            for name, pts in a.get("lines", {}).items():
                if name in LINES:
                    axis, val = LINES[name]
                    lines += [((q["x"] * width, q["y"] * height), axis, val) for q in pts]
    return pairs, lines


def residuals_m(H, pairs, lines):
    """Sai số khớp trên mặt sân (m): khoảng cách chân người tới GT và điểm-trên-vạch tới vạch."""
    res = []
    if pairs:
        uv, XY = map(np.array, zip(*pairs))
        res += list(np.linalg.norm(project(H, uv) - XY, axis=1))
    if lines:
        uv = np.array([p for p, _, _ in lines])
        proj = project(H, uv)
        res += [abs(q[0 if axis == "X" else 1] - val) for q, (_, axis, val) in zip(proj, lines)]
    return np.array(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsr-root", required=True, help="thư mục chứa các SNGS-xxx (có Labels-GameState.json, img1/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=10, help="lấy mỗi N frame (frame liền nhau gần như trùng nhau)")
    ap.add_argument("--val-clips", type=int, default=2, help="số clip cuối dành cho val")
    ap.add_argument("--max-residual-m", type=float, default=1.0, help="bỏ frame có sai số khớp trung vị lớn hơn")
    ap.add_argument("--min-visible", type=int, default=4)
    args = ap.parse_args()

    root, out = Path(args.gsr_root), Path(args.out)
    clips = sorted(p.name for p in root.iterdir() if (p / "Labels-GameState.json").exists())
    split_of = {c: ("val" if i >= len(clips) - args.val_clips else "train") for i, c in enumerate(clips)}
    stats = {"frames": 0, "kept": 0, "no_fit": 0, "bad_fit": 0, "few_visible": 0, "visible": []}

    for clip in tqdm(clips, desc="Clips"):
        d = json.load(open(root / clip / "Labels-GameState.json", encoding="utf-8"))
        anns_of = {}
        for a in d["annotations"]:
            anns_of.setdefault(a["image_id"], []).append(a)
        split = split_of[clip]
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

        for img in d["images"][:: args.stride]:
            stats["frames"] += 1
            w, h = img["width"], img["height"]
            pairs, lines = frame_constraints(anns_of.get(img["image_id"], []), w, h)
            H = fit_homography(pairs, lines)
            if H is None:
                stats["no_fit"] += 1
                continue
            if np.median(residuals_m(H, pairs, lines)) > args.max_residual_m:
                stats["bad_fit"] += 1
                continue

            G = np.linalg.inv(H)  # sân -> ảnh
            homog = np.c_[PITCH.vertices, np.ones(32)] @ G.T
            kp = project(G, PITCH.vertices)
            vis = (homog[:, 2] > 0) & (kp[:, 0] >= 0) & (kp[:, 0] < w) & (kp[:, 1] >= 0) & (kp[:, 1] < h)
            if vis.sum() < args.min_visible:
                stats["few_visible"] += 1
                continue
            stats["kept"] += 1
            stats["visible"].append(int(vis.sum()))

            # box của object "pitch" = khung bao các keypoint nhìn thấy
            x1, y1 = kp[vis].min(axis=0)
            x2, y2 = kp[vis].max(axis=0)
            box = [(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, max(x2 - x1, 1) / w, max(y2 - y1, 1) / h]
            kpts = []
            for (x, y), v in zip(kp, vis):
                kpts += [x / w, y / h, 2] if v else [0, 0, 0]
            name = f"{clip}_{Path(img['file_name']).stem}"
            (out / "labels" / split / f"{name}.txt").write_text(
                " ".join(["0"] + [f"{x:.6f}" for x in box] + [f"{x:.6f}" if i % 3 != 2 else str(int(x)) for i, x in enumerate(kpts)])
            )
            shutil.copy2(root / clip / "img1" / img["file_name"], out / "images" / split / f"{name}.jpg")

    (out / "data.yaml").write_text(
        yaml.safe_dump(
            {"path": str(out.resolve()), "train": "images/train", "val": "images/val", "kpt_shape": [32, 3], "flip_idx": PITCH.flip_idx, "names": {0: "pitch"}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    vis = np.array(stats.pop("visible"))
    print(
        f"{stats['frames']} frame: giữ {stats['kept']}, không fit được {stats['no_fit']}, khớp kém {stats['bad_fit']}, "
        f"< {args.min_visible} keypoint {stats['few_visible']}. Keypoint nhìn thấy / ảnh: trung vị {int(np.median(vis)) if len(vis) else 0}"
    )
    print(f"Đã ghi {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
