"""Kiểm tra dataset keypoint sân export từ Roboflow (YOLOv8-pose) và ghi data.yaml đã sửa flip_idx.

Lật ảnh ngang (augment fliplr của Ultralytics) phải đổi tên keypoint trái <-> phải. Nếu flip_idx sai,
model học nhầm "góc trái" thành "góc phải" mà mAP vẫn có thể trông ổn -> homography sai hoàn toàn.
flip_idx ở đây được tính từ toạ độ sân (src/pitch/config.py), không tin flip_idx của file export.

    python scripts/prepare_pitch_dataset.py --src /kaggle/working/pitch --out /kaggle/working/pitch/data_fixed.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pitch.config import PITCH  # noqa: E402

SPLITS = {"train": "train", "val": "valid", "test": "test"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="thư mục export Roboflow (có data.yaml, train/, valid/)")
    ap.add_argument("--out", required=True, help="đường dẫn data.yaml mới")
    args = ap.parse_args()
    src = Path(args.src).resolve()

    cfg = yaml.safe_load(open(src / "data.yaml", encoding="utf-8"))
    print(f"names: {cfg.get('names')}, kpt_shape: {cfg.get('kpt_shape')}")
    if list(cfg.get("kpt_shape", [])) != [32, 3]:
        raise SystemExit("Cần 32 keypoint dạng (x, y, visibility) để khớp src/pitch/config.py")
    if cfg.get("flip_idx") is not None and list(cfg["flip_idx"]) != PITCH.flip_idx:
        print(f"⚠ flip_idx của file export khác flip_idx tính từ toạ độ sân:\n  export: {cfg['flip_idx']}\n  đúng:   {PITCH.flip_idx}")

    # thống kê: mỗi ảnh thấy bao nhiêu keypoint, keypoint nào hiếm
    for split, folder in SPLITS.items():
        labels = list((src / folder / "labels").glob("*.txt")) if (src / folder / "labels").exists() else []
        if not labels:
            continue
        visible = []
        for f in labels:
            for line in f.read_text().splitlines():
                kpts = np.array(line.split()[5:], dtype=float).reshape(-1, 3)
                visible.append(kpts[:, 2] > 0)
        visible = np.array(visible)
        per_img = visible.sum(axis=1)
        rare = np.argsort(visible.mean(axis=0))[:5] + 1
        print(f"{split}: {len(labels)} ảnh, keypoint thấy được / ảnh: trung vị {int(np.median(per_img))}, "
              f"< 4 điểm: {(per_img < 4).mean():.0%} ảnh; 5 keypoint hiếm nhất (1-based): {rare.tolist()}")

    out_cfg = {
        "path": str(src),
        **{s: f"{f}/images" for s, f in SPLITS.items() if (src / f / "images").exists()},
        "kpt_shape": [32, 3],
        "flip_idx": PITCH.flip_idx,
        "names": cfg["names"],
    }
    Path(args.out).write_text(yaml.safe_dump(out_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Đã ghi {args.out}")


if __name__ == "__main__":
    main()
