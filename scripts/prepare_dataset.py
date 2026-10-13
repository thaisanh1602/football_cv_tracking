"""Chuyển dataset export từ Roboflow (YOLO format) sang cấu trúc + thứ tự class của configs/data.yaml.

Roboflow: <src>/{train,valid,test}/{images,labels}, class order ví dụ [ball, goalkeeper, player, referee]
Đích:     <dst>/images/{train,val,test}, <dst>/labels/{train,val,test}, class order theo CLASS_NAMES

    python scripts/prepare_dataset.py --src /kaggle/working/roboflow --dst /kaggle/working/dataset
"""
import argparse
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.label_mapping import CLASS_MAP  # noqa: E402

SPLITS = {"train": "train", "valid": "val", "test": "test"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)

    names = yaml.safe_load(open(src / "data.yaml", encoding="utf-8"))["names"]
    names = dict(enumerate(names)) if isinstance(names, list) else names
    remap = {int(i): CLASS_MAP[n] for i, n in names.items() if n in CLASS_MAP}
    print(f"Class gốc: {names}\nRemap: {remap}")

    for src_split, dst_split in SPLITS.items():
        if not (src / src_split).exists():
            continue
        img_out, lbl_out = dst / "images" / dst_split, dst / "labels" / dst_split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        n = 0
        for img in (src / src_split / "images").iterdir():
            shutil.copy2(img, img_out / img.name)
            lbl = src / src_split / "labels" / f"{img.stem}.txt"
            lines = []
            if lbl.exists():
                for line in lbl.read_text().splitlines():
                    parts = line.split()
                    if parts and int(parts[0]) in remap:
                        lines.append(" ".join([str(remap[int(parts[0])]), *parts[1:]]))
            (lbl_out / f"{img.stem}.txt").write_text("\n".join(lines))
            n += 1
        print(f"{src_split} -> {dst_split}: {n} ảnh")


if __name__ == "__main__":
    main()
