"""Train detector 4 class (chạy trên Kaggle, GPU local 4GB không đủ cho imgsz 1280).

    python scripts/train.py --data configs/data.yaml --model yolo11m.pt --imgsz 1280 --epochs 100
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "configs" / "data.yaml"))
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--imgsz", type=int, default=1280)  # 640 thì bóng chỉ còn vài pixel
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--project", default="outputs/runs")
    ap.add_argument("--name", default="det4")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        close_mosaic=10,
    )
    model.val(data=args.data, imgsz=args.imgsz, split="val")  # in mAP từng class -> ghi vào README


if __name__ == "__main__":
    main()
