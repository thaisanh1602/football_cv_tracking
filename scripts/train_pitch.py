"""Train YOLO-pose phát hiện 32 keypoint sân (chạy trên Kaggle).

    python scripts/train_pitch.py --data /kaggle/working/pitch/data_fixed.yaml --model yolo11m-pose.pt
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data.yaml từ prepare_pitch_dataset.py (đã có flip_idx đúng)")
    ap.add_argument("--model", default="yolo11m-pose.pt")
    ap.add_argument("--imgsz", type=int, default=1280)  # keypoint ở xa camera rất nhỏ
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--project", default="outputs/runs")
    ap.add_argument("--name", default="pitch32")
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
        mosaic=0.0,  # mosaic ghép 4 mảnh sân không liên quan -> phá hình học toàn cục mà keypoint dựa vào
        fliplr=0.5,  # an toàn vì flip_idx đã đúng
    )
    model.val(data=args.data, imgsz=args.imgsz, split="val")


if __name__ == "__main__":
    main()
