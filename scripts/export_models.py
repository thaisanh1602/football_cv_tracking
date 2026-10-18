"""Export detector / keypoint sân sang ONNX (chạy CPU, HF Spaces) và TensorRT (GPU NVIDIA, Turing trở lên).

TensorRT engine gắn với đúng loại GPU lúc build (engine build trên Kaggle T4 chỉ chạy trên T4), nên chỉ dùng
để đo FPS GPU; bản deploy CPU dùng ONNX. Mỗi imgsz ra 1 file riêng vì ONNX / engine có kích thước đầu vào cố định.

    python scripts/export_models.py --weights outputs/best.pt outputs/pitch_best.pt --formats onnx engine --imgsz 1280 960 640
"""
import argparse
import shutil
import tempfile
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--formats", nargs="+", default=["onnx"], choices=["onnx", "engine"])
    ap.add_argument("--imgsz", nargs="+", type=int, default=[1280])
    ap.add_argument("--out", default="outputs/export")
    ap.add_argument("--device", default=None, help="engine cần GPU, vd 0")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp())
    for w in args.weights:
        # Ultralytics ghi file export cạnh file .pt; chép sang thư mục ghi được trước
        # (vd /kaggle/input là read-only)
        src = work / Path(w).name
        shutil.copy2(w, src)
        for fmt in args.formats:
            for sz in args.imgsz:
                kw = dict(format=fmt, imgsz=sz, device=args.device)
                if fmt == "onnx":
                    kw |= dict(simplify=True, opset=17)
                else:
                    kw |= dict(half=True)  # FP16 trên tensor core
                path = Path(YOLO(str(src)).export(**kw))
                dst = out / f"{Path(w).stem}_{sz}{path.suffix}"
                shutil.move(str(path), dst)
                print(f"{w} -> {dst} ({dst.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
