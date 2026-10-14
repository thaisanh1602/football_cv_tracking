"""Tải vài clip SoccerNet GSR mà không tải cả file zip (test 9 GB).

Đọc mục lục zip qua HTTP range (remotezip), rồi với mỗi clip tải nguyên khối byte liền nhau của nó bằng
1 request (~200 MB) và tự tách file. Nhanh hơn nhiều so với 1 request cho mỗi ảnh.

    pip install remotezip
    python scripts/download_gsr.py --clips SNGS-116 SNGS-117 SNGS-118 --out data/SoccerNet/gsr/test
    python scripts/download_gsr.py --split train --n-clips 30 --out /tmp/gsr/train
"""
import argparse
import struct
import zlib
from pathlib import Path

import requests
from remotezip import RemoteZip
from tqdm import tqdm

URL = "https://huggingface.co/datasets/SoccerNet/SN-GSR-2025/resolve/main/{split}.zip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--clips", nargs="+", default=None)
    ap.add_argument("--n-clips", type=int, default=None, help="lấy N clip đầu của split (thay cho --clips)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    url = URL.format(split=args.split)

    with RemoteZip(url) as z:
        infos = z.infolist()
    clips = args.clips or sorted({f.filename.split("/")[0] for f in infos if f.filename.startswith("SNGS-")})[: args.n_clips]

    for clip in clips:
        entries = [f for f in infos if f.filename.startswith(clip + "/") and not f.is_dir()]
        if not entries:
            print(f"Không có {clip} trong {args.split}.zip")
            continue
        start = min(f.header_offset for f in entries)
        end = max(f.header_offset + 30 + len(f.filename.encode()) + 1024 + f.compress_size for f in entries)
        resp = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=60)
        resp.raise_for_status()
        buf = bytearray()
        with tqdm(total=end - start, unit="B", unit_scale=True, desc=clip) as bar:
            for chunk in resp.iter_content(1 << 20):
                buf += chunk
                bar.update(len(chunk))

        for f in entries:
            o = f.header_offset - start
            name_len, extra_len = struct.unpack("<HH", buf[o + 26 : o + 30])
            data = bytes(buf[o + 30 + name_len + extra_len : o + 30 + name_len + extra_len + f.compress_size])
            if f.compress_type == 8:
                data = zlib.decompress(data, -15)
            dst = Path(args.out) / f.filename
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
        print(f"{clip}: {len(entries)} file")


if __name__ == "__main__":
    main()
