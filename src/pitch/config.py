"""Cấu hình sân theo kích thước chuẩn FIFA (mét), giữ đúng thứ tự 32 keypoint của dataset
Roboflow "football-field-detection" (giống SoccerPitchConfiguration của roboflow/sports).

Không dùng thẳng config của roboflow/sports vì nó là sân 120 x 70 m, vòng cấm sâu 20.15 m
(chuẩn 16.5 m) -> tốc độ / quãng đường tính ra sẽ lệch 10-20%.

Hệ tọa độ: gốc ở góc sân, x dọc chiều dài [0, 105], y dọc chiều rộng [0, 68].
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PitchConfig:
    length: float = 105.0
    width: float = 68.0
    penalty_box_width: float = 40.32
    penalty_box_length: float = 16.5
    goal_box_width: float = 18.32
    goal_box_length: float = 5.5
    centre_circle_radius: float = 9.15
    penalty_spot_distance: float = 11.0

    @property
    def vertices(self) -> np.ndarray:
        L, W = self.length, self.width
        pw, pl = self.penalty_box_width, self.penalty_box_length
        gw, gl = self.goal_box_width, self.goal_box_length
        r, s = self.centre_circle_radius, self.penalty_spot_distance
        return np.array(
            [
                (0, 0),  # 1
                (0, (W - pw) / 2),  # 2
                (0, (W - gw) / 2),  # 3
                (0, (W + gw) / 2),  # 4
                (0, (W + pw) / 2),  # 5
                (0, W),  # 6
                (gl, (W - gw) / 2),  # 7
                (gl, (W + gw) / 2),  # 8
                (s, W / 2),  # 9
                (pl, (W - pw) / 2),  # 10
                (pl, (W - gw) / 2),  # 11
                (pl, (W + gw) / 2),  # 12
                (pl, (W + pw) / 2),  # 13
                (L / 2, 0),  # 14
                (L / 2, W / 2 - r),  # 15
                (L / 2, W / 2 + r),  # 16
                (L / 2, W),  # 17
                (L - pl, (W - pw) / 2),  # 18
                (L - pl, (W - gw) / 2),  # 19
                (L - pl, (W + gw) / 2),  # 20
                (L - pl, (W + pw) / 2),  # 21
                (L - s, W / 2),  # 22
                (L - gl, (W - gw) / 2),  # 23
                (L - gl, (W + gw) / 2),  # 24
                (L, 0),  # 25
                (L, (W - pw) / 2),  # 26
                (L, (W - gw) / 2),  # 27
                (L, (W + gw) / 2),  # 28
                (L, (W + pw) / 2),  # 29
                (L, W),  # 30
                (L / 2 - r, W / 2),  # 31
                (L / 2 + r, W / 2),  # 32
            ],
            dtype=np.float64,
        )

    # các đoạn vạch nối giữa 2 keypoint (index từ 0) để vẽ radar
    edges: tuple = (
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (6, 7),
        (9, 10), (10, 11), (11, 12), (13, 14), (14, 15),
        (15, 16), (17, 18), (18, 19), (19, 20), (22, 23),
        (24, 25), (25, 26), (26, 27), (27, 28), (28, 29),
        (0, 13), (1, 9), (2, 6), (3, 7), (4, 12), (5, 16),
        (13, 24), (17, 25), (22, 26), (23, 27), (20, 28), (16, 29),
    )

    @property
    def flip_idx(self) -> list[int]:
        """Lật ảnh ngang tương đương x -> length - x: mỗi keypoint đổi chỗ với điểm đối xứng của nó."""
        v = self.vertices
        mirrored = np.column_stack([self.length - v[:, 0], v[:, 1]])
        dist = np.linalg.norm(mirrored[:, None] - v[None], axis=2)
        idx = dist.argmin(axis=1)
        assert np.allclose(dist.min(axis=1), 0), "config không đối xứng"
        return idx.tolist()


PITCH = PitchConfig()
