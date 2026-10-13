# Mapping từ class gốc của dataset sang 4 class chuẩn
CLASS_MAP = {
    "player": 0,
    "goalkeeper": 1,
    "referee": 2,
    "ball": 3,
    # thêm các alias tùy dataset, ví dụ:
    "main_referee": 2,
    "side_referee": 2,
    "assistant_referee": 2,
    # alias COCO, để chạy thử pipeline bằng yolo11n.pt trước khi có model 4 class
    "person": 0,
    "sports ball": 3,
}

CLASS_NAMES = ["player", "goalkeeper", "referee", "ball"]
PLAYER_ID = CLASS_NAMES.index("player")
GOALKEEPER_ID = CLASS_NAMES.index("goalkeeper")
REFEREE_ID = CLASS_NAMES.index("referee")
BALL_ID = CLASS_NAMES.index("ball")