#config.py
import os
from datetime import datetime
import cv2
import numpy as np

# 路徑
RAW_VIDEO_PATH  = r"/home/ubuntu/Yolo_V7/game_video.mp4"
PROC_VIDEO_PATH = r"/home/ubuntu/Yolo_V7/firmRoot/tools/WB_CLAHE_JSON_ROI.mp4"

TRACKING_JSON = r"/home/ubuntu/Yolo_V7/output_detection/game_video.json"
OUTPUT_VIDEO = r"/home/ubuntu/Yolo_V7/firmRoot/output/output.mp4"
HIGHLIGHT_DIR = r"/home/ubuntu/Yolo_V7/firmRoot/output/highlights"
LOG_DIR = r"/home/ubuntu/Yolo_V7/firmRoot/output/logs"
FFMPEG_PATH = r"/usr/bin/ffmpeg"
MODEL_PATH = r"/home/ubuntu/Yolo_V7/firmRoot/best.pt"

os.makedirs(HIGHLIGHT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 門檻/參數
PERSON_THRESHOLD = 0.3
BALL_THRESHOLD = 0.45
CONFIRMATION_FRAMES = 5
MIN_CONFIRM_COUNT = 3
MERGE_WINDOW = 15
CONTACT_MIN_SEC = 0.25 #持球交集
GOAL_MIN_SEC    = 0.25 #進球交集

JERSEY_CSV = os.path.join(
    LOG_DIR, datetime.now().strftime("jersey_log_%Y%m%d_%H%M%S.csv")
)
JERSEY_STATS_TXT = os.path.join(LOG_DIR, "jersey_stats.txt")

def _bgr_to_lab1(bgr):
    """單點 BGR -> LAB (float32)"""
    arr = np.uint8([[bgr]])
    lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab[0,0,:]  # (L, A, B)

def color_distance_lab(bgr1, bgr2):
    def _is_black(bgr): return bgr[0] < 50 and bgr[1] < 50 and bgr[2] < 50
    def _is_white(bgr): return bgr[0] > 220 and bgr[1] > 220 and bgr[2] > 220

    if (_is_black(bgr1) and _is_black(bgr2)) or (_is_white(bgr1) and _is_white(bgr2)):
        return 0.0

    lab1 = _bgr_to_lab1(bgr1)
    lab2 = _bgr_to_lab1(bgr2)
    de = float(np.linalg.norm(lab1 - lab2))
    return de