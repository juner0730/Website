import os, json, cv2, numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# ========= 路徑設定 =========
VIDEO_PATH = r"/home/ubuntu/Yolo_V7/game_video.mp4"
TRACKING_JSON = r"/home/ubuntu/Yolo_V7/output_detection/game_video.json"
OUTPUT_DIR = r"/tools"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========= 實用函式 =========
def clamp_int(v: float, lo: int, hi: int) -> int:
    return int(v if v >= lo else lo) if v <= hi else int(hi)

def torso_roi_from_bbox(
    bbox_xyxy: Tuple[float, float, float, float], frame_w: int, frame_h: int
) -> Optional[Tuple[int, int, int, int]]:
    x1f, y1f, x2f, y2f = bbox_xyxy
    # 夾在畫面內
    x1 = clamp_int(x1f, 0, frame_w - 1)
    y1 = clamp_int(y1f, 0, frame_h - 1)
    x2 = clamp_int(x2f, 0, frame_w - 1)
    y2 = clamp_int(y2f, 0, frame_h - 1)
    # 修正可能的反序
    if x2 <= x1 or y2 <= y1:
        return None
    w = x2 - x1
    h = y2 - y1
    if w <= 1 or h <= 1:
        return None
    return (x1, y1, w, h)

class RoiClaheApplier:
    def __init__(self, clip: float = 2.0, tiles: int = 2):
        self.clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
        self._ycrcb = None  
        self._roi_shape = None

    def apply_inplace(self, img: np.ndarray, roi_xywh: Tuple[int, int, int, int]) -> None:
        x, y, w, h = roi_xywh
        # 邊界快速檢查
        if w <= 0 or h <= 0:
            return
        H, W = img.shape[:2]
        if x >= W or y >= H:
            return
        # 限制 ROI 在畫面內
        w = min(w, W - x)
        h = min(h, H - y)
        if w <= 1 or h <= 1:
            return
        roi = img[y:y + h, x:x + w]
        if self._roi_shape != roi.shape:
            self._ycrcb = np.empty_like(roi)  # (h,w,3) uint8
            self._roi_shape = roi.shape

        cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb, dst=self._ycrcb)

        y_ch, cr, cb = cv2.split(self._ycrcb)
        y_eq = self.clahe.apply(y_ch)

        cv2.merge((y_eq, cr, cb), dst=self._ycrcb)
        cv2.cvtColor(self._ycrcb, cv2.COLOR_YCrCb2BGR, dst=roi)

def load_tracking(tracking_json_path: str) -> Dict[str, dict]:
    if not os.path.exists(tracking_json_path):
        print(f"[警告] 找不到 TRACKING_JSON：{tracking_json_path}")
        return {}
    try:
        with open(tracking_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        print("[警告] TRACKING_JSON 不是 dict 格式，已忽略。")
        return {}
    except Exception as e:
        print(f"[警告] 讀取 TRACKING_JSON 失敗：{e}")
        return {}

def export_video_wb_roi_clahe_from_json(
    video_path: str,
    tracking_json_path: str,
    effect_name: str = "WB_CLAHE_JSON_ROI",
) -> None:
    tracking_data = load_tracking(tracking_json_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[錯誤] 無法開啟影片！"); return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = os.path.join(OUTPUT_DIR, f"{effect_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    if not writer.isOpened():
        print("[錯誤] 無法建立輸出檔案！"); cap.release(); return

    roi_clahe = RoiClaheApplier()
    pbar = tqdm(total=total if total > 0 else None, desc=f"{effect_name}", unit="f")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rec = tracking_data.get(str(frame_idx), {})
            persons_raw = rec.get("person", []) or []

            # 篩選所有符合閾值的人
            persons = []
            for item in persons_raw:
                if not (isinstance(item, list) and len(item) >= 3):
                    continue
                (x1y1, x2y2, sc) = item[:3]
                if not (isinstance(x1y1, (list, tuple)) and isinstance(x2y2, (list, tuple)) and len(x1y1) >= 2 and len(x2y2) >= 2):
                    continue
                x1, y1 = float(x1y1[0]), float(x1y1[1])
                x2, y2 = float(x2y2[0]), float(x2y2[1])
                persons.append((x1, y1, x2, y2, float(sc)))


            # 處理所有人
            for (x1, y1, x2, y2, _sc) in persons:
                roi = torso_roi_from_bbox((x1, y1, x2, y2), W, H)
                if roi is not None:
                    roi_clahe.apply_inplace(frame, roi)

            writer.write(frame)
            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        writer.release()
        cap.release()

    print(f"{effect_name} 輸出完成：{out_path}")


# ========= 主程式 =========
if __name__ == "__main__":
    export_video_wb_roi_clahe_from_json(
        video_path=VIDEO_PATH,
        tracking_json_path=TRACKING_JSON,
        effect_name="WB_CLAHE_JSON_ROI",
    )
    print("所有影片輸出完成！")
