#color.py
import cv2
import numpy as np

# ===== 可調參數 =====
_H_BIN = 10        # 色相量化的步長(hue180)
S_MIN = 30         # 有效彩度下限
V_MIN = 40         # 有效亮度下限
BLACK_V_MAX = 30   # HSV 黑色上限
WHITE_S_MAX = 50   # HSV 白色彩度上限
WHITE_V_MIN = 200  # HSV 白色亮度下限
ROI_RESIZE = 60    # ROI 取樣大小
GRAY_DELTA = 15    # BGR 三通道彼此差異上限，用於偵測黑/白/灰

SMOOTH_KSIZE = 5        # 高斯模糊核(奇數；設0不啟用)
MIN_COLOR_PROP = 0.12   # 最小色彩佔比

def _crop_torso_roi(image, bbox):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0: 
        return None
    h_step = h / 10.0
    roi_y1 = int(y1 + 2 * h_step)
    roi_y2 = int(y1 + 5 * h_step)
    w_step = w / 5.0
    roi_x1 = int(x1 + 1 * w_step)
    roi_x2 = int(x1 + 4 * w_step)

    roi_x1 = max(x1, min(roi_x1, x2 - 1))
    roi_x2 = max(roi_x1 + 1, min(roi_x2, x2))
    roi_y1 = max(y1, min(roi_y1, y2 - 1))
    roi_y2 = max(roi_y1 + 1, min(roi_y2, y2))
    crop = image[roi_y1:roi_y2, roi_x1:roi_x2]
    return crop if crop.size else None

def _palette_from_h(h_bin_center):
    h = h_bin_center  # hue in [0,180)
    if (h < 15) or (h >= 160):   # red
        return (0, 0, 255)
    if 15 <= h < 35:             # yellow
        return (0, 255, 255)
    if 35 <= h < 85:             # green
        return (0, 255, 0)
    if 85 <= h < 135:            # blue/cyan
        return (255, 0, 0)
    return (255, 165, 0)         # orange/magenta fallback

def classify_team_color_by_hist(image, bbox):
    crop = _crop_torso_roi(image, bbox)
    if crop is None:
        return (0, 0, 0)

    crop = cv2.resize(crop, (ROI_RESIZE, ROI_RESIZE), interpolation=cv2.INTER_AREA)
    
    # 降噪
    if isinstance(SMOOTH_KSIZE, int) and SMOOTH_KSIZE >= 3 and SMOOTH_KSIZE % 2 == 1:
        crop = cv2.GaussianBlur(crop, (SMOOTH_KSIZE, SMOOTH_KSIZE), 0, borderType=cv2.BORDER_REPLICATE)

    # --- HSV 判定 ---
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.int32)
    S = hsv[:, :, 1].astype(np.int32)
    V = hsv[:, :, 2].astype(np.int32)

    total = H.size
    if total == 0:
        return (0, 0, 0)

    bgr = crop.astype(np.int32)
    B, G, R = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
    gray_mask = (np.abs(R - G) <= GRAY_DELTA) & (np.abs(G - B) <= GRAY_DELTA) & (np.abs(B - R) <= GRAY_DELTA)
    black_bgr_mask = gray_mask & (R < 60) & (G < 60) & (B < 60)
    white_bgr_mask = gray_mask & (R > 90) & (G > 90) & (B > 90)

    black_mask = (V <= BLACK_V_MAX) | black_bgr_mask
    white_mask = ((S <= WHITE_S_MAX) & (V >= WHITE_V_MIN)) | white_bgr_mask

    valid_color = (S >= S_MIN) & (V >= V_MIN) & (~black_mask) & (~white_mask)

    black_prop = float(black_mask.sum()) / total
    white_prop = float(white_mask.sum()) / total

    color_prop = 0.0
    palette_bgr = None
    if valid_color.any():
        h_valid = H[valid_color]
        bins = h_valid // _H_BIN
        counts = np.bincount(bins, minlength=180 // _H_BIN)
        best_bin = int(np.argmax(counts))
        best_count = int(counts[best_bin])
        color_prop = float(best_count) / total

        # 佔比不足 → 視為噪點
        if color_prop >= MIN_COLOR_PROP:
            h_center = int(best_bin * _H_BIN + _H_BIN // 2)
            palette_bgr = _palette_from_h(h_center)

    # --- 三者取最大佔比 ---
    if black_prop >= white_prop and black_prop >= color_prop:
        return (0, 0, 0)
    if white_prop >= black_prop and white_prop >= color_prop:
        return (255, 255, 255)
    return palette_bgr if palette_bgr is not None else (0, 0, 0)
