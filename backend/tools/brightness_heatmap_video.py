# -*- coding: utf-8 -*-
"""
將影片輸出為亮度熱力圖影片：
- mode=grid：每幀分網格平均亮度 → 放大到全幅 → 上色
- mode=pixel：直接用像素亮度（可選模糊）→ 上色
- 可與原畫面疊合 (--blend)，可時間緩動 (--ema)

用法：
  python tools/brightness_heatmap_video.py --use-config
  python tools/brightness_heatmap_video.py --video "C:\path\to\input.mp4" --mode grid --grid 48x27 --blend 0.35 --ema 0.25
"""

import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm

CMAPS = {
    "inferno": cv2.COLORMAP_INFERNO,
    "turbo":   cv2.COLORMAP_TURBO,
    "jet":     cv2.COLORMAP_JET,
}

def load_video_path_from_config():
    try:
        from config import VIDEO_PATH
        return VIDEO_PATH
    except Exception:
        return None

def bgr_to_y(img_bgr):
    # 取 YCrCb 的 Y 當亮度，範圍 0..255
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]

def grid_average(y, gw, gh):
    """把亮度圖 y 降採樣到 (gh, gw) 的平均值網格，再放大回原尺寸。"""
    h, w = y.shape
    # 用縮小/放大實現格平均，速度比逐格迭代快很多
    small = cv2.resize(y, (gw, gh), interpolation=cv2.INTER_AREA)
    up    = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return up

def apply_colormap(gray_u8, cmap_name):
    cmap = CMAPS.get(cmap_name, cv2.COLORMAP_INFERNO)
    return cv2.applyColorMap(gray_u8, cmap)

def main():
    ap = argparse.ArgumentParser(description="影片亮度熱力圖輸出")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--video", type=str, help="輸入影片路徑")
    src.add_argument("--use-config", action="store_true", help="使用 config.VIDEO_PATH")
    ap.add_argument("--out", type=str, default=None, help="輸出 mp4 路徑（預設為 <input>_heatmap.mp4）")
    ap.add_argument("--mode", type=str, default="grid", choices=["grid", "pixel"], help="視覺化模式")
    ap.add_argument("--grid", type=str, default="48x27", help="grid 模式的網格數，如 48x27")
    ap.add_argument("--blur", type=int, default=0, help="pixel 模式的高斯模糊核大小（奇數，0=不模糊）")
    ap.add_argument("--ema", type=float, default=0.0, help="時間緩動係數 (0~1)，0=關閉；建議 0.15~0.35")
    ap.add_argument("--blend", type=float, default=0.0, help="與原畫面的疊合比例 (0~1)，0=只看熱圖")
    ap.add_argument("--cmap", type=str, default="inferno", choices=list(CMAPS.keys()), help="色圖")
    args = ap.parse_args()

    # 來源路徑
    if args.use_config:
        video_path = load_video_path_from_config()
        if not video_path:
            raise RuntimeError("找不到 config.VIDEO_PATH，請改用 --video 指定路徑")
    else:
        if not args.video:
            raise RuntimeError("請用 --video 指定影片，或改用 --use-config")
        video_path = args.video

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"找不到影片：{video_path}")

    # 解析 grid
    if "x" in args.grid.lower():
        gw_str, gh_str = args.grid.lower().split("x")
        gw, gh = int(gw_str), int(gh_str)
    else:
        gw, gh = 48, 27

    # 輸出檔名
    if args.out:
        out_path = args.out
    else:
        root, ext = os.path.splitext(video_path)
        out_path = root + "_heatmap.mp4"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片：{video_path}")

    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps= cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    # 時間緩動狀態
    ema_on = 0.0 < args.ema < 1.0
    ema_map = None

    pbar = tqdm(total=total if total > 0 else None, desc="Rendering heatmap", unit="frame")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            y = bgr_to_y(frame)

            if args.mode == "grid":
                # 格平均 → 方塊熱圖
                gray = grid_average(y, gw, gh)
            else:
                # 像素模式（可選模糊）
                gray = y.copy()
                if args.blur and args.blur % 2 == 1 and args.blur > 0:
                    gray = cv2.GaussianBlur(gray, (args.blur, args.blur), 0)

            # 時間緩動（在灰階域進行）
            if ema_on:
                if ema_map is None:
                    ema_map = gray.astype(np.float32)
                else:
                    ema_map = (1 - args.ema) * ema_map + args.ema * gray.astype(np.float32)
                gray_vis = np.clip(ema_map, 0, 255).astype(np.uint8)
            else:
                gray_vis = gray.astype(np.uint8)

            # 上色
            heat = apply_colormap(gray_vis, args.cmap)

            # 疊合
            blend = float(np.clip(args.blend, 0.0, 1.0))
            if blend > 0:
                out_frame = cv2.addWeighted(heat, blend, frame, 1.0 - blend, 0)
            else:
                out_frame = heat

            writer.write(out_frame)
            pbar.update(1)
    finally:
        pbar.close()
        writer.release()
        cap.release()

    print(f"[OK] 已輸出：{os.path.abspath(out_path)}")

if __name__ == "__main__":
    main()
