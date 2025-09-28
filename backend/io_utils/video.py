import os, cv2
from vision.smoothing import compute_safe_crop, ZoomController

def save_clip(
    input_path, start_f, end_f, out_path, fps, output_size, view_meta,
    max_shift_px=50,      # 每幀最大中心位移（像素）
    follow_alpha=0.3,     # 中心緩動係數
    deadzone_px=8,        # 中心死區
    zoom_deadband_pct=0.20,   # 寬度死區 %
    zoom_confirm_frames=8,    # 連續超出死區幀數後才換檔
    zoom_ease_frames=12,      # 換檔後用幾幀滑到新大小
    zoom_max_rate_pct=0.10,   # 每幀最大縮放變化百分比
    zoom_bounds=(40, 600),    # 寬度下限/上限（像素）
):
    def _as_pos_int(v):
        if v is None:
            return None
        try:
            iv = int(round(float(v)))
            return iv if iv > 0 else None
        except Exception:
            return None

    if end_f < start_f:
        return False

    cap2 = cv2.VideoCapture(input_path)
    if not cap2.isOpened():
        print("無法開啟影片以輸出精華：", input_path)
        return False

    frame_w = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap2.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    # === 初始化追焦中心 ===
    init_center = None
    for f in range(start_f, end_f + 1):
        meta = view_meta.get(f)
        if meta:
            bc = meta.get("ball_center")
            if bc and len(bc) == 2 and bc[0] is not None and bc[1] is not None:
                init_center = (int(bc[0]), int(bc[1]))
                break
    if init_center is None:
        init_center = (frame_w // 2, frame_h // 2)
    last_cx, last_cy = int(init_center[0]), int(init_center[1])

    # === 初始化縮放寬度 ===
    init_w = None
    for f in range(start_f, end_f + 1):
        meta = view_meta.get(f)
        if meta:
            w = _as_pos_int(meta.get("holder_width"))
            if w:
                init_w = w
                break
    if init_w is None:
        init_w = max(1, frame_w // 16)

    # 限制在邊界內
    init_w = max(zoom_bounds[0], min(init_w, zoom_bounds[1]))

    zoom = ZoomController(
        init_width=init_w,
        deadband_pct=zoom_deadband_pct,
        confirm_frames=zoom_confirm_frames,
        ease_frames=zoom_ease_frames,
        max_rate_pct=zoom_max_rate_pct,
        scale_bounds=zoom_bounds
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path, fourcc, fps, output_size)

    current = start_f
    while current <= end_f:
        ret, frm = cap2.read()
        if not ret:
            break

        meta = view_meta.get(current)

        # 追焦（球中心）
        if meta:
            bc = meta.get("ball_center")
            if bc and bc[0] is not None and bc[1] is not None:
                tx, ty = int(bc[0]), int(bc[1])
                dx, dy = tx - last_cx, ty - last_cy
                dist2 = dx*dx + dy*dy
                if dist2 > deadzone_px * deadzone_px:
                    dist = dist2 ** 0.5
                    if dist > max_shift_px:
                        scale = max_shift_px / dist
                        tx = int(last_cx + dx * scale)
                        ty = int(last_cy + dy * scale)
                    # 緩動中心
                    last_cx = int(round((1 - follow_alpha) * last_cx + follow_alpha * tx))
                    last_cy = int(round((1 - follow_alpha) * last_cy + follow_alpha * ty))

        # 目標縮放寬度
        suggested_w = None
        if meta:
            w = _as_pos_int(meta.get("holder_width"))
            if w:
                suggested_w = max(zoom_bounds[0], min(w, zoom_bounds[1]))

        cur_w = zoom.observe(suggested_w if suggested_w is not None else zoom.cur_width)

        x1, y1, x2, y2 = compute_safe_crop(
            center_xy=(last_cx, last_cy),
            base_width=cur_w,
            frame_w=frame_w,
            frame_h=frame_h,
            min_size=(320, 180)
        )

        crop = frm[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frm
        writer.write(cv2.resize(crop, output_size, interpolation=cv2.INTER_LINEAR))
        current += 1

    writer.release()
    cap2.release()
    return True
