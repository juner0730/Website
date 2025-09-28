def compute_safe_crop(center_xy, base_width, frame_w, frame_h, min_size=(320,180)):
    cx, cy = center_xy
    ideal_w = max(min_size[0], int(base_width * 16))
    ideal_h = max(min_size[1], int(base_width * 9))
    ideal_w = min(ideal_w, frame_w); ideal_h = min(ideal_h, frame_h)
    half_w, half_h = ideal_w//2, ideal_h//2
    x1, y1 = cx-half_w, cy-half_h; x2, y2 = x1+ideal_w, y1+ideal_h
    # 邊界修正
    if x1 < 0: x2 -= x1; x1 = 0
    if y1 < 0: y2 -= y1; y1 = 0
    if x2 > frame_w: x1 -= (x2-frame_w); x2 = frame_w
    if y2 > frame_h: y1 -= (y2-frame_h); y2 = frame_h
    cur_w, cur_h = x2-x1, y2-y1
    if cur_w < min_size[0] or cur_h < min_size[1]:
        cur_w = max(cur_w, min_size[0]); cur_h = max(cur_h, min_size[1])
        half_w, half_h = cur_w//2, cur_h//2
        x1, y1 = cx-half_w, cy-half_h; x2, y2 = x1+cur_w, y1+cur_h
        if x1 < 0: x2 -= x1; x1 = 0
        if y1 < 0: y2 -= y1; y1 = 0
        if x2 > frame_w: x1 -= (x2-frame_w); x2 = frame_w
        if y2 > frame_h: y1 -= (y2-frame_h); y2 = frame_h
    return int(x1), int(y1), int(x2), int(y2)

class ZoomController:
    def __init__(self, init_width, deadband_pct=0.12, confirm_frames=8,
                 ease_frames=12, max_rate_pct=0.10, scale_bounds=(40, 600)):
        self.cur_width = float(init_width)
        self.target_width = float(init_width)
        self.deadband_pct = deadband_pct
        self.confirm_frames = confirm_frames
        self.ease_frames = max(1, ease_frames)
        self.max_rate_pct = max_rate_pct
        self.min_w, self.max_w = scale_bounds

        self._drift_cnt = 0
        self._ease_step = 0

    def _clamp(self, w):
        return max(self.min_w, min(self.max_w, w))

    def observe(self, suggested_width):
        if suggested_width <= 0:
            # 沒有可靠量測 → 只根據 easing 往 target 走
            return self._ease_once()

        suggested_width = self._clamp(float(suggested_width))
        # 相對偏差
        rel = abs(suggested_width - self.cur_width) / max(1.0, self.cur_width)

        if rel <= self.deadband_pct:
            # 在死區內 → 重置漂移計數，不換目標，但繼續處理既有 easing
            self._drift_cnt = 0
            return self._ease_once()

        # 超出死區 → 連續計數
        self._drift_cnt += 1
        if self._drift_cnt >= self.confirm_frames:
            # 觸發換檔：設定新目標，重置 easing
            self.target_width = suggested_width
            self._ease_step = 0
            self._drift_cnt = 0

        return self._ease_once()

    def _ease_once(self):
        # 若已在目標附近，就當到位
        if abs(self.target_width - self.cur_width) < 1e-6:
            return self.cur_width

        # 線性或指數緩動都可；這裡用簡單線性：分 ease_frames 走完
        remain = self.ease_frames - self._ease_step
        remain = max(1, remain)
        step = (self.target_width - self.cur_width) / remain

        # 每幀最大速率限制（百分比）
        max_step = self.cur_width * self.max_rate_pct
        if step > 0:
            step = min(step, max_step)
        else:
            step = max(step, -max_step)

        self.cur_width = self._clamp(self.cur_width + step)
        self._ease_step += 1
        return self.cur_width
