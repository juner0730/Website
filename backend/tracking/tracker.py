# tracking/tracker.py
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Callable
import numpy as np
import lap
from vision.color import classify_team_color_by_hist
from config import CONFIRMATION_FRAMES, MIN_CONFIRM_COUNT

ColorHintFn = Callable[["EuclideanTracker", List[Tuple[Tuple[int,int],Tuple[int,int],float]]], List[Optional[Tuple[int,int,int]]]]

@dataclass
class TrackedObject:  # (兩回合匹配)
    id: int
    cls: str  # "person" / "sports ball" / "gate"
    bbox: Tuple[int, int, int, int]
    last_seen: int
    confirmed: bool = False
    appear_history: List[int] = field(default_factory=list)
    status: str = "Steady"
    dominant_color: Tuple[int, int, int] = (0, 0, 0)
    team_color: Tuple[int, int, int] = (0, 0, 0)
    score: float = 0.0
    prev_center: Optional[Tuple[int, int]] = None
    center: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.width = x2 - x1
        self.height = y2 - y1

    def update_bbox(self, new_bbox: Tuple[int, int, int, int]):
        x1, y1, x2, y2 = new_bbox
        new_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        self.prev_center = self.center or new_center
        self.center = new_center
        self.bbox = new_bbox
        self.width = x2 - x1
        self.height = y2 - y1

class EuclideanTracker:
    def __init__(self,
        strict_cost_limit: float = 80.0,    # 第1回合距離門檻（像素）
        relaxed_cost_limit: float = 160.0,  # 第2回合距離門檻（較大像素）
        max_lost_age_active: int = 10,      # Active物件失聯多少幀 → 轉入Lost池
        max_lost_age_buffer: int = 45,      # Lost池最多保留幀數
        team_penalty_strict: float = 1e6,   # 顏色不同時的懲罰（嚴格）
        team_penalty_relaxed: float = 200.0,# 顏色不同時的懲罰（寬鬆）
        motion_alpha: float = 0.60,         # 速度外推權重
        use_color_cost: bool = True,        # NEW: 是否啟用顏色懲罰（Pass1=False, Pass2=True）
        det_color_hint_fn: Optional[ColorHintFn] = None # NEW: 偵測框的顏色提示函式（由 Pass1 產生）
    ):
        self.next_id = 0
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.lost_pool: Dict[int, TrackedObject] = {}  # 失聯緩衝池

        self.class_params = {
            "person": dict(
                strict_cost_limit=strict_cost_limit,
                relaxed_cost_limit=relaxed_cost_limit,
                max_lost_age_active=max_lost_age_active,
                max_lost_age_buffer=max_lost_age_buffer,
                team_penalty_strict=team_penalty_strict,
                team_penalty_relaxed=team_penalty_relaxed,
            ),
            "sports ball": dict(
                strict_cost_limit=relaxed_cost_limit * 1.1,
                relaxed_cost_limit=relaxed_cost_limit * 1.6,
                max_lost_age_active=max_lost_age_active + 5,
                max_lost_age_buffer=max_lost_age_buffer + 30,
                team_penalty_strict=0.0,
                team_penalty_relaxed=0.0,
            ),
            "gate": dict(
                strict_cost_limit=strict_cost_limit * 0.8,
                relaxed_cost_limit=relaxed_cost_limit * 0.8,
                max_lost_age_active=max_lost_age_active + 20,
                max_lost_age_buffer=max_lost_age_buffer + 60,
                team_penalty_strict=0.0,
                team_penalty_relaxed=0.0,
            ),
        }

        self.NEED_CONFIRM = {"person": True, "sports ball": True, "gate": True}
        self.CONF_FRAMES = {"person": CONFIRMATION_FRAMES, "sports ball": CONFIRMATION_FRAMES, "gate": CONFIRMATION_FRAMES}
        self.CONF_MIN_COUNT = {"person": MIN_CONFIRM_COUNT, "sports ball": MIN_CONFIRM_COUNT + 2, "gate": MIN_CONFIRM_COUNT}

        self.motion_alpha = motion_alpha
        self.team_penalty_strict = team_penalty_strict
        self.team_penalty_relaxed = team_penalty_relaxed
        self.ball_id: Optional[int] = None

        # NEW
        self.use_color_cost = use_color_cost
        self.det_color_hint_fn = det_color_hint_fn
        self.current_frame_idx: Optional[int] = None  # 由外部在呼叫 update() 前設定

    # ---------- helper ----------
    def _center(self, bbox: Tuple[int, int, int, int]):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def _predict_center(self, obj: TrackedObject):
        if obj.center is None:
            return self._center(obj.bbox)
        if obj.prev_center is None:
            return obj.center
        dx = obj.center[0] - obj.prev_center[0]
        dy = obj.center[1] - obj.prev_center[1]
        return (obj.center[0] + self.motion_alpha * dx, obj.center[1] + self.motion_alpha * dy)

    def _maybe_confirm(self, obj: TrackedObject, frame_idx: int):
        if not self.NEED_CONFIRM.get(obj.cls, False):
            obj.confirmed = True
            return
        if obj.confirmed:
            return
        win = self.CONF_FRAMES.get(obj.cls, CONFIRMATION_FRAMES)
        req = self.CONF_MIN_COUNT.get(obj.cls, MIN_CONFIRM_COUNT)
        hist = obj.appear_history[-win:]
        cnt = len(hist)
        if len(obj.appear_history) >= win and cnt >= req:
            obj.confirmed = True

    def _build_cost(
        self,
        cls_name: str,
        existing_objs: List[TrackedObject],
        det_boxes: List[Tuple[Tuple[int,int],Tuple[int,int],float]],
        frame,
        is_relaxed: bool
    ):
        num_objs = len(existing_objs)
        num_dets = len(det_boxes)
        if num_objs == 0 or num_dets == 0:
            return np.empty((0, 0), dtype=np.float32), []

        dist = np.zeros((num_objs, num_dets), dtype=np.float32)
        color_pen = np.zeros((num_objs, num_dets), dtype=np.float32)
        det_team_colors: List[Optional[Tuple[int,int,int]]] = [None] * num_dets

        need_team = (cls_name == "person") and self.use_color_cost

        # 1) 先用 Pass1 的提示（若有）
        if need_team and self.det_color_hint_fn is not None:
            try:
                hinted = self.det_color_hint_fn(self, det_boxes)  # 期望回傳長度==num_dets 的 BGR/None
                if hinted and len(hinted) == num_dets:
                    det_team_colors = hinted[:]  # 覆蓋
            except Exception:
                pass

        # 2) 尚未被提示補滿的（且需要顏色成本時），才做即時計算
        if need_team and any(c is None for c in det_team_colors):
            for j, det in enumerate(det_boxes):
                if det_team_colors[j] is None:
                    (x1, y1), (x2, y2), _ = det
                    det_team_colors[j] = classify_team_color_by_hist(frame, (x1, y1, x2, y2))

        # 3) 距離成本 + 顏色懲罰（嚴格相等）
        for i, obj in enumerate(existing_objs):
            pcx, pcy = self._predict_center(obj)
            for j, det in enumerate(det_boxes):
                (x1, y1), (x2, y2), _ = det
                dcx, dcy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                dist[i, j] = np.hypot(pcx - dcx, pcy - dcy)

                if (cls_name == "person") and self.use_color_cost:
                    same = (obj.team_color == det_team_colors[j])
                    color_pen[i, j] = 0.0 if same else (self.team_penalty_relaxed if is_relaxed else self.team_penalty_strict)

        cost = dist + color_pen
        return cost, det_team_colors

    def _assign(self, cost: np.ndarray, cost_limit: float):
        if cost.size == 0:
            return {}, set()
        _, x, _ = lap.lapjv(cost, extend_cost=True, cost_limit=cost_limit)
        assigned = {}
        unmatched_dets = set(range(cost.shape[1]))
        for i, j in enumerate(x):
            if j >= 0 and j < cost.shape[1] and cost[i, j] <= cost_limit:
                assigned[i] = j
                unmatched_dets.discard(j)
        return assigned, unmatched_dets

    # ---------- 主邏輯 ----------
    def update(
        self,
        detections: Dict[str, List[Tuple[Tuple[int, int], Tuple[int, int], float]]],
        frame_idx: int,
        frame=None
    ):
        self.current_frame_idx = frame_idx  # 讓 det_color_hint_fn 可取用
        matched_ids = set()
        for cls_name, det_boxes in detections.items():
            if not det_boxes:
                continue

            p = self.class_params.get(cls_name, self.class_params["person"])
            strict_limit = p["strict_cost_limit"]
            relaxed_limit = p["relaxed_cost_limit"]
            max_lost_active = p["max_lost_age_active"]
            max_lost_buffer = p["max_lost_age_buffer"]

            active_objs: List[TrackedObject] = []
            lost_objs: List[TrackedObject] = []
            for obj in self.tracked_objects.values():
                if obj.cls != cls_name:
                    continue
                lost_age = frame_idx - obj.last_seen
                if lost_age <= max_lost_active:
                    active_objs.append(obj)
                else:
                    self.lost_pool[obj.id] = obj

            for obj in list(self.lost_pool.values()):
                if obj.cls != cls_name:
                    continue
                lost_age = frame_idx - obj.last_seen
                if lost_age <= max_lost_buffer:
                    lost_objs.append(obj)
                else:
                    self.lost_pool.pop(obj.id, None)

            # 球單例優先
            if cls_name == "sports ball" and self.ball_id is not None:
                active_objs.sort(key=lambda o: (o.id != self.ball_id,))
                lost_objs.sort(key=lambda o: (o.id != self.ball_id,))

            # pass-1：confirmed Active（嚴格）
            stable_active = [o for o in active_objs if o.confirmed]
            strict_assigned = {}
            det_left = list(range(len(det_boxes)))

            if stable_active:
                cost, det_team_colors = self._build_cost(cls_name, stable_active, det_boxes, frame, is_relaxed=False)
                assigned_map, _ = self._assign(cost, strict_limit)
                for i, j in assigned_map.items():
                    obj = stable_active[i]
                    (x1, y1), (x2, y2), score = det_boxes[j]
                    obj.update_bbox((x1, y1, x2, y2))
                    obj.last_seen = frame_idx
                    obj.appear_history.append(frame_idx)
                    obj.score = score
                    if cls_name == "person" and det_team_colors[j] is not None:
                        obj.team_color = det_team_colors[j]
                        obj.dominant_color = det_team_colors[j]
                    self._maybe_confirm(obj, frame_idx)
                    matched_ids.add(obj.id)
                    strict_assigned[j] = obj.id
                det_left = [k for k in det_left if k not in strict_assigned]

            # pass-2：lost（寬鬆）
            relaxed_assigned = {}
            if det_left and lost_objs:
                det_subset = [det_boxes[k] for k in det_left]
                cost, det_team_colors = self._build_cost(cls_name, lost_objs, det_subset, frame, is_relaxed=True)
                assigned_map, _ = self._assign(cost, relaxed_limit)
                for i, j_sub in assigned_map.items():
                    obj = lost_objs[i]
                    j_global = det_left[j_sub]
                    (x1, y1), (x2, y2), score = det_boxes[j_global]
                    obj.update_bbox((x1, y1, x2, y2))
                    obj.last_seen = frame_idx
                    obj.appear_history.append(frame_idx)
                    obj.score = score
                    if cls_name == "person" and det_team_colors[j_sub] is not None:
                        obj.team_color = det_team_colors[j_sub]
                        obj.dominant_color = det_team_colors[j_sub]
                    self._maybe_confirm(obj, frame_idx)
                    matched_ids.add(obj.id)
                    relaxed_assigned[j_global] = obj.id
                    self.lost_pool.pop(obj.id, None)
                det_left = [k for k in det_left if k not in relaxed_assigned]

            # pass-3：未確認的新生 Active（嚴格）
            newborn_active = [o for o in active_objs if not o.confirmed]
            newborn_assigned = {}
            if det_left and newborn_active:
                det_subset = [det_boxes[k] for k in det_left]
                cost, det_team_colors = self._build_cost(cls_name, newborn_active, det_subset, frame, is_relaxed=False)
                assigned_map, _ = self._assign(cost, strict_limit)
                for i, j_sub in assigned_map.items():
                    obj = newborn_active[i]
                    j_global = det_left[j_sub]
                    (x1, y1), (x2, y2), score = det_boxes[j_global]
                    obj.update_bbox((x1, y1, x2, y2))
                    obj.last_seen = frame_idx
                    obj.appear_history.append(frame_idx)
                    obj.score = score
                    if cls_name == "person" and det_team_colors[j_sub] is not None:
                        obj.team_color = det_team_colors[j_sub]
                        obj.dominant_color = det_team_colors[j_sub]
                    self._maybe_confirm(obj, frame_idx)
                    matched_ids.add(obj.id)
                    newborn_assigned[j_global] = obj.id
                det_left = [k for k in det_left if k not in newborn_assigned]

            # 建立新物件（球單例守門）
            if det_left:
                for idx in det_left:
                    (x1, y1), (x2, y2), score = det_boxes[idx]

                    if cls_name == "sports ball" and self.ball_id is not None:
                        holder = self.tracked_objects.get(self.ball_id) or self.lost_pool.get(self.ball_id)
                        if holder is not None:
                            last_age = frame_idx - holder.last_seen
                            if last_age <= max_lost_buffer:
                                continue  # 保持單一球 ID，不新增

                    if cls_name == "person":
                        # 若啟用顏色成本，優先用提示；否則即時計算
                        if self.use_color_cost and self.det_color_hint_fn is not None:
                            hinted = self.det_color_hint_fn(self, [det_boxes[idx]])
                            team_col = hinted[0] if hinted and hinted[0] is not None else classify_team_color_by_hist(frame, (x1, y1, x2, y2))
                        else:
                            team_col = (0, 0, 0) if self.use_color_cost else classify_team_color_by_hist(frame, (x1, y1, x2, y2))
                    else:
                        team_col = (0, 0, 0)

                    new_obj = TrackedObject(
                        id=self.next_id,
                        cls=cls_name,
                        bbox=(x1, y1, x2, y2),
                        last_seen=frame_idx,
                        appear_history=[frame_idx],
                        dominant_color=team_col,
                        team_color=team_col,
                        score=score,
                    )
                    if not self.NEED_CONFIRM.get(cls_name, False):
                        new_obj.confirmed = True

                    self.tracked_objects[self.next_id] = new_obj
                    self._maybe_confirm(new_obj, frame_idx)
                    matched_ids.add(self.next_id)

                    if cls_name == "sports ball" and self.ball_id is None:
                        self.ball_id = self.next_id
                    self.next_id += 1

        # 狀態更新與回收
        for obj_id, obj in list(self.tracked_objects.items()):
            p = self.class_params.get(obj.cls, self.class_params["person"])
            lost_age = frame_idx - obj.last_seen
            if lost_age > p["max_lost_age_active"]:
                self.lost_pool[obj_id] = obj
                obj.status = "Lost"
            else:
                obj.status = "Steady"

        for obj_id, obj in list(self.lost_pool.items()):
            p = self.class_params.get(obj.cls, self.class_params["person"])
            if frame_idx - obj.last_seen > p["max_lost_age_buffer"]:
                if obj.cls == "sports ball" and self.ball_id == obj_id:
                    self.ball_id = None
                self.lost_pool.pop(obj_id, None)

        return [o for oid, o in self.tracked_objects.items() if o.confirmed and oid in matched_ids]
