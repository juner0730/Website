import cv2
import json
import os
import lap
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, List
from typing import Optional
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--video', required=True, help='Path to input video')
parser.add_argument('--json', required=True, help='Path to tracking JSON')
parser.add_argument('--output', default='output.mp4', help='Output video path')
args = parser.parse_args()

VIDEO_PATH = args.video
TRACKING_JSON = args.json
OUTPUT_VIDEO = args.output

# === 參數設定 ===
PERSON_THRESHOLD = 0.3
BALL_THRESHOLD = 0.45
CONFIRMATION_FRAMES = 5
MIN_CONFIRM_COUNT = 3

@dataclass
class TrackedObject:
    id: int
    cls: str
    bbox: Tuple[int, int, int, int]
    last_seen: int
    confirmed: bool = False
    appear_history: List[int] = field(default_factory=list)
    status: str = "Steady"
    dominant_color: Tuple[int, int, int] = (0, 0, 0)
    team_color: Tuple[int, int, int] = (0, 0, 0)
    score: float = 0.0  
    prev_center: Tuple[int, int] = None
    center: Optional[Tuple[int, int]] = None
    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.width = x2 - x1
        self.height = y2 - y1

    def update_bbox(self, new_bbox):
        x1, y1, x2, y2 = new_bbox
        new_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        # 儲存前一中心點
        if self.center is not None:
            self.prev_center = self.center
        else:
            self.prev_center = new_center
        self.center = new_center
        self.bbox = new_bbox
        self.width = x2 - x1
        self.height = y2 - y1


# --- 主色提取 ---
def get_dominant_color(image, bbox):
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return (0, 0, 0)
    crop = cv2.resize(crop, (30, 30))
    #crop = cv2.convertScaleAbs(crop, alpha=1, beta=20)
    data = crop.reshape((-1, 3)).astype(np.float32)
    _, _, centers = cv2.kmeans(data, 1, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        1, cv2.KMEANS_RANDOM_CENTERS)
    return tuple(map(int, centers[0]))

def classify_color_bgr(bgr):
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv
    b , g , r = bgr
    if b < 50 and g < 50 and r < 50:
        return (0, 0, 0)        #black
    if s < 40:
        return (255, 255, 255)  # white
    if h < 15 or h > 160:
        return (0, 0, 255)      # red
    elif 15 <= h < 35:
        return (0, 255, 255)    # yellow
    elif 35 <= h < 85:
        return (0, 255, 0)      # green
    elif 85 <= h < 135:
        return (255, 0, 0)      # blue
    else:
        return (255, 165, 0)    # orange
    
def bboxes_intersect(bbox1, bbox2):
    # bbox = (x1, y1, x2, y2)
    x1, y1, x2, y2 = bbox1
    xx1, yy1, xx2, yy2 = bbox2
    return not (x2 < xx1 or xx2 < x1 or y2 < (yy1 + yy2)//2 or yy2 < y1)

def bbox_fully_inside(inner_bbox, outer_bbox):
    # inner_bbox =（球）
    # outer_bbox =（龍門）
    x1, y1, x2, y2 = inner_bbox
    gx1, gy1, gx2, gy2 = outer_bbox
    corners = [
        (x1, y1),
        (x1, y2),
        (x2, y1),
        (x2, y2),
    ]
    for x, y in corners:
        if not (gx1 <= x <= gx2 and gy1 <= y <= gy2):
            return False
    return True


# --- 追蹤器 ---
class EuclideanTracker:
    def __init__(self):
        self.next_id = 0
        self.tracked_objects: Dict[int, TrackedObject] = {}


    def _get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    def update(self, detections: Dict[str, List[Tuple[Tuple[int, int], Tuple[int, int], float]]], frame_idx: int, frame=None):
        matched_ids = set()
        for cls_name, det_boxes in detections.items():
            current_objs = [obj for obj in self.tracked_objects.values() if obj.cls == cls_name]
            if not det_boxes:
                continue
            num_objs = len(current_objs)
            num_dets = len(det_boxes)
            if num_objs > 0:
                dist_matrix = np.zeros((num_objs, num_dets))
                team_matrix = np.zeros((num_objs, num_dets))
                for i, obj in enumerate(current_objs):
                    obj_center = self._get_center(obj.bbox)
                    for j, det in enumerate(det_boxes):
                        (x1, y1), (x2, y2), score = det
                        det_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                        # 距離
                        dist = np.hypot(obj_center[0] - det_center[0], obj_center[1] - det_center[1])
                        dist_matrix[i, j] = dist
                        # 隊伍強制過濾
                        if cls_name == "person":
                            new_color = get_dominant_color(frame, (x1, y1, x2, y2))
                            det_team_color = classify_color_bgr(new_color)
                            team_matrix[i, j] = 0 if obj.team_color == det_team_color else 1e6
                        else:
                            team_matrix[i, j] = 0
                cost_matrix = dist_matrix + team_matrix
                _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=80)
                det_matched = set()
                obj_matched = set()
                for i, det_idx in enumerate(x):
                    if det_idx >= 0 and cost_matrix[i, det_idx] < 1e5:  # 可配對
                        obj = current_objs[i]
                        (x1, y1), (x2, y2), score = det_boxes[det_idx]
                        obj.update_bbox((x1, y1, x2, y2))
                        obj.last_seen = frame_idx
                        obj.appear_history.append(frame_idx)
                        obj.score = score
                        if cls_name == "person":
                            obj.dominant_color = get_dominant_color(frame, (x1, y1, x2, y2))
                            obj.team_color = classify_color_bgr(obj.dominant_color)
                        if not obj.confirmed and len(obj.appear_history) >= CONFIRMATION_FRAMES:
                            count = sum(
                                1 for i in obj.appear_history[-CONFIRMATION_FRAMES:]
                                if i >= frame_idx - CONFIRMATION_FRAMES
                            )
                            if count >= MIN_CONFIRM_COUNT:
                                obj.confirmed = True
                        matched_ids.add(obj.id)
                        obj_matched.add(i)
                        det_matched.add(det_idx)
                for j, det in enumerate(det_boxes):
                    if j not in det_matched:
                        (x1, y1), (x2, y2), score = det
                        new_color = get_dominant_color(frame, (x1, y1, x2, y2)) if cls_name == "person" else (0,0,0)
                        det_team_color = classify_color_bgr(new_color) if cls_name == "person" else (0,0,0)
                        new_obj = TrackedObject(
                            id=self.next_id,
                            cls=cls_name,
                            bbox=(x1, y1, x2, y2),
                            last_seen=frame_idx,
                            appear_history=[frame_idx],
                            dominant_color=new_color,
                            team_color=det_team_color,
                            score=score
                        )
                        self.tracked_objects[self.next_id] = new_obj
                        matched_ids.add(self.next_id)
                        self.next_id += 1
            else:
                for det in det_boxes:
                    (x1, y1), (x2, y2), score = det
                    new_color = get_dominant_color(frame, (x1, y1, x2, y2)) if cls_name == "person" else (0,0,0)
                    det_team_color = classify_color_bgr(new_color) if cls_name == "person" else (0,0,0)
                    new_obj = TrackedObject(
                        id=self.next_id,
                        cls=cls_name,
                        bbox=(x1, y1, x2, y2),
                        last_seen=frame_idx,
                        appear_history=[frame_idx],
                        dominant_color=new_color,
                        team_color=det_team_color,
                        score=score
                    )
                    self.tracked_objects[self.next_id] = new_obj
                    matched_ids.add(self.next_id)
                    self.next_id += 1

        for obj_id, obj in self.tracked_objects.items():
            if frame_idx - obj.last_seen > 10:
                obj.status = "Lost"
            else:
                obj.status = "Steady"

        return [
            obj for obj_id, obj in self.tracked_objects.items()
            if obj.confirmed and obj_id in matched_ids
        ]

# --- 最靠近球的球員 ---
def get_closest_person_id(ball_obj, tracked_objects):
    cx, cy = (ball_obj.bbox[0] + ball_obj.bbox[2]) / 2, (ball_obj.bbox[1] + ball_obj.bbox[3]) / 2
    min_dist = float("inf")
    closest_id = None
    for obj in tracked_objects:
        if obj.cls == "person" and obj.status == "Steady":
            px, py = (obj.bbox[0] + obj.bbox[2]) / 2, (obj.bbox[1] + obj.bbox[3]) / 2
            dist = np.hypot(cx - px, cy - py)
            if dist < min_dist:
                min_dist = dist
                closest_id = obj.id
    return closest_id

# --- 主程式流程 ---
with open(TRACKING_JSON, "r") as f:
    tracking_data = json.load(f)

cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# --- 變數---
tracker = EuclideanTracker()
frame_idx = 0
last_ball_owner_id = None
ball_owner_changes = []
last_contact_id = None 
contact_frames = 0  
# --- 龍門---
goal_contact_frames = 0
ball_inside_gate = False 
goals = [] 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    key = str(frame_idx)
    detections = {"person": [], "sports ball": [], "gate": []}

    if key in tracking_data:
        frame_data = tracking_data[key]
        if "person" in frame_data:
            for (x1, y1), (x2, y2), score in frame_data["person"]:
                if score >= PERSON_THRESHOLD:
                    detections["person"].append(((x1, y1), (x2, y2), score))

        if "sports ball" in frame_data:
            ball_candidates = [
                ((x1, y1), (x2, y2), score) for (x1, y1), (x2, y2), score in frame_data["sports ball"]
                if score >= BALL_THRESHOLD
            ]
            if ball_candidates:
                best_ball = max(ball_candidates, key=lambda x: x[2])
                detections["sports ball"].append((best_ball[0], best_ball[1], best_ball[2]))
        if "gate" in frame_data:
            for (x1, y1), (x2, y2), score in frame_data["gate"]:
                detections["gate"].append(((x1, y1), (x2, y2), score))
    tracked = tracker.update(detections, frame_idx, frame=frame)
    
    gate_objs = [obj for obj in tracked if obj.cls == "gate"]
    ball_objs = [obj for obj in tracked if obj.cls == "sports ball"]
    
    if ball_objs:
        ball_obj = ball_objs[0]
        closest_person_id = get_closest_person_id(ball_obj, tracked)
        closest_person_obj = next((o for o in tracked if o.id == closest_person_id), None)

        if closest_person_obj:
            is_contact = bboxes_intersect(ball_obj.bbox, closest_person_obj.bbox)
            if is_contact and closest_person_id == last_contact_id:
                contact_frames += 1
            elif is_contact:
                contact_frames = 1
                last_contact_id = closest_person_id
            else:
                contact_frames = 0
                last_contact_id = None

            if contact_frames >= 1 and closest_person_id != last_ball_owner_id:
                last_owner_obj = next((o for o in tracked if o.id == last_ball_owner_id), None)
                curr_owner_obj = closest_person_obj
                if last_owner_obj and curr_owner_obj:
                    if last_owner_obj.team_color != curr_owner_obj.team_color:
                        ball_owner_changes.append((frame_idx, last_ball_owner_id, closest_person_id))
                        cv2.putText(
                            frame,
                            f"Ball Possession Change ({last_ball_owner_id} to {closest_person_id})",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2
                        )
                else:
                    ball_owner_changes.append((frame_idx, last_ball_owner_id, closest_person_id))
                last_ball_owner_id = closest_person_id
                
    if gate_objs and ball_objs:
        gate_obj = gate_objs[0]
        ball_obj = ball_objs[0]
        in_gate = bbox_fully_inside(ball_obj.bbox, gate_obj.bbox)
        if in_gate:
            goal_contact_frames += 1
        else:
            goal_contact_frames = 0

        # 檢查進球判斷
        if not ball_inside_gate and goal_contact_frames >= 3:
            print(f"進球: frame={frame_idx}")
            goals.append((frame_idx, ball_obj.id))
            ball_inside_gate = True 
        elif not in_gate:
            ball_inside_gate = False
    else:
        goal_contact_frames = 0
        ball_inside_gate = False
                
    for obj in tracked:
        x1, y1, x2, y2 = obj.bbox

        if obj.cls == "gate":
            color = (128, 255, 0) 
            label = "GATE"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        elif obj.cls == "person":
            color = obj.team_color
            hsv = cv2.cvtColor(np.uint8([[obj.dominant_color]]), cv2.COLOR_BGR2HSV)[0][0]
            label = f"ID:{obj.id} {obj.dominant_color} "
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + 20, y1), obj.dominant_color, -1)
        elif obj.cls == "sports ball":
            color = (0, 255, 0)
            label = f"Ball {obj.score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        '''''
        if obj.center and obj.prev_center:
            cx, cy = obj.center
            px, py = obj.prev_center
            if abs(cx - px) + abs(cy - py) > 5:
                cv2.arrowedLine(frame, (px, py), (cx, cy), (0,255,255), 2, tipLength=0.3)
                angle = np.degrees(np.arctan2(cy - py, cx - px))
                cv2.putText(frame, f"{angle:.0f} deg", (cx, cy+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        '''''
    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

print("\n球權變化紀錄（frame_idx, from_id, to_id）:")
for event in ball_owner_changes:
    print(event)
print("\n進球紀錄（frame_idx, ball_id）:")
for goal in goals:
    print(goal)

print("影片已儲存：", OUTPUT_VIDEO)
