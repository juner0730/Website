# app.py
import os, json, cv2, time, subprocess
from collections import defaultdict
from tqdm import tqdm

from config import *
from tracking.tracker import EuclideanTracker
from tracking.geometry import bboxes_intersect, bbox_fully_inside
from events.segments import merge_possession_changes, build_possession_segments, attach_goal_segments
from vision.jersey import detect as detect_jersey
from io_utils.video import save_clip
from io_utils.logging_utils import open_jersey_csv, write_stats_txt
from vision.color import classify_team_color_by_hist

def resolve_final_jersey(jersey_stats, min_votes=5, min_avg_conf=0.40):
    final = {}
    for pid, num_dict in jersey_stats.items():
        cand = []
        for num, confs in num_dict.items():
            count = len(confs)
            avg_c = sum(confs)/count if count else 0.0
            cand.append((str(num), count, avg_c))
        cand.sort(key=lambda x: (x[1], x[2]), reverse=True)
        chosen = None
        for num, count, avg_c in cand:
            if count >= min_votes and avg_c >= min_avg_conf:
                chosen = num
                break
        final[pid] = chosen  # 可能是 None
    return final

def _label_team_color(bgr):
    b,g,r = bgr
    if r>max(g,b)+40: return "RED"
    if b>max(r,g)+40: return "BLUE"
    if g>max(r,b)+40: return "GREEN"
    if r>200 and g>200 and b>200: return "WHITE"
    if r<60 and g<60 and b<60: return "BLACK"
    return "OTHER"

# --- IoU 工具（給 hint 對齊用） ---
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1); ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def export_highlights_by_jersey(
    segments, frame_view_meta,
    final_jersey_map,
    tracker_obj_index, 
    video_path, fps, out_size, total_frames,
    base_dir,
    do_concat=False
):
    buckets = defaultdict(lambda: defaultdict(list))
    for seg in segments:
        pid = seg["to_id"]
        jersey = final_jersey_map.get(pid)
        jersey_label = "unknown" if jersey is None else jersey
        team_bgr = tracker_obj_index.get(pid, (0,0,0))
        jersey_team = f"{jersey_label}_{_label_team_color(team_bgr)}"
        tag = seg.get("tag","possession")
        buckets[jersey_team][pid].append({
            "pid": pid, "start": seg["start"], "end": seg["end"], "tag": tag
        })

    for jersey_team, by_pid in buckets.items():
        concat_dir = os.path.join(base_dir, "by_jersey", jersey_team)
        os.makedirs(concat_dir, exist_ok=True)
        final_clip_paths = []
        for pid, items in by_pid.items():
            if not items: 
                continue
            items.sort(key=lambda x: x["start"])
            merged = [items[0]]
            for i in range(1, len(items)):
                prev = merged[-1]; curr = items[i]
                if curr["start"] <= prev["end"]:
                    prev["end"] = max(prev["end"], curr["end"])
                else:
                    merged.append(curr)
            folder = os.path.join(concat_dir, f"pid_{pid:03d}")
            os.makedirs(folder, exist_ok=True)
            for m in merged:
                final_out_path = os.path.join(
                    folder, f"{m['start']:06d}_{m['end']:06d}_{m['tag']}_pid{pid:03d}.mp4"
                )
                ok = save_clip(
                    video_path, m["start"], m["end"], final_out_path, 
                    fps, out_size, view_meta=frame_view_meta
                )
                if ok:
                    final_clip_paths.append(final_out_path)

        list_path = os.path.join(concat_dir, f"{jersey_team}_concat.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for p in final_clip_paths:
                f.write(f"file '{p}'\n")

        if do_concat and len(final_clip_paths) >= 2:
            output_path = os.path.join(concat_dir, f"{jersey_team}.mp4")
            try:
                subprocess.run(
                    [FFMPEG_PATH, "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", output_path],
                    check=True
                )
                print(f"[OK] 合併完成：{output_path}")
            except Exception as e:
                print(f"[WARN] 合併失敗（已保留清單）：{list_path}，錯誤：{e}")

# ========= 初始化 =========
ANALYSIS_VIDEO_PATH = PROC_VIDEO_PATH if (PROC_VIDEO_PATH and os.path.exists(PROC_VIDEO_PATH)) else RAW_VIDEO_PATH

with open(TRACKING_JSON, "r") as f:
    tracking_data = json.load(f)

# proc video
cap = cv2.VideoCapture(ANALYSIS_VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {ANALYSIS_VIDEO_PATH}")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# raw video
cap_raw = cv2.VideoCapture(RAW_VIDEO_PATH)
if not cap_raw.isOpened():
    raise RuntimeError(f"Cannot open raw video: {RAW_VIDEO_PATH}")
width_raw  = int(cap_raw.get(cv2.CAP_PROP_FRAME_WIDTH))
height_raw = int(cap_raw.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_raw    = cap_raw.get(cv2.CAP_PROP_FPS) or fps
cap_raw.release()

if (width, height) != (width_raw, height_raw):
    print(f"[WARN] 分析用影片尺寸 {width}x{height} 與原始 {width_raw}x{height_raw} 不一致，"
          "請確認 tracking JSON 與兩段影片解析度一致以避免位移。")
if abs(fps - fps_raw) > 1e-3:
    print(f"[WARN] 分析用 fps={fps:.3f} 與原始 fps={fps_raw:.3f} 不一致，"
          "用原始 fps 進行剪輯。")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

# ========= Pass 1：位置追蹤 + 顏色投票，產生 per-frame hints =========
def build_detections_for_frame(frame_idx):
    key = str(frame_idx)
    detections = {"person": [], "sports ball": [], "gate": []}
    if key in tracking_data:
        data = tracking_data[key]
        for cls_ in ("person","sports ball","gate"):
            if cls_ in data:
                for (x1,y1),(x2,y2),score in data[cls_]:
                    thr = PERSON_THRESHOLD if cls_=="person" else BALL_THRESHOLD if cls_=="sports ball" else 0
                    if score >= thr:
                        detections[cls_].append(((x1,y1),(x2,y2),score))
        if detections["sports ball"]:
            best = max(detections["sports ball"], key=lambda x: x[2])
            detections["sports ball"] = [best]
    return detections

# 讀一遍影片，做 Pass1
cap_pass1 = cv2.VideoCapture(ANALYSIS_VIDEO_PATH)
color_votes = defaultdict(lambda: defaultdict(int))      # pid -> {BGR: count}
pass1_boxes_by_frame = defaultdict(list)                 # frame_idx -> [(bbox_xyxy, pid)]
# 連續色彩投票狀態
streak_state = defaultdict(lambda: {"last": None, "len": 0})

tracker_pass1 = EuclideanTracker(
    use_color_cost=False,  # 不用顏色成本
    det_color_hint_fn=None
)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
pass1_path = os.path.join(os.path.dirname(OUTPUT_VIDEO), "pass1.mp4")
out_pass1 = cv2.VideoWriter(pass1_path, fourcc, fps, (width, height))

pbar1 = tqdm(total=total_frames if total_frames > 0 else None, desc="Pass1 (collect colors)", unit="frame")
frame_idx = 0
while True:
    ret, frame = cap_pass1.read()
    if not ret: break
    detections = build_detections_for_frame(frame_idx)
    tracked = tracker_pass1.update(detections, frame_idx, frame)

    # 統計顏色 + 記錄本幀 bbox
    for o in tracked:
        if o.cls != "person": 
            continue
        x1,y1,x2,y2 = o.bbox
        pass1_boxes_by_frame[frame_idx].append(((x1,y1,x2,y2), o.id))
        # 顏色投票（每幀投一次；若效能吃緊可改為每2幀投一次）
        bgr = classify_team_color_by_hist(frame, o.bbox)

        state = streak_state[o.id]
        if state["last"] is None or bgr != state["last"]:
            # 顏色變了，重新起算
            state["last"] = bgr
            state["len"]  = 1
        else:
            # 顏色延續
            state["len"] += 1

        # 只有連續滿 3 幀才開始投票（第3幀起每幀+1）
        if state["len"] >= 3:
            color_votes[o.id][bgr] += 1
        draw_color = bgr if bgr != (0, 0, 0) else (0, 255, 255)
        label = f"{o.cls} ID:{o.id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 1)

    # 把加了框的這一幀寫進 pass1.mp4
    out_pass1.write(frame)
    frame_idx += 1
    pbar1.update(1)
out_pass1.release()
print(f"Pass1 視訊已輸出：{pass1_path}")

cap_pass1.release()
try:
    pbar1.close()
except:
    pass

# 選每個 id 的代表色
pass1_color_by_id = {}
for pid, votes in color_votes.items():
    # 這裡可用你加強版的挑色器，示意用最簡：
    top = max(votes.items(), key=lambda kv: kv[1])[0] if votes else (0,0,0)
    pass1_color_by_id[pid] = top

# 建立「精確座標對應」的 per-frame hint map
hint_map_by_frame = {}  # frame_idx -> dict[(x1,y1,x2,y2)] = color_bgr
for fidx, items in pass1_boxes_by_frame.items():
    d = {}
    for (bbox_xyxy, pid) in items:
        color = pass1_color_by_id.get(pid, (0,0,0))
        # 確保是整數 tuple，避免型別差異
        x1,y1,x2,y2 = map(int, bbox_xyxy)
        d[(x1,y1,x2,y2)] = color
    hint_map_by_frame[fidx] = d

# 生成提供給 tracker 的 hint 函式（用 IoU 對齊）
def make_det_color_hint_fn(hint_map_by_frame):
    def _fn(tracker_obj, det_boxes):
        fidx = getattr(tracker_obj, "current_frame_idx", None)
        if fidx is None: 
            return [None] * len(det_boxes)
        d = hint_map_by_frame.get(fidx, {})
        out = []
        for (p1, p2), (p3, p4), _ in det_boxes:
            key = (int(p1), int(p2), int(p3), int(p4))
            out.append(d.get(key))  # 命中才有顏色，沒命中就 None
        return out
    return _fn


det_color_hint_fn = make_det_color_hint_fn(hint_map_by_frame)

# ========= Pass 2：使用 hints + 嚴格 team color 追蹤 + 事件判定 =========
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
log_file, csv_writer = open_jersey_csv(JERSEY_CSV)
jersey_stats = defaultdict(lambda: defaultdict(list))

tracker = EuclideanTracker(
    use_color_cost=True,
    det_color_hint_fn=det_color_hint_fn,
    team_penalty_strict=1e6,
    team_penalty_relaxed=200.0
)

frame_idx = 0
last_ball_owner_id, last_contact_id = None, None
contact_frames, goal_contact_frames = 0, 0
ball_inside_gate = False
ball_owner_changes, goals = [], []
frame_view_meta = {}
contact_need = max(1, int(CONTACT_MIN_SEC * fps))  # 按fps計算sec相對的frame
goal_need    = max(1, int(GOAL_MIN_SEC * fps))     # 承上

cap_pass2 = cv2.VideoCapture(ANALYSIS_VIDEO_PATH)
pbar = tqdm(total=total_frames if total_frames > 0 else None, desc="Pass2 (tracking+events)", unit="frame")

while True:
    ret, frame = cap_pass2.read()
    if not ret: break

    detections = build_detections_for_frame(frame_idx)
    tracked = tracker.update(detections, frame_idx, frame)
    gate_objs = [o for o in tracked if o.cls=="gate"]
    ball_objs = [o for o in tracked if o.cls=="sports ball"]

    if ball_objs:
        ball = ball_objs[0]
        cx, cy = ((ball.bbox[0]+ball.bbox[2])/2, (ball.bbox[1]+ball.bbox[3])/2)
        min_d, closest = float("inf"), None
        for o in tracked:
            if o.cls=="person" and o.status=="Steady":
                px, py = ((o.bbox[0]+o.bbox[2])/2, (o.bbox[1]+o.bbox[3])/2)
                d = ((cx-px)**2 + (cy-py)**2) ** 0.5
                if d < min_d: min_d, closest = d, o

        if closest:
            if bboxes_intersect(ball.bbox, closest.bbox):
                if closest.id == last_contact_id: contact_frames += 1
                else: contact_frames, last_contact_id = 1, closest.id
            else:
                contact_frames, last_contact_id = 0, None

            if last_ball_owner_id is None and contact_frames >= contact_need:
                last_ball_owner_id = closest.id
                ball_owner_changes.append((frame_idx, None, last_ball_owner_id))
            elif contact_frames >= contact_need and closest.id != last_ball_owner_id:
                ball_owner_changes.append((frame_idx, last_ball_owner_id, closest.id))
                last_ball_owner_id = closest.id

            bx1, by1, bx2, by2 = ball.bbox
            ball_cx = (bx1 + bx2) // 2
            ball_cy = (by1 + by2) // 2
            holder_width = max(1, closest.width)
            frame_view_meta[frame_idx] = {
                "ball_center": (int(ball_cx), int(ball_cy)),
                "holder_width": int(holder_width) 
            }

    # 進球判定
    if gate_objs and ball_objs:
        gate, ball = gate_objs[0], ball_objs[0]
        in_gate = any(bbox_fully_inside(ball.bbox, g.bbox) for g in gate_objs)
        if in_gate: 
            goal_contact_frames += 1
        else: 
            goal_contact_frames, ball_inside_gate = 0, False

        if not ball_inside_gate and goal_contact_frames >= goal_need:
            goals.append((frame_idx, ball.id))
            ball_inside_gate = True
    else:
        goal_contact_frames, ball_inside_gate = 0, False

    # 繪製與背號（仍預設關閉）
    for o in tracked:
        x1,y1,x2,y2 = o.bbox
        color = (128,255,0) if o.cls=="gate" else (0,255,0) if o.cls=="sports ball" else o.team_color
        label = "GATE" if o.cls=="gate" else ("Ball %.2f"%o.score if o.cls=="sports ball" else f"ID:{o.id} ")
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        cv2.putText(frame,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5 if o.cls!="gate" else 1,color,2)

        # 若之後啟用背號偵測，取消註解即可
        # if o.cls=="person":
        #     res = detect_jersey(frame, o.bbox)
        #     if res:
        #         number, confs = res
        #         if number:
        #             cv2.putText(frame, f"No:{number}", (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        #             csv_writer.writerow([frame_idx, o.id, number, confs])
        #             jersey_stats[o.id][number].extend(confs)

    # 防止 save_clip KeyError
    if frame_idx not in frame_view_meta:
        frame_view_meta[frame_idx] = {"ball_center": None, "holder_width": None}

    out.write(frame)
    frame_idx += 1

    if frame_idx % 300 == 0:
        pbar.set_postfix({"owner_changes": len(ball_owner_changes), "goals": len(goals)})
    pbar.update(1)

# 資源釋放
cap_pass2.release(); out.release(); log_file.close()
try:
    pbar.close()
except:
    pass

# 事件→片段
ball_owner_changes.sort(key=lambda x: x[0])
merged = merge_possession_changes(ball_owner_changes, window=MERGE_WINDOW)
pos_segs = build_possession_segments(merged, total_frames, pre_margin=60, post_margin=60)
goal_segs = attach_goal_segments(goals, merged, total_frames, margin=30, default_owner=last_ball_owner_id)

# 高光輸出
export_count = 0
segments_all = pos_segs + goal_segs
print(
    f"[DEBUG] raw_changes={len(ball_owner_changes)}, "
    f"merged_changes={len(merged)}, "
    f"pos_segments={len(pos_segs)}, "
    f"goal_segments={len(goal_segs)}, "
    f"total_segments={len(segments_all)}"
)
for i, s in enumerate(pos_segs, 1):
    print(f"[POS {i}] owner={s['to_id']}, {s['start']}->{s['end']}, tag={s.get('tag','possession')}")

for seg in segments_all:
    folder = os.path.join(HIGHLIGHT_DIR, f"{seg['to_id']}")
    os.makedirs(folder, exist_ok=True)
    tag = seg.get("tag", "possession")
    out_path = os.path.join(folder, f"{seg['start']:06d}_{seg['end']:06d}_{tag}.mp4")
    ok = save_clip(RAW_VIDEO_PATH, seg["start"], seg["end"], out_path, fps_raw, (width, height), view_meta=frame_view_meta)
    export_count += int(ok)
print(f"總計輸出精華片段：{export_count} 段")

# 背號決策
final_jersey_map = resolve_final_jersey(jersey_stats, min_votes=5, min_avg_conf=0.40)

# 取得每位球員的代表隊色（以 Pass2 tracker 最終顏色）
tracker_obj_index = {}
for obj in tracker.tracked_objects.values():
    if obj.cls == "person":
        tracker_obj_index[obj.id] = obj.team_color  # BGR tuple

# 依背號分群輸出
_ = export_highlights_by_jersey(
    segments=segments_all,
    frame_view_meta=frame_view_meta,
    final_jersey_map=final_jersey_map,
    tracker_obj_index=tracker_obj_index,
    video_path=RAW_VIDEO_PATH,
    fps=fps_raw,
    out_size=(width_raw, height_raw),
    total_frames=total_frames,
    base_dir=HIGHLIGHT_DIR
)

# 背號統計
write_stats_txt(JERSEY_STATS_TXT, jersey_stats)
print("背號統計檔已輸出：", JERSEY_STATS_TXT)
