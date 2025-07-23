import cv2, json
from collections import defaultdict
import chardet
from pathlib import Path
import sys

# 接收參數：輸出資料夾、影片路徑
if len(sys.argv) < 3:
    print("❗ 用法：python combined.py <output_dir> <video_path>")
    sys.exit(1)

output_dir = Path(sys.argv[1])
video_path = Path(sys.argv[2])
pose_json_path = output_dir / "alphapose.json"
track_json_path = output_dir / "tracking.json"
output_path = output_dir / "combined_overlay.mp4"

def center(b): x1,y1,x2,y2 = b; return (x1+x2)/2, (y1+y2)/2
def distance(b1, b2): x1,y1 = center(b1); x2,y2 = center(b2); return ((x1-x2)**2 + (y1-y2)**2)**0.5

def load_json_auto_encoding(path):
    with open(path, 'rb') as f:
        raw = f.read()
        encoding = chardet.detect(raw)['encoding']
    return json.loads(raw.decode(encoding))

poses = load_json_auto_encoding(pose_json_path)
tracks = load_json_auto_encoding(track_json_path)

pose_by_frame = defaultdict(list)
for p in poses:
    f = int(p["image_id"].split(".")[0])
    b = p["box"]
    p["box"] = [min(b[0],b[2]),min(b[1],b[3]),max(b[0],b[2]),max(b[1],b[3])]
    pose_by_frame[f].append(p)

pairs = [(0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,8),(8,9),(9,10),
         (8,12),(12,13),(13,14),(0,15),(0,16)]

# 開啟影片與輸出
cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)
w,h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret: break

    track = next((x for x in tracks if x["frame"] == frame_idx), None)
    if track:
        ps = pose_by_frame.get(frame_idx, [])
        for p in track.get("person", []):
            tid = p["track_id"]
            b1 = p["bbox"]
            x1, y1, x2, y2 = map(int, b1)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"ID:{tid}", (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            best = min(ps, key=lambda q: distance(q["box"], b1), default=None)
            if best and distance(best["box"], b1) < 100:
                k = best["keypoints"]
                avg_y = sum(k[i*3+1] for i in range(17)) / 17
                if avg_y < 350: continue

                for i in range(17):
                    x, y = int(k[i*3]), int(k[i*3+1])
                    cv2.circle(frame, (x, y), 2, (0,255,0), -1)

                for i,j in pairs:
                    if k[i*3+2]>0.3 and k[j*3+2]>0.3:
                        pt1 = (int(k[i*3]), int(k[i*3+1]))
                        pt2 = (int(k[j*3]), int(k[j*3+1]))
                        cv2.line(frame, pt1, pt2, (255,255,0), 1)

        for b in track.get("sports ball", []):
            x1,y1,x2,y2 = map(int, b["bbox"])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 1)

    out.write(frame)
    frame_idx += 1

    if frame_idx % 10 == 0:
        percent = frame_idx / total_frames * 100
        print(f"處理中：{frame_idx}/{total_frames} ({percent:.1f}%)", end="\r", flush=True)

cap.release()
out.release()
print("\n✅ 合成輸出完成：", output_path)