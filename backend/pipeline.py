import subprocess
import sys
from pathlib import Path
import torch

if not torch.cuda.is_available():
    print("❌ 未偵測到 GPU，可用 CUDA 無法使用，請確認驅動與環境設定正確！")
    sys.exit(1)
else:
    print(f"✅ 偵測到 GPU：{torch.cuda.get_device_name(0)}")

if len(sys.argv) < 2:
    print("請提供影片路徑作為參數，例如：python pipeline.py media/raw/abc.mp4")
    sys.exit(1)

video_path = Path(sys.argv[1])
video_name = video_path.stem
output_dir = Path("outputs") / video_path.stem
output_dir.mkdir(parents=True, exist_ok=True)

# 1. 執行 AlphaPose（人體骨架偵測）
print("[1/5] 執行 alphapose.py（demo_inference）...")
subprocess.run([
    "python", "Alphapose/scripts/demo_inference.py",
    "--cfg", "Alphapose/configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml",
    "--checkpoint", "Alphapose/pretrained_models/halpe26_fast_res50_256x192.pth",
    "--video", str(video_path),
    "--save_video"
])

default_output_json = Path("alphapose-results.json")  # 或 outputs/alphapose-results.json
default_output_video = Path("AlphaPose_" + video_path.name)

if default_output_json.exists():
    shutil.move(str(default_output_json), output_dir / "alphapose-results.json")

if default_output_video.exists():
    shutil.move(str(default_output_video), output_dir / f"AlphaPose_{video_path.name}")

# 2. 執行 tracking.py
print("[2/5] 執行 tracking.py...")
subprocess.run(["python", "tracking.py", str(video_path), str(output_dir)])

# 3. 執行背號偵測
print("[3/5] 執行 number.py...")
subprocess.run(["python", "number.py", str(video_path), str(output_dir)])

# 4. 整合 JSON 資料
print("[4/5] 執行 combined.py...")
subprocess.run(["python", "combined.py", str(output_dir)])

# 5. 剪輯 highlight
print("[5/5] 執行 highlight_editor.py...")
subprocess.run(["python", "highlight_editor.py", str(output_dir)])

print("✅ 處理完成：", video_path.name)
