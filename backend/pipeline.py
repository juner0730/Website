# 這是示範版：把輸入影片直接切出幾段假片，命名為 player_10_clipX.mp4
# 請替換為你的實際 pipeline（YOLO/AlphaPose/...）
import sys
import shutil
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
MEDIA_DIR = APP_DIR / "media"
HLS_DIR = MEDIA_DIR / "highlights"

def main():
    if len(sys.argv) < 3:
        print("usage: pipeline.py <src> <vid> [sub]")
        sys.exit(1)
    src = Path(sys.argv[1])
    vid = sys.argv[2]
    sub = sys.argv[3] if len(sys.argv) >= 4 else "public"

    out_dir = HLS_DIR / sub / vid
    out_dir.mkdir(parents=True, exist_ok=True)

    # TODO: 你的追蹤 / 剪輯流程；下方僅為示範輸出三個片段
    for i in range(1, 4):
        # 真實情況：請把剪出來的 mp4 存到 out_dir
        # 這裡用拷貝原檔來模擬輸出
        dst = out_dir / f"player_10_clip{i}.mp4"
        try:
            shutil.copyfile(src, dst)
        except Exception:
            # 如果原檔過大/格式不合，這純為示範
            with dst.open("wb") as f:
                f.write(b"\x00\x00")

if __name__ == "__main__":
    main()
