# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import shutil
import subprocess
import time
import json

# Google 驗證相關
from google.oauth2 import id_token
from google.auth.transport import requests
from fastapi.middleware.cors import CORSMiddleware

HIGHLIGHT_DIR = Path("media/highlights")


app = FastAPI()

# Enable CORS for local frontend (react)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======== Google 登入驗證 API ========

class TokenRequest(BaseModel):
    token: str

GOOGLE_CLIENT_ID = "579826086181-vppevqtsk2lhi5ko5qm8du2kr0v7jmro.apps.googleusercontent.com"

@app.post("/auth/google")
async def auth_google(data: TokenRequest):
    try:
        # 驗證 Google 傳來的 ID token
        idinfo = id_token.verify_oauth2_token(
            data.token,
            requests.Request(),
            GOOGLE_CLIENT_ID
        )
        return {
            "status": "success",
            "email": idinfo["email"],
            "name": idinfo.get("name", ""),
            "google_id": idinfo["sub"]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ======== 上傳影片 API =========

UPLOAD_DIR = Path("media/raw")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/highlights/{video_id}")
def get_highlight_players(video_id: str):
    video_highlight_dir = HIGHLIGHT_DIR / video_id
    if not video_highlight_dir.exists():
        raise HTTPException(status_code=404, detail="找不到精華影片資料夾")

    player_files = sorted(video_highlight_dir.glob("*.mp4"))
    player_map = []
    for file in player_files:
        # 假設檔名格式為：player_88_clip1.mp4、player_88_clip2.mp4...
        parts = file.stem.split("_")
        if len(parts) >= 2 and parts[0] == "player":
            try:
                player_num = int(parts[1])
                player_map.append((player_num, file.name))
            except ValueError:
                continue

    # 整理成 dict：{88: [xxx.mp4, yyy.mp4], 21: [zzz.mp4]}
    result = {}
    for num, fname in player_map:
        result.setdefault(num, []).append(fname)

    return {"players": sorted(result.items())}

@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    try:
        timestamp = int(time.time())
        original_name = Path(video.filename).name.replace(" ", "_")
        filename = f"{timestamp}_{original_name}"
        video_path = UPLOAD_DIR / filename

        # 儲存影片
        with video_path.open("wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # 啟動影片處理流程（非同步）
        subprocess.Popen(["python", "pipeline.py", str(video_path)])

        return {
            "message": f"影片已上傳，處理中：{filename}",
            "video_id": video_path.stem
        }

    except Exception as e:
        return {"message": f"上傳失敗：{str(e)}"}
