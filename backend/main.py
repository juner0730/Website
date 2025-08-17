# backend/main.py
import os
import json
import time
import shutil
import subprocess
from uuid import uuid4
from pathlib import Path
from typing import Set, Dict, Any, List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Google 驗證
from google.oauth2 import id_token
from google.auth.transport import requests

# ====== 路徑設定 ======
BASE_DIR = Path(__file__).resolve().parent
MEDIA_DIR = BASE_DIR / "media"
UPLOAD_DIR = MEDIA_DIR / "raw"
HIGHLIGHT_DIR = MEDIA_DIR / "highlights"
for d in (UPLOAD_DIR, HIGHLIGHT_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI()

# 靜態檔提供（前端以 /media/... 播放/下載）
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")

# CORS（前端在 5173）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Google 登入（方案 C：後端白名單） ======
class TokenRequest(BaseModel):
    token: str

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")

def _split_env(v: str) -> Set[str]:
    return {x.strip().lower() for x in v.replace(";", ",").split(",") if x.strip()}

ALLOWED_EMAILS: Set[str] = _split_env(os.getenv("ALLOWED_EMAILS", ""))
ALLOWED_DOMAINS: Set[str] = _split_env(os.getenv("ALLOWED_DOMAINS", ""))

def _is_allowed(info: Dict[str, Any]) -> bool:
    # 若兩者皆未設定，代表不限制
    if not ALLOWED_EMAILS and not ALLOWED_DOMAINS:
        return True
    email = (info.get("email") or "").lower()
    hd = (info.get("hd") or "").lower()  # Workspace 會有；個人 Gmail 多半沒有
    domain_from_email = email.split("@")[-1] if "@" in email else ""
    if ALLOWED_EMAILS and email in ALLOWED_EMAILS:
        return True
    if ALLOWED_DOMAINS:
        if domain_from_email in ALLOWED_DOMAINS:
            return True
        if hd and hd in ALLOWED_DOMAINS:
            return True
    return False

@app.post("/auth/google")
async def auth_google(data: TokenRequest):
    try:
        info = id_token.verify_oauth2_token(data.token, requests.Request(), GOOGLE_CLIENT_ID)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid Google token")
    if not _is_allowed(info):
        raise HTTPException(status_code=403, detail="Not allowed")
    return {
        "status": "success",
        "email": info.get("email", ""),
        "name": info.get("name", ""),
        "google_id": info.get("sub", ""),
    }

# 若前端目前打的是 /auth/google/callback，也讓它可用
@app.post("/auth/google/callback")
async def auth_google_callback(data: TokenRequest):
    return await auth_google(data)

# ====== 上傳與狀態 ======
# 簡易記憶體索引：{ id: {"filename": ..., "status": "processing|completed"} }
VIDEOS: Dict[str, Dict[str, Any]] = {}

def _refresh_status():
    """依據是否已有 highlight mp4，自動把狀態標為 completed。"""
    for vid, meta in list(VIDEOS.items()):
        hdir = HIGHLIGHT_DIR / vid
        if hdir.exists() and any(hdir.glob("*.mp4")):
            meta["status"] = "completed"

@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    vid = uuid4().hex  # 供前端識別的 id
    dest_dir = UPLOAD_DIR / vid
    dest_dir.mkdir(parents=True, exist_ok=True)

    original_name = Path(video.filename).name.replace(" ", "_")
    saved_path = dest_dir / original_name
    with saved_path.open("wb") as f:
        f.write(await video.read())

    # 記錄狀態（先標 processing）
    VIDEOS[vid] = {"filename": original_name, "status": "processing"}

    # 啟動你的處理流程（請在 pipeline.py 內把輸出放到 media/highlights/<vid>/）
    # 你若需要把 vid 帶給 pipeline，自行修改 pipeline 支援參數
    try:
        subprocess.Popen(["python", "pipeline.py", str(saved_path), vid])
    except Exception:
        # 備援：若 pipeline 目前只支援一個參數，就先傳影片路徑
        subprocess.Popen(["python", "pipeline.py", str(saved_path)])

    return {"message": "上傳成功，正在處理中", "id": vid, "filename": original_name}

@app.get("/videos")
def list_videos():
    _refresh_status()
    return VIDEOS

# ====== Highlights 給兩種前端頁面使用 ======
@app.get("/highlights/{video_id}")
def get_highlights_flat(video_id: str):
    """
    給 results.jsx 使用，回傳：
    { "players": [ { "player": "10", "file": "/media/highlights/<id>/player_10_clip1.mp4" }, ... ] }
    """
    hdir = HIGHLIGHT_DIR / video_id
    if not hdir.exists():
        return {"players": []}

    items: List[Dict[str, str]] = []
    for f in sorted(hdir.glob("*.mp4")):
        parts = f.stem.split("_")
        if len(parts) >= 2 and parts[0] == "player":
            player = parts[1]
            items.append({
                "player": player,
                "file": f"/media/highlights/{video_id}/{f.name}",
            })
    return {"players": items}

@app.get("/clips/{video_id}")
def get_clips_grouped(video_id: str):
    """
    給 playerresults.jsx 使用，回傳：
    { "10": ["player_10_clip1.mp4", ...], "21": [...] }
    """
    hdir = HIGHLIGHT_DIR / video_id
    if not hdir.exists():
        raise HTTPException(status_code=404, detail="找不到精華影片資料夾")

    grouped: Dict[str, List[str]] = {}
    for f in sorted(hdir.glob("*.mp4")):
        parts = f.stem.split("_")
        if len(parts) >= 2 and parts[0] == "player":
            player = parts[1]
            grouped.setdefault(player, []).append(f.name)
    return grouped

@app.get("/ping")
def ping():
    return {"ok": True}
