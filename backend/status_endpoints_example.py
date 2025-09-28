# backend/status_endpoints_example.py
# Drop-in endpoints to expose duration on your website
import os, json, time
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

UPLOAD_BASE_DIR = os.getenv("UPLOAD_BASE_DIR", "/data/uploads")

def email_to_dir(email: str) -> str:
    return email.replace("@", "_at_")

router = APIRouter()

def _meta_path(email: str, video_id: str):
    email_at = email_to_dir(email)
    base = os.path.join(UPLOAD_BASE_DIR, email_at, os.path.splitext(video_id)[0])
    return os.path.join(base, "_job.json")

@router.get("/api/process/status/{video_id}")
def get_status(video_id: str, user = Depends(lambda: {"email":"you@example.com"})):  # replace Depends(...) with your auth
    p = _meta_path(user["email"], video_id)
    if not os.path.isfile(p):
        raise HTTPException(404, "No status found")
    data = json.load(open(p, "r", encoding="utf-8"))
    # if still running, compute live elapsed
    if data.get("status") == "running" and data.get("start_ts"):
        data["duration_sec"] = int(time.time()) - int(data["start_ts"])
    return JSONResponse(data)
