from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Optional

import httpx
from fastapi import FastAPI, Depends, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from jose import jwt
from pydantic import BaseModel

# === 環境變數 ===
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

JWT_SECRET = os.getenv("JWT_SECRET", "dev-only-secret-change-me")
JWT_ISSUER = "website.local"
ALLOWED_EMAILS = [x.strip() for x in os.getenv("ALLOWED_EMAILS", "").split(",") if x.strip()]

# === FastAPI & CORS ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Models ===
class User(BaseModel):
    sub: str
    email: str
    name: Optional[str] = None
    picture: Optional[str] = None

# === Session helpers ===
SESSION_COOKIE_NAME = "token"

def make_session_jwt(user: User) -> str:
    payload = {
        "iss": JWT_ISSUER,
        "sub": user.sub,
        "email": user.email,
        "name": user.name,
        "picture": user.picture,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def read_session_jwt(token: str) -> User:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"], options={"verify_aud": False})
        return User(
            sub=str(payload.get("sub")),
            email=str(payload.get("email")),
            name=payload.get("name"),
            picture=payload.get("picture"),
        )
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid session token: {e}")

def is_email_allowed(email: str) -> bool:
    if not ALLOWED_EMAILS:
        return True
    email = email.lower()
    for rule in ALLOWED_EMAILS:
        r = rule.lower()
        if r.startswith("@") and email.endswith(r):
            return True
        if email == r:
            return True
    return False

def get_current_user(request: Request) -> User:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="Missing session")
    return read_session_jwt(token)

# === OAuth: start ===
@app.get("/auth/google/start")
def google_start(next: Optional[str] = None):
    scope = "openid email profile"
    state = next or FRONTEND_ORIGIN
    auth_url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={GOOGLE_REDIRECT_URI}"
        f"&response_type=code"
        f"&scope={httpx.QueryParams({'scope': scope})['scope']}"
        f"&state={state}"
        f"&access_type=offline&include_granted_scopes=true&prompt=consent"
    )
    return RedirectResponse(auth_url)

# === OAuth: callback ===
@app.get("/auth/google/callback")
async def google_callback(code: str, state: Optional[str] = None):
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code",
    }
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(token_url, data=data)
            r.raise_for_status()
            tok = r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=401, detail=f"Token exchange failed: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token exchange error: {e}")

    id_token = tok.get("id_token")
    if not id_token:
        raise HTTPException(status_code=401, detail="Missing id_token from Google")

    try:
        claims = jwt.get_unverified_claims(id_token)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid id_token: {e}")

    if claims.get("aud") != GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=401, detail="Wrong audience")
    email = claims.get("email")
    if not email:
        raise HTTPException(status_code=401, detail="No email in id_token")
    if not is_email_allowed(email):
        raise HTTPException(status_code=403, detail="Account not allowed")

    user = User(sub=str(claims.get("sub")), email=email, name=claims.get("name"), picture=claims.get("picture"))
    session_jwt = make_session_jwt(user)

    redirect_to = state or FRONTEND_ORIGIN
    resp = RedirectResponse(redirect_to, status_code=302)
    resp.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_jwt,
        httponly=True,
        samesite="lax",
        secure=False,  # 本機開發用；上線請改 True + HTTPS
        path="/",
        max_age=60 * 60 * 24 * 7,  # 7 天
    )
    return resp

# === Session APIs ===
@app.get("/me")
def me(user: User = Depends(get_current_user)):
    return user

@app.post("/logout")
def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return resp

# === Minimal: 上傳 / 列表 / 靜態檔案 ===
BASE_DIR = os.getenv("UPLOAD_BASE_DIR", "./uploads")
os.makedirs(BASE_DIR, exist_ok=True)

def _user_dir(user: User) -> str:
    safe = user.email.replace("@", "_at_")
    d = os.path.join(BASE_DIR, safe)
    os.makedirs(d, exist_ok=True)
    return d

def _file_info(path: str):
    st = os.stat(path)
    return {
        "id": os.path.basename(path),
        "original_name": os.path.basename(path),
        "status": "uploaded",
        "created_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
    }

@app.post("/upload")
@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...), user: User = Depends(get_current_user)):
    user_dir = os.path.join(UPLOAD_BASE_DIR, email_to_dir(user["email"]))
    os.makedirs(user_dir, exist_ok=True)

    target_name = make_target_filename(file.filename)
    
    safe_name = f"{uuid.uuid4().hex}_{file.filename}"
    dst = os.path.join(user_dir, safe_name)

    with open(dst, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return {"ok": True, "file": _file_info(dst)}

@app.get("/videos")
@app.get("/api/videos")
def list_videos(user: User = Depends(get_current_user)):
    user_dir = _user_dir(user)
    items = []
    for name in sorted(os.listdir(user_dir), reverse=True):
        p = os.path.join(user_dir, name)
        if os.path.isfile(p):
            items.append({
                "id": name,
                "original_name": strip_ts_for_display(name),
                "status": "uploaded", 
                "created_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
            })
    items.sort(key=lambda x: x["create_at"], reverse=True)
    return items

# 讓瀏覽器可直接讀取上傳檔案（未來預覽/下載）
app.mount("/files", StaticFiles(directory=BASE_DIR), name="files")
