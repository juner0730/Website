# Futsal Highlights – Quick Start

A tiny web app to upload futsal videos, then (later) generate highlights.  
This README shows how to **download, configure, and run** it quickly.

---

## 1) Requirements
- **Docker & Docker Compose**
- A **Google OAuth 2.0** Client (Web application)

---

## 2) Clone & Folders
```bash
git clone <your-repo-url> futsal-highlights
cd futsal-highlights
mkdir -p data/uploads
```

---

## 3) Google OAuth Setup (one-time)
1. Go to **Google Cloud Console → APIs & Services → Credentials**.
2. Create **OAuth client ID** (type: *Web application*).
3. Add **Authorized redirect URI**:
   ```
   http://localhost:8000/auth/google/callback
   ```
4. If the app is in **Testing** mode, add your Google account under **Test users**.
5. Copy the **Client ID** and **Client Secret**.

---

## 4) Configure Environment

Create `backend/.env`:
```env
GOOGLE_CLIENT_ID=YOUR_GOOGLE_CLIENT_ID
GOOGLE_CLIENT_SECRET=YOUR_GOOGLE_CLIENT_SECRET
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback
FRONTEND_ORIGIN=http://localhost:5173
ALLOWED_EMAILS=you@example.com                 # or @yourcompany.com
JWT_SECRET=change-this-secret
UPLOAD_BASE_DIR=/data/uploads                  # mapped to ./data/uploads via docker
```

> (Optional) If you use a custom API base in the frontend, set `VITE_API_BASE` in your frontend env.  
> Defaults to `http://localhost:8000`.

---

## 5) Run (Docker)
```bash
docker compose up --build
```

- Frontend: **http://localhost:5173**
- Backend: **http://localhost:8000**

> If your repo includes `docker-compose.override.yml`, uploads are stored on your host at:  
> `./data/uploads/<your_email_at_>/`

---

## 6) Use the App
1. Open **http://localhost:5173**.
2. Click **「使用 Google 登入」** → complete Google login.  
   - Only whitelisted emails in `ALLOWED_EMAILS` can sign in.
3. You’ll be redirected to **/onepage**.
4. **Upload a video** via the upload card (drag & drop or pick a file).
5. See your uploads under **歷史影片**.  
   - Files are saved per user in `./data/uploads/<email_with _at_>`.

**Direct file access (optional):**
```
http://localhost:8000/files/<email_at_>/<filename>
```

**API endpoints (authenticated):**
- `POST /api/upload` — upload file (form field: `file`)
- `GET  /api/videos` — list uploaded files
- `GET  /me`         — current user
- `POST /logout`     — sign out

---

## 7) Common Tips
- Always open the site at **http://localhost:5173** (not container IP).  
  Otherwise the cookie won’t be sent to the backend.
- If upload fails with storage errors, ensure:
  - You created `data/uploads/`
  - Docker volume is mounted
  - Disk has free space

---

## 8) Project Structure (high level)
```
backend/
  main.py               # FastAPI app (OAuth, session, upload, list, static /files)
  requirements.txt
  .env                  # your backend config (create this)
frontend/
  src/                  # React app (login, onepage UI)
data/
  uploads/              # uploaded files (created on first run)
```

---

## 9) Stop
```bash
docker compose down
```

---

That’s it. Have fun!
