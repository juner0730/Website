五人制足球AI運動員追蹤與精彩時刻捕捉系統

**本專案能做什麼
- **Google 登入**（僅白名單帳號可登入）
- **上傳影片**
- **查看個人上傳清單**
- （未來）自動產生精華剪輯

**事前準備
- 安裝 **Docker & Docker Compose**
- 申請一組 **Google OAuth 2.0** 用戶端（類型：*Web application*）

> 若尚未安裝 Docker：<https://docs.docker.com/get-docker/>


1) 取得程式碼並建立資料夾
```bash
git clone https://github.com/juner0730/Website.git
cd Website
mkdir -p data/uploads         # 上傳檔案會存到這裡（宿主機）
**如果想要自己設定後端環境變數
檔案：backend/.env
GOOGLE_CLIENT_ID=你的_CLIENT_ID
GOOGLE_CLIENT_SECRET=你的_CLIENT_SECRET
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback
FRONTEND_ORIGIN=http://localhost:5173
ALLOWED_EMAILS=you@example.com            # 或 @yourcompany.com（整個網域）
JWT_SECRET=請自行更換的隨機字串
UPLOAD_BASE_DIR=/data/uploads             # 由 docker volume 對應到 ./data/uploads

特別注意**
ALLOWED_EMAILS：只有列在這裡的信箱（或網域）才能登入。
FRONTEND_ORIGIN：本機開發請務必是 http://localhost:5173。
UPLOAD_BASE_DIR：建議維持 /data/uploads，檔案才會落在 ./data/uploads。


2）啟動（Docker）
docker compose up --build
前端：http://localhost:5173
後端：http://localhost:8000
**請務必用 http://localhost:5173
 開站（不要用容器 IP）。
否則瀏覽器不會把登入 Cookie 帶給後端。


3）登入
開啟 http://localhost:5173
點 「使用 Google 登入」 並完成 Google 流程
成功後會自動導向 /onepage
後端 log 會出現：GET /me 200 OK就表示登入成功
瀏覽器 DevTools → Application → Cookies → http://localhost:8000
 看到一個 token


4）上傳影片
在 /onepage 的上傳卡片上 拖曳檔案 或 點按選檔。
上傳完成後，檔案會出現在 歷史影片 清單。
檔案存在宿主機：./data/uploads/<你的_email_把@改成_at_>/檔名
直接連結（可選）：http://localhost:8000/files/<email_at_>/<檔名>


5）停止服務
docker compose down
專案結構（簡版）
  backend/
    main.py               # FastAPI（OAuth、Session、Upload、List、/files 靜態）
    requirements.txt
    .env                  # 後端環境變數
  frontend/
    src/                  # React（登入頁、onepage UI）
  data/
    uploads/              # 上傳檔案（宿主機 volume）
