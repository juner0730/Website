# backend/Dockerfile
FROM python:3.10-slim

# 安裝必要工具
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# 設定工作目錄
WORKDIR /app

# 安裝依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製 backend 所有檔案
COPY . .

# 開發用啟動 FastAPI server（熱更新用 --reload）
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
