# 使用官方 node 基底
FROM node:20

# 設定工作目錄
WORKDIR /app

# 複製 package.json 與 lock 檔案（加快 cache）
COPY package*.json ./

# 安裝依賴（會在 node_modules 安裝 vite）
RUN npm install

# 複製剩餘 frontend 資料
COPY --link ./src ./src
COPY --link ./index.html .
COPY --link ./tailwind.config.js .

# 啟動 vite（使用 npm script，不直接用 binary）
CMD ["vite", "--host", "0.0.0.0"]
