# frontend/Dockerfile
FROM node:22-alpine

WORKDIR /App

# 先只复制 lock 文件，确保依赖层可缓存
COPY package.json package-lock.json ./

# 使用 BuildKit 的 npm 缓存
RUN --mount=type=cache,target=/root/.npm \
    npm ci --prefer-offline --no-audit

# 再复制源码（改源码不会触发重新安装依赖）
COPY . .

EXPOSE 5173
# 让 Vite 监听 0.0.0.0 以便容器外访问
CMD ["npm","run","dev","--","--host","0.0.0.0"]
