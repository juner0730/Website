# frontend/Dockerfile
FROM node:22-alpine

WORKDIR /App

COPY package.json package-lock.json ./

RUN --mount=type=cache,target=/root/.npm \
    npm ci --prefer-offline --no-audit

COPY . .

EXPOSE 5173
CMD ["npm","run","dev","--","--host","0.0.0.0"]
