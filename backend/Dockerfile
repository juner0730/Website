FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先只复制依赖清单，最大化缓存利用且保证会执行安装
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip && pip install -r /tmp/requirements.txt

# 再复制源码
COPY . .

EXPOSE 8000
CMD ["python","-m","uvicorn","main:app","--host","0.0.0.0","--port","8000","--reload"]
