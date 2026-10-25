# 离线评测环境（RTX 6000 / CUDA 12）。构建：在仓库根目录执行 docker build -f deploy/rtx/Dockerfile -t limbcheck-eval .
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip libegl1 libgles2 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && python3.11 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /app
COPY pyproject.toml README.md ./
COPY limbcheck ./limbcheck
COPY configs ./configs
COPY models ./models
COPY tests ./tests
RUN pip install -e ".[mediapipe,rtmw-gpu,sam2,dev]"

CMD ["limbcheck", "list"]
