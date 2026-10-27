@echo off
chcp 65001 >nul
cd /d %~dp0
rem 需要 Python 3.10 及以上。建议先在仓库根目录运行 pip install -e ".[mediapipe,rtmw]" 启用服务端判定（首次启动下载模型，页面打不开时稍等后刷新）。启用大模型复核：把 .env.example 复制为 .env 并填写 VLM_API_KEY
start "" http://localhost:8000
python server.py
pause