@echo off
chcp 65001 >nul
cd /d %~dp0
rem 需要 Python 3.10 及以上（只用标准库，无需 pip install）。启用大模型复核：把 .env.example 复制为 .env 并填写 VLM_API_KEY
start "" http://localhost:8000
python server.py
pause