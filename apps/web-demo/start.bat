@echo off
chcp 65001 >nul
cd /d %~dp0
rem 需要 Python 3.10 及以上（只用标准库，无需 pip install）。启用大模型复核：先 set VLM_API_KEY=... 再运行本文件
start "" http://localhost:8000
python server.py
pause