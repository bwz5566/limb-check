#!/bin/sh
# 需要 Python 3.10 及以上（只用标准库，无需 pip install）。启用大模型复核：先 export VLM_API_KEY=... 再运行本文件
cd "$(dirname "$0")"
(sleep 1 && (open http://localhost:8000 2>/dev/null || xdg-open http://localhost:8000 2>/dev/null)) &
python3 server.py
