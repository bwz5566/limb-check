#!/usr/bin/env bash
# 检查大模型服务是否可用，并用一张图跑一次完整的 L2 复核（提示词、解析、转换）。
# 用法：./check.sh http://10.0.0.8:8001/v1 Qwen/Qwen3-VL-8B-Instruct 手背照片.jpg
set -euo pipefail
BASE=${1:?用法：./check.sh <VLM_BASE_URL> <模型名> <手背朝前的照片>}
MODEL=${2:?缺少模型名}
IMG=${3:?缺少图片：请给一张双手手背朝前的照片}
curl -fsS "$BASE/models" | head -c 400; echo
limbcheck vlm-probe --image "$IMG" --view backs --vlm-base-url "$BASE" --vlm-model "$MODEL" --vlm-schema
