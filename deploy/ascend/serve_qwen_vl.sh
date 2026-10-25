#!/usr/bin/env bash
# 在昇腾上用 vLLM-Ascend 部署 Qwen-VL，提供 OpenAI 兼容接口。
# 网页 demo 和 limbcheck 把 VLM_BASE_URL 指向 http://<本机IP>:<PORT>/v1 即可使用。
#
# 用法示例：
#   IMAGE=quay.io/ascend/vllm-ascend:<与本机 CANN 匹配的版本> CARDS=0 MODEL=Qwen/Qwen3-VL-8B-Instruct PORT=8001 ./serve_qwen_vl.sh
#   IMAGE=... CARDS=1,2 MODEL=Qwen/Qwen3-VL-32B-Instruct PORT=8002 ./serve_qwen_vl.sh
#   IMAGE=... CARDS=4,5,6,7 MODEL=Qwen/Qwen2.5-VL-72B-Instruct PORT=8003 ./serve_qwen_vl.sh
set -euo pipefail

: "${IMAGE:?请设置 IMAGE，例如 quay.io/ascend/vllm-ascend:<版本>（按 vLLM-Ascend 官方文档选择与本机 CANN 匹配的版本）}"
MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}
CARDS=${CARDS:-0}                                   # 逗号分隔的 NPU 卡号
PORT=${PORT:-8001}
TP=${TP:-$(awk -F, '{print NF}' <<<"$CARDS")}       # 张量并行数，默认等于卡数
MAX_LEN=${MAX_LEN:-16384}
MAX_IMAGES=${MAX_IMAGES:-6}
CACHE=${CACHE:-$HOME/.cache/modelscope}             # 模型缓存目录（国内默认从 ModelScope 下载）
NAME=${NAME:-limbcheck-vl-$PORT}
EXTRA_ARGS=${EXTRA_ARGS:-}                          # 其他 vllm serve 参数，如 "--quantization ascend"（W8A8 权重）

devices=()
for c in ${CARDS//,/ }; do devices+=(--device "/dev/davinci$c"); done

mkdir -p "$CACHE"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --restart unless-stopped --shm-size 16g \
  "${devices[@]}" \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v "$CACHE":/root/.cache/modelscope \
  -e VLLM_USE_MODELSCOPE=True \
  -p "$PORT":8000 \
  "$IMAGE" \
  vllm serve "$MODEL" --host 0.0.0.0 --port 8000 \
    --served-model-name "$MODEL" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_LEN" \
    --limit-mm-per-prompt "{\"image\": $MAX_IMAGES}" \
    --trust-remote-code $EXTRA_ARGS

echo "已启动容器 $NAME。查看日志：docker logs -f $NAME"
echo "就绪后检查：curl http://localhost:$PORT/v1/models"
