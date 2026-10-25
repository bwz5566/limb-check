# 昇腾部署：Qwen-VL（L2 大模型复核）

用 [vLLM-Ascend](https://github.com/vllm-project/vllm-ascend) 在昇腾 NPU 上起 OpenAI 兼容接口，
网页 demo 和 `limbcheck` 只需把 `VLM_BASE_URL` 指过来，代码不用改。

> 本目录脚本**尚未在真实昇腾环境验证**（开发环境没有 NPU）。镜像版本、参数名以 vLLM-Ascend 官方文档为准，
> 第一次部署请先确认昇腾型号、单卡显存和 CANN 版本（交接文档第 15 章第 1 项）。

## 前提

- 已安装昇腾驱动和固件，`npu-smi info` 能看到卡。
- 已安装 Docker，能拉取 vLLM-Ascend 镜像（`quay.io/ascend/vllm-ascend`，版本与 CANN 匹配；内网可先导出导入）。
- 模型从 ModelScope 下载（脚本已设 `VLLM_USE_MODELSCOPE=True`）；离线环境先把权重放进 `CACHE` 目录。

## 启动

```bash
cd deploy/ascend
# M0 对比三档模型（按单卡 64 GB 估算；32 GB 的卡需要更多卡或量化权重）
IMAGE=quay.io/ascend/vllm-ascend:<版本> CARDS=0       MODEL=Qwen/Qwen3-VL-8B-Instruct   PORT=8001 ./serve_qwen_vl.sh
IMAGE=quay.io/ascend/vllm-ascend:<版本> CARDS=1,2     MODEL=Qwen/Qwen3-VL-32B-Instruct  PORT=8002 ./serve_qwen_vl.sh
IMAGE=quay.io/ascend/vllm-ascend:<版本> CARDS=4,5,6,7 MODEL=Qwen/Qwen2.5-VL-72B-Instruct PORT=8003 ./serve_qwen_vl.sh
```

可调参数（环境变量）：`TP`（张量并行，默认等于卡数）、`MAX_LEN`（上下文长度）、`MAX_IMAGES`（每次请求最多几张图，demo 每次 1 张）、
`CACHE`（模型缓存目录）、`EXTRA_ARGS`（其他 `vllm serve` 参数，如量化权重需要的参数）。
如果启动报 `--limit-mm-per-prompt` 格式错误，旧版本 vLLM 用 `image=6` 的写法，改脚本里那一行即可。

## 检查

```bash
curl http://localhost:8001/v1/models
./check.sh http://localhost:8001/v1 Qwen/Qwen3-VL-8B-Instruct 手背照片.jpg     # 跑一次完整 L2：提示词 → 模型 → 解析
```

`vlm-probe` 会打印模型原始输出、解析后的结构和每个检测项的结论。输出不合规（不是约定 JSON）时会列出原因。

## 接入

```bash
# 网页 demo
VLM_BASE_URL=http://<昇腾服务器>:8001/v1 VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct VLM_SCHEMA=1 python apps/web-demo/server.py
# 离线评测
limbcheck eval --data data/m0 --pipeline configs/pipelines/vlm-only.yaml --vlm-base-url http://<昇腾服务器>:8001/v1 --vlm-model Qwen/Qwen3-VL-8B-Instruct --vlm-schema
# 压测：并发 8，共 200 次请求，输出延迟分位数和 JSON 合规率
limbcheck bench-vlm --image 手背照片.jpg --view backs --concurrency 8 --requests 200 --vlm-base-url http://<昇腾服务器>:8001/v1 --vlm-model Qwen/Qwen3-VL-8B-Instruct
```

## 微调（M2 之后）

数据积累后可用 ms-swift 在昇腾上对 Qwen-VL 做 LoRA 微调（训练数据：截图 + 人工确认后的标准 JSON 答案）。
微调后的模型按同样方式部署，换 `MODEL` 即可，并先在回归集上对比（docs/VALIDATION.md 第 9 节）。
