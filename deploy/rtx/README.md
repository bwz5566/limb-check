# RTX 6000 部署：服务端 L1 模型（RTMW + SAM 2）与开发期大模型

> 本目录内容**尚未在 RTX 6000 上验证**：开发环境无 GPU，且无法访问模型下载站点。代码路径已写好，首次运行请按下面步骤核对。

## 1. 评测环境

```bash
# 系统依赖（MediaPipe 需要 EGL）
sudo apt-get install -y libegl1 libgles2 libgl1
# Python 3.10+，CUDA 12.x
pip install -e ".[mediapipe,rtmw-gpu,sam2,dev]"
pytest
```

或用本目录的 Dockerfile：

```bash
docker build -f deploy/rtx/Dockerfile -t limbcheck-eval .
docker run --gpus all -v $PWD/data:/app/data -v $PWD/runs:/app/runs limbcheck-eval \
  limbcheck eval --data data/m0 --pipeline configs/pipelines/rtmw-sam2-gpu.yaml
```

## 2. 模型文件

| 模型 | 获取 | 离线用法 |
| --- | --- | --- |
| RTMW / RTMPose-hand（rtmlib，ONNX） | 首次运行自动从 download.openmmlab.com 下载到 `~/.cache/rtmlib/hub` | 在能联网的机器上先跑一次，把缓存目录拷过来；或在流水线配置里写 `keypoints: {type: rtmw+hand, det: /models/xxx.onnx, pose: /models/xxx.onnx, hand_det: ..., hand_pose: ...}` |
| SAM 2.1 tiny | Hugging Face `facebook/sam2.1-hiera-tiny`，或 GitHub facebookresearch/sam2 提供的 checkpoint | `contour: {type: sam2, device: cuda, checkpoint: /models/sam2.1_hiera_tiny.pt}` |
| MediaPipe（hand / pose / selfie 分割） | 已随仓库放在 `models/` | — |

## 3. 跑对比

```bash
limbcheck eval --data data/m0 --pipeline configs/pipelines/mediapipe-skin.yaml
limbcheck eval --data data/m0 --pipeline configs/pipelines/rtmw-skin-cpu.yaml
limbcheck eval --data data/m0 --pipeline configs/pipelines/rtmw-sam2-gpu.yaml
limbcheck compare runs/<三次运行的目录> -o runs/l1-compare.md
limbcheck bench --data data/m0 --pipeline configs/pipelines/rtmw-sam2-gpu.yaml --repeat 3     # 吞吐与延迟
```

## 4. 开发期在 RTX 6000 上跑 Qwen-VL

昇腾环境就绪前，可以在一张 RTX 6000（48 GB）上用 vLLM 跑 8B 模型做联调：

```bash
pip install vllm
VLLM_USE_MODELSCOPE=True vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8001 --max-model-len 16384 \
  --limit-mm-per-prompt '{"image": 6}' --gpu-memory-utilization 0.85
```

然后 `VLM_BASE_URL=http://<RTX 服务器>:8001/v1`。正式方案 L2 放昇腾，RTX 6000 留给 L1 实时推理。
