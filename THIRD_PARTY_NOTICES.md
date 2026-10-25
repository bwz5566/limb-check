# 第三方组件与模型

| 组件 | 位置 | 来源 | 许可 |
| --- | --- | --- | --- |
| MediaPipe Tasks Vision（JS 与 wasm，0.10.14） | `apps/web-demo/vendor/` | npm `@mediapipe/tasks-vision` | Apache-2.0 |
| Hand Landmarker 模型 `hand_landmarker.task` | `models/` | Google MediaPipe 官方模型 | Apache-2.0（以官方模型卡为准） |
| Pose Landmarker 模型 `pose_landmarker_full.task` | `models/` | Google MediaPipe 官方模型 | Apache-2.0（以官方模型卡为准） |
| Selfie Multiclass 分割模型 `selfie_multiclass_256x256.tflite` | `models/` | Google MediaPipe 官方模型 | Apache-2.0（以官方模型卡为准） |

可选依赖（按需安装，不随仓库分发）：

| 组件 | 用途 | 许可 | 备注 |
| --- | --- | --- | --- |
| mediapipe（Python） | 离线评测的关键点与分割 | Apache-2.0 | |
| rtmlib / RTMW / RTMPose | 服务端关键点 | Apache-2.0 | 模型权重来自 OpenMMLab |
| SAM 2 / SAM 2.1 | 手指轮廓复核 | Apache-2.0 | |
| Qwen-VL 系列 | L2 大模型复核 | 以各模型发布页的许可为准 | 部分尺寸有单独许可条款，商用前核实 |
| YOLO11（若后续采用） | 外观项分割 | AGPL-3.0 / 企业许可 | 商用需购买企业许可，否则改用 MMSegmentation |

模型文件随仓库提供是为了在无法访问 storage.googleapis.com 的网络环境下也能直接运行。正式商用前请法务核实各模型卡的许可条款。
