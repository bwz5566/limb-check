# 详细说明：术语、判定逻辑与代码导读

这份文档把仓库里用到的**不常见的词**、**每一步判定的逻辑和公式**、**每个代码文件做什么**集中讲清楚，
让第一次接触项目的开发、算法、测试同事，以及需要理解“系统为什么这样判”的业务、法务同事都能看懂。

与其他文档的分工：

| 文档 | 回答的问题 |
| --- | --- |
| [README](../README.md) | 项目是什么、怎么跑起来 |
| [METRICS.md](METRICS.md) | 落地要看哪些指标、目标值、需要多少样本 |
| [VALIDATION.md](VALIDATION.md) | M0 验证怎么组织：人员、场景、步骤、出结果 |
| [EXTENDING.md](EXTENDING.md) | 需求变了改哪里：阈值、模型、提示词、检测项、岗位 |
| [HANDOFF.md](HANDOFF.md) | 项目背景、决策过程、模型选型、合规要求（第 16 章有一份简短术语表） |
| **本文** | 术语是什么意思、判定逻辑每一步怎么算、代码怎么组织 |

**怎么读**：只想知道结论怎么来的，看第 1、3、6 节；要调阈值或换模型，看第 4、5、7 节；要改代码，看第 8、10、11 节。
第 2 节术语表可以随时回查。

文中的数字都是 `configs/thresholds.yaml`（version 1）的默认值。阈值改了以后以配置文件为准，公式不变。

---

## 目录

1. [全景：一次检测从摄像头到报告](#1-全景一次检测从摄像头到报告)
2. [术语表](#2-术语表)
3. [基本约定](#3-基本约定)
4. [L1 判定原理（逐项）](#4-l1-判定原理逐项)
5. [L2 视觉大模型：提示词、解析与护栏](#5-l2-视觉大模型提示词解析与护栏)
6. [仲裁：L1 和 L2 怎么合成一个结论](#6-仲裁l1-和-l2-怎么合成一个结论)
7. [指标怎么算](#7-指标怎么算)
8. [网页 demo 的内部逻辑](#8-网页-demo-的内部逻辑)
9. [合成缺指样本](#9-合成缺指样本)
10. [代码逐模块导读](#10-代码逐模块导读)
11. [数据格式示例](#11-数据格式示例)
12. [常见疑问](#12-常见疑问)

---

## 1. 全景：一次检测从摄像头到报告

```
摄像头画面（未镜像的原始帧；页面上显示的镜像只是 CSS 效果）
  │ 每一帧：MediaPipe 关键点（手 21 点 / 人体 33 点），在浏览器里运行
  ▼
引导状态机（app.js）：按岗位配置的步骤逐个提示动作；动作到位并保持约 1 秒
  │ → 采集该“视角”的一张干净截图（关键点检测和截图用的是同一帧）
  │
  ├─► L1：浏览器内用关键点和像素做规则判定（手指长度比、美甲色差、四肢可见度与左右比）
  │
  ├─► L2：截图发给 server.py → 视觉大模型（固定提示词，要求只输出 JSON）→ 解析成统一结构
  │
  ▼
仲裁（server.py 调用 limbcheck.arbiter）
  │ 每个视角 × 每个检测项得到一个结论；同一检测项多个视角取最严重的
  ▼
报告：岗位要求逐条核对 + 证据截图 + 理由；验证模式下附上测试员填写的真实情况，可导出 JSON
  ▼
离线（limbcheck 命令行）
  import-demo 导入 → score 算浏览器端的指标 → eval 用其他模型重跑同一批截图 → compare 横向对比 → viz 画出漏检误报
```

五个最基本的概念：

| 概念 | 含义 | 例子 |
| --- | --- | --- |
| 检测项（check） | 要判断的一件事 | `fingers` 手指、`nails` 美甲、`limbs` 四肢 |
| 视角（view） | 一种拍摄姿势，对应一张截图 | `palms` 双手掌心、`backs` 双手手背、`stand` 正面站立 |
| 来源（source） | 谁给出的判断 | `L1:mediapipe`、`L1:contour-skin`、`L2:Qwen/Qwen3-VL-8B-Instruct` |
| 结论（decision） | 仲裁后的对外结论 | 满足、疑似问题·待人工确认、需人工复核、需重做、未评估 |
| 会话（session） | 一个人完整做一次检测 | 精密装配岗位一个会话有 6 张截图（stand、back、arms、arms_up、palms、backs） |

视角能看到哪些检测项定义在 `configs/checks.json`。默认配置：

| 视角 | 能评估的检测项 |
| --- | --- |
| `stand` 正面站立 | limbs、tattoo、scar、hair |
| `back` 背面站立 | tattoo、scar、hair |
| `arms` 双臂侧平举、`arms_up` 双臂上举 | limbs、tattoo、scar |
| `palms` 双手掌心 | fingers |
| `backs` 双手手背 | fingers、nails、tattoo、scar |
| `hand` 单手近景（离线数据用） | fingers、nails |

`liveness`（活体/防作弊）不对应任何视角，它来自随机手势挑战步骤的结果（第 8.5 节）。

---

## 2. 术语表

### 2.1 业务与判定

| 术语 | 解释 | 在哪里 |
| --- | --- | --- |
| M0 / M1 / 试点 | M0 是落地前的技术验证（本仓库）；M1 是正式产品第一版；试点是在真实招聘中小范围使用 | METRICS.md 第 2 节 |
| L1 | 第一层判定：专用小模型（关键点、分割）加规则，速度快、可解释，但会被“脑补”骗过 | `limbcheck/geometry.py`、`l1checks.py`、`app.js` |
| L2 | 第二层判定：视觉大模型（VLM）看截图给出结构化结论，作为 L1 的独立复核 | `limbcheck/prompts.py`、`vlm.py` |
| L3 / L4 | 交接文档里规划的语音引导助手、报告与复核助手，本仓库未实现 | HANDOFF.md 第 4 章 |
| 仲裁（arbitration） | 把 L1、L2 两路结论按固定规则合成一个对外结论 | `limbcheck/arbiter.py` |
| 岗位画像（job profile） | 一个岗位启用哪些检测项、每项是判定还是只记录、引导步骤、书面依据 | `configs/job_profiles/*.yaml` |
| 判定模式（mode） | `judge` 判定：结论对外有效；`record_only` 只记录：照常计算但对外显示“未评估”；`off`：岗位未启用，不计算 | `config.check_mode` |
| 书面依据（basis） | 外观项（纹身、疤痕、发色）要求的成文依据，如员工手册某条。没有依据时，配置加载会把这些项强制降为 `record_only` | `config.normalize_profile` |
| 正样本 / 负样本 | 真实情况“有问题”（缺指、有美甲）的样本是正样本，“没问题”的是负样本 | `dataset.truth_positive` |
| `positive` 三值 | `True` 发现问题，`False` 正常，`None` 无法判断（没检测到手、模型失败、输出看不懂） | `core.py` 文件头 |
| `pass` 满足 | 所有可用来源都认为正常且置信度够高，自动通过 | `core.PASS` |
| `fail_suggested` 疑似问题·待人工确认 | 有来源认为有问题。系统**从不输出最终“不通过”**，最多给出这个建议，由人确认 | `core.FAIL_SUGGESTED` |
| `review` 需人工复核 | 两路矛盾、置信度不足、重做后仍无法判断 | `core.REVIEW` |
| `redo` 需重做 | 画质或姿势不够、模型都无法判断，且还有重做机会 | `core.REDO` |
| `not_assessed` 未评估 | 只记录的项目，或岗位未启用 | `core.NOT_ASSESSED` |
| 影子结论（shadow） | 不管模式如何都按 `judge` 算出的结论。`record_only` 项对外是“未评估”，但影子结论照样记录，用来积累基线、评估效果 | `Arbitration.shadow` |
| 尝试次数（attempt）/ `max_redo` | 第几次尝试（从 0 起）。还能重做时才可能给 `redo`；用完重做机会后同样情况给 `review` | `arbiter.judge` |
| `shadow_first_attempt` | 按“第一次尝试”算的影子结论，用来统计首次尝试的重做率 | `pipeline.run`、`server.arbitrate_session` |
| 一致（agree） | L1、L2 都有结论时是否一致；都说有问题时还要求指向同一只手 | `arbiter.agreement` |
| 真值（truth）/ 标注 | 测试员记录的受检者真实情况，用于算检出率、误报率 | `dataset.py` 文件头 |
| 证据截图（evidence） | 每个视角的一张干净截图（发给 L2、用于离线重跑）和带关键点叠加的缩略图（报告里看） | `app.js captureView` |
| 随机挑战（challenge） | 随机要求伸出若干根手指，限时完成，用来拦截回放录像 | `app.js STEP_LIB.challenge` |
| 回放攻击 / 照片攻击 / 虚拟摄像头注入 | 对着摄像头播放录像 / 举照片 / 用 OBS 等软件把视频直接注入成摄像头 | VALIDATION.md 第 2 节 |
| 活体（liveness） | 证明镜头前是真人在实时配合 | `checks.json` 的 `liveness` |

### 2.2 关键点与几何

| 术语 | 解释 |
| --- | --- |
| 关键点（keypoint / landmark） | 模型在图上标出的身体部位坐标，如“右手腕在 (412, 530)” |
| 手部 21 点 | 0 腕；拇指 1–4；食指 5–8；中指 9–12；无名指 13–16；小指 17–20。MediaPipe Hands 与 COCO-WholeBody 手部顺序相同 |
| CMC / MCP / PIP / DIP / IP / TIP | 腕掌关节 / 掌指关节（指根）/ 近端指间关节 / 远端指间关节 / 拇指的指间关节 / 指尖。食指到小指每指 4 点依次是 MCP、PIP、DIP、TIP；拇指是 CMC、MCP、IP、TIP |
| 人体关键点 | MediaPipe Pose 33 点、COCO 17 点、COCO-WholeBody 133 点（含双手各 21 点）。代码里统一按关节名存放（`nose`、`left_shoulder`……），见 `keypoints.py` |
| 可见度（visibility） | Pose 模型给每个点的 0–1 分数，表示这个点是否真的在画面中可见。肢体缺失时模型常给出低可见度的“猜测点” |
| 掌长（palm_len） | 腕（0）到中指根（9）的距离，用作手部尺度，很多阈值按掌长的比例设定，与离镜头远近无关 |
| 链长（path_len） | 一根手指关键点依次相连的折线长度，如食指链长 = 5→6、6→7、7→8 三段距离之和 |
| 长度比（ratio） | 某指链长 ÷ 中指链长 |
| 伸直（extended） | 按关节夹角和指尖离腕距离判断手指是否伸直（第 4.1 节） |
| 对捏（pinch） | 拇指尖与另一指尖距离小于 0.32 × 掌长 |
| 左右手标签（handedness） | 模型给出的“这是左手还是右手”及其置信度 |
| 解剖学左右 | 受检者本人的左右。本项目所有“左右”都指这个，与画面左右无关 |
| 镜像（mirrored） | 自拍画面左右翻转。前置摄像头预览通常是镜像的；本项目的计算和截图一律用**未镜像**的原始画面 |
| 手性 / `orient_sign` | 用两个向量的叉积符号区分掌心还是手背朝镜头（第 4.4 节） |
| 叉积（cross product） | 二维向量 (x1, y1)、(x2, y2) 的叉积 = x1·y2 − y1·x2。符号表示第二个向量在第一个向量的顺时针还是逆时针方向 |
| 关键点“脑补” | 关键点模型按“完整的手”训练，手指缺了也会照样输出 21 个点，把缺失的指尖画在背景上。这是 L1 只能作初判的根本原因 |
| 躯干长（torso） | 左右肩到同侧髋距离的平均值，身体动作的尺度 |
| 肢段（segment） | 上臂（肩–肘）、前臂（肘–腕）、大腿（髋–膝）、小腿（膝–踝） |

### 2.3 图像与颜色

| 术语 | 解释 |
| --- | --- |
| RGB / sRGB | 屏幕和照片的颜色编码，每通道 0–255。sRGB 是非线性的（有 gamma），转 Lab 前要先线性化 |
| CIE Lab | 接近人眼感知的颜色空间：L 亮度（0–100），a 红绿轴（正为红），b 黄蓝轴（正为黄）。D65 是标准日光白点。两种颜色在 Lab 里的距离大致对应“看起来差多少” |
| ΔE（色差） | 两个 Lab 颜色的欧氏距离（CIE76 公式）。约 2 是人眼刚能察觉，10 以上是明显不同，20 以上差别很大 |
| 彩度（chroma） | √(a² + b²)，颜色有多“鲜艳”。黑、灰、白的彩度接近 0 |
| 色相（hue） | atan2(b, a) 的角度，颜色偏红、黄、绿、蓝 |
| 亮度（luminance） | 0.299R + 0.587G + 0.114B，0–255 |
| 拉普拉斯方差 | 对灰度图做拉普拉斯滤波（找边缘）后的方差。画面越清晰边缘越多，方差越大；模糊时方差小 |
| 掩码（mask） | 与图片同尺寸的 0/1 矩阵，1 表示“属于目标”（例如手部皮肤） |
| 分割（segmentation） | 给每个像素分类（背景、头发、身体皮肤、面部皮肤、衣物、其他）。本项目用 MediaPipe 的 `selfie_multiclass_256x256` |
| 形态学运算 | 膨胀（dilate）把掩码向外扩几个像素；腐蚀（erode）向内缩；闭运算（close）= 先膨胀再腐蚀，填小洞、接小缝；开运算（open）= 先腐蚀再膨胀，去掉零星噪点。“核”（kernel）是运算窗口的大小 |
| 连通域（connected component） | 掩码里彼此相连的一块区域。用来只保留“包含手掌”的那一块，去掉背景里颜色碰巧接近皮肤的块 |
| ROI | 感兴趣区域（Region of Interest），只在这块矩形里计算，省时间也减少干扰 |
| 覆盖率（tip coverage） | 指尖段的采样点有多少落在手部掩码内（第 4.2 节） |
| 图像修复（inpainting） | 用周围像素填补被抹掉的区域。本项目用 OpenCV 的 Telea 算法（从边界向内逐步推算） |
| 羽化 | 边缘用高斯模糊做渐变过渡，避免生硬的拼接痕迹 |
| data URL / base64 | 把图片编码成 `data:image/jpeg;base64,...` 文本，可直接放进 JSON 发给大模型 |

### 2.4 模型与部署

| 术语 | 解释 |
| --- | --- |
| MediaPipe Tasks | Google 的端侧视觉库。`HandLandmarker`（手 21 点）、`PoseLandmarker`（人体 33 点）、`ImageSegmenter`（分割）。模型文件是 `.task` / `.tflite`，网页里通过 WebAssembly（wasm）运行，可用 GPU 加速（delegate） |
| RTMW / RTMPose / rtmlib | OpenMMLab 的关键点模型：RTMW 输出全身 133 点，RTMPose-hand 是手部近景模型。rtmlib 用 ONNX Runtime 运行它们，不需要装 mmpose |
| ONNX Runtime | 通用的模型推理引擎，CPU、CUDA 都能跑 |
| SAM 2 / SAM 2.1 | Meta 的“分割一切”模型：给几个提示点，它分割出点所在的物体。`hiera-tiny` 是最小的版本。本项目用它做手部掩码 |
| 提示点（point prompt） | 告诉 SAM “这些点属于目标”。本项目给手腕、食指到小指的指根和掌心中点 |
| VLM（视觉大模型） | 能看图回答问题的大模型，本项目用 Qwen-VL 系列（Qwen3-VL-8B / 32B、Qwen2.5-VL-72B，或百炼的 qwen3.7-plus / qwen3.8-max） |
| OpenAI 兼容接口 | 请求格式与 OpenAI 的 `/v1/chat/completions` 相同。vLLM、vLLM-Ascend、百炼、Ollama 都支持，所以切换模型只需改地址和模型名 |
| `temperature` | 采样随机性。本项目设为 0，同一张图尽量给出同样的答案 |
| JSON Schema / 结构化输出 | 在请求里附一份 JSON 格式说明（`response_format`），服务端在生成时强制输出合法 JSON。vLLM 支持；不支持的服务会返回 400，客户端自动退回普通模式 |
| 提示词版本 | 提示词放在 `configs/prompts/v1.json`，改提示词要复制成 v2 再切换，便于回滚和回归对比 |
| 护栏（guardrail） | 防止模型输出越界的检查：只保留要求的检测项、标签必须在白名单内等（第 5.4 节） |
| vLLM / vLLM-Ascend | 高性能大模型推理服务；vLLM-Ascend 是适配华为昇腾 NPU 的版本 |
| 昇腾 NPU / CANN | 华为的 AI 加速卡及其驱动和算子库。CANN 版本决定能用哪个 vLLM-Ascend 镜像 |
| TP（张量并行） | 一个模型拆到多张卡上同时算，`--tensor-parallel-size`。32B 模型通常用 2 卡（TP=2） |
| `--max-model-len` / `--limit-mm-per-prompt` | 单次请求的最大上下文长度 / 单次请求最多几张图（默认 16384 / 6） |
| ModelScope | 国内的模型下载站，`VLLM_USE_MODELSCOPE=True` 时从这里下载权重 |
| DashScope（阿里云百炼） | 阿里云的大模型 API 服务，有 OpenAI 兼容模式。截图会发到外部，需在授权书里写明 |
| LoRA / ms-swift | 只训练少量附加参数的微调方法 / 支持在昇腾上做微调的训练框架 |
| PCK、AP | 关键点模型的公开评测指标，解释见 HANDOFF.md 5.3 节 |

### 2.5 统计与指标

| 术语 | 解释 |
| --- | --- |
| 检出率（recall，召回率、灵敏度） | 正样本里被发现的比例 = 发现的正样本 ÷ 全部正样本 |
| 漏检率（miss rate） | 正样本里被自动“满足”放过的比例 |
| 误报率（false alarm rate / FPR） | 负样本里被错误标为有问题（或送人工）的比例 |
| 特异度（specificity） | 负样本里被正确判为正常的比例 = 1 − 误报率（模型单独统计时） |
| 精确率（precision） | 判为有问题的样本里真的有问题的比例 |
| 覆盖率（coverage） | 有标注的样本里，模型给出了明确结论（不是“无法判断”）的比例 |
| 一致率（accuracy） | 系统结论与真实情况一致的比例 |
| 自动通过率 / 需人工比例 | 结论为“满足”的比例 / 结论为“疑似问题”或“需人工复核”的比例 |
| 首次重做率 | 按第一次尝试算，会被要求重做的比例 |
| 置信区间（CI） | 用有限样本估计比例时的不确定范围。95% 置信区间的意思是：用这种方法算出的区间，约 95% 的情况下包含真实值 |
| Wilson 区间 | 一种比例的置信区间算法，样本少或比例接近 0、100% 时比简单的“±1.96 × 标准误”可靠（第 7.7 节） |
| 三法则（rule of three） | 0 次事件出现在 n 次试验中，真实概率的 95% 上限约为 3/n。例如 0/100 误报，真实误报率上限约 3% |
| Cohen's κ（kappa） | 扣除“碰巧一致”后的一致程度：1 完全一致，0 与随机一样，0.6 以上一般认为较好（第 7.4 节） |
| 点估计 / `ci_low` / `ci_high` | 直接算出的比例 / 置信区间下限 / 上限。验收标准可以要求“区间下限 ≥ 目标”，比点估计更严格 |
| `min_n` | 样本数下限。达不到时验收结果是“样本不足”，既不算通过也不算未达标 |
| P50 / P95 | 中位数 / 95% 的请求都不超过的耗时 |
| 吞吐（throughput）/ 并发（concurrency） | 每秒处理多少张 / 同时有多少个请求在处理 |
| 公平性差距（gap） | 不同分组（如肤色、光线、设备）之间检出率或误报率的最大差。只统计样本数 ≥ 10 的分组 |
| 校准集 / 测试集 | 调阈值只用校准集，最终报告用没参与调参的测试集，并按受检者划分，防止“过拟合”（调出来的阈值只对这批人有效） |
| 回归集 | 固定不变的一批评测数据，每次改模型、阈值、提示词前后各跑一次，确认没有变差 |

### 2.6 工程

| 术语 | 解释 |
| --- | --- |
| 节点（node）/ 注册表（registry） | 流水线里每个可替换的部件叫节点，分五类：`keypoints`、`contour`、`segmenter`、`l1`、`arbiter`。注册表按“类型 + 名称”找到对应的实现 |
| 插件（plugin） | 自己写的模块，用 `@register` 把新节点登记进注册表，在流水线配置的 `plugins:` 里列出即可，不改主干代码 |
| spec | 配置里对节点的写法：名称（`mediapipe`）、完整路径（`mypkg.models:MyModel`）或带参数的字典（`{type: sam2, device: cuda}`） |
| 流水线（pipeline） | 一套节点组合（关键点 + 轮廓复核 + 分割 + L1 判定列表 + 大模型），定义在 `configs/pipelines/*.yaml` |
| 适配器（adapter） | 把不同模型的输出转换成统一格式（`Hand`、`Body`）的包装类 |
| Protocol | Python 的“接口约定”：只要实现了约定的方法就能用，不需要继承 |
| 单一来源 | 检测项、视角、阈值、岗位只在 `configs/` 里定义一次；网页 demo 读的 `config.generated.json` 由它生成，测试会检查两者是否同步 |
| JSONL | 每行一个 JSON 对象的文本文件，便于追加和逐行读取 |
| `pip install -e .` | “可编辑安装”：包指向仓库里的源代码，改代码立即生效，配置文件也按仓库目录查找 |
| CI | 持续集成。每次推送到 GitHub 自动安装依赖、检查网页配置同步、跑全部单元测试（`.github/workflows/ci.yml`） |

---

## 3. 基本约定

### 3.1 为什么 `positive` 要有第三个值 `None`

“没检测到手”和“手是正常的”是两回事。如果把无法判断当成正常，摄像头没对准的人就会被自动放行；当成有问题，又会冤枉人。
所以每个判定都区分三种结果，`None` 在仲裁里走“重做或人工复核”的路径（第 6 节），在指标里按最不利的情况计（第 7.2 节）。

### 3.2 左右与镜像

- 所有“左右”都是**受检者本人**的左右（解剖学左右）。
- 网页上显示的视频用 CSS（`transform: scaleX(-1)`）做了镜像，方便受检者对照自己；**截图、关键点计算、发给大模型的图都是未镜像的原始画面**。
- 未镜像画面里，受检者面对镜头时，**他的右手出现在画面左侧**。大模型的系统提示词里明确写了这一点。
- 离线数据如果是自拍镜像图，在流水线配置里设 `mirrored: true`：模型给出的左右手标签、人体关节的左右名称都换回受检者本人的左右，
  按位置分左右手和判断掌心/手背时也会相应取反（`keypoints.mirror_body`、`assign_sides`、`facing`）。

### 3.3 坐标

像素坐标，原点在左上角，x 向右、y **向下**。MediaPipe 输出 0–1 的归一化坐标，代码里乘以图宽高转成像素。
y 向下这一点会影响叉积的符号（第 4.4 节）。

### 3.4 部位（part）的写法

| 写法 | 含义 | 谁会用 |
| --- | --- | --- |
| `right_hand` / `left_hand` | 某只手 | 手指、美甲的 L1 和 L2 |
| `right_hand.index` | 某只手的某根手指（thumb / index / middle / ring / little） | 手指 L1；L2 返回了 `fingers` 列表时 |
| `left_arm`、`right_leg` | 整条肢体 | 四肢 L1（可见度低） |
| `left_forearm`、`right_thigh` 等 | 某侧肢段 | 四肢 L1（左右长度比异常） |
| `head` | 头部 | 发色 |
| `all_visible` | 所有可见部位 | 纹身、疤痕 L2 “没有发现”时 |

仲裁比较两路是否指向同一处时，手指只比较到“哪只手”：L1 说 `right_hand.index`，L2 说 `right_hand`，算一致。

### 3.5 来源（source）的命名

`L1:` 或 `L2:` 前缀加模型名。几路 L1 合并时用 `+` 连接，例如 `L1:mediapipe+contour-skin`。
网页端的关键点结果是 `L1:mediapipe-web`，美甲色差是 `L1:deltaE`，发色是 `L1:mpseg-hair`，大模型是 `L2:<模型名>`。

### 3.6 置信度不是概率

每个判定都附一个 0–1 的置信度（confidence），但它是**按规则算出的分数**（离阈值越远越高），不是经过校准的概率。
它唯一的用途是和仲裁参数 `high_conf`（0.8）、`low_conf`（0.5）比较。置信度的计算方式本身也属于 M0 校准的内容。

---

## 4. L1 判定原理（逐项）

L1 的判定函数都在 `limbcheck/geometry.py`、`limbcheck/adapters/contour.py`、`limbcheck/appearance.py`，
通过 `limbcheck/l1checks.py` 按检测项注册。网页端 `app.js` 里有同一套公式的 JavaScript 版本（手指、美甲、四肢），两边结果一致。

| 注册名 | 检测项 | 需要的模型节点 | 网页端是否有 |
| --- | --- | --- | --- |
| `fingers.keypoints` | 手指 | keypoints | 有 |
| `fingers.contour` | 手指 | keypoints + contour | 无（只在离线评测） |
| `nails.delta_e` | 美甲 | keypoints | 有 |
| `limbs.pose` | 四肢 | keypoints | 有 |
| `hair.color` | 发色 | segmenter | 无 |

纹身、疤痕目前没有 L1，只靠 L2。

### 4.1 手指：关键点长度比（`fingers.keypoints`）

**思路**：正常人各手指长度的比例大致固定（以中指为 1，食指约 0.88、小指约 0.72）。某根手指缺失或短缩时，它的链长相对中指会明显偏短。
用比例而不用绝对长度，是为了不受离镜头远近、手的大小影响。

**公式**（`geometry.analyze_fingers`）：

```
ratio(指) = 链长(指) ÷ 链长(中指)
limit(指) = ref(指) × short_factor
ratio < limit → 记“疑似缺失或短缩”（missing_or_short）
```

| 参数（`fingers` 段） | 默认值 | 含义 |
| --- | --- | --- |
| `ref` | 拇指 0.95、食指 0.88、中指 1.0、无名指 0.93、小指 0.72 | 正常比例的参考值 |
| `short_factor` | 0.6 | 低于参考值的 60% 才报。例如食指的界限是 0.88 × 0.6 = 0.528 |
| `middle_to_palm_min` | 0.45 | 中指自己是参照物，不能和自己比，所以另外检查“中指链长 ÷ 掌长”，低于 0.45 记中指疑似短缩 |

**置信度**：

- 报问题的手指：`min(1, 1.5 − ratio / limit)`。比例越低越确定。
- 判正常的手：`min(1, 0.5 + margin) × 手的检测置信度`，其中 `margin = min(ratio / limit − 1)`，即最接近界限的那根手指离界限有多远。

**算例**：

- 右手食指比例 0.30，界限 0.528 → 报问题，置信度 = 1.5 − 0.30 / 0.528 = **0.932**。
- 小指比例 0.40，界限 0.72 × 0.6 = 0.432 → 报问题，但置信度只有 1.5 − 0.926 = **0.574**（刚过界限，把握不大）。
- 一只正常的手，五指比例为 0.92、0.86、1.0、0.95、0.70，最接近界限的是拇指（0.92 / 0.57 − 1 = 0.614），
  检测置信度 0.97 → 置信度 = min(1, 1.114) × 0.97 = **0.97**。

**伸直判断**（`finger_extended`，用于动作引导和随机挑战数手指）：

- 食指到小指：∠(MCP, PIP, TIP) > 145°，且指尖离腕的距离 > PIP 离腕距离的 1.1 倍。
- 拇指：∠(CMC, MCP, TIP) > 140°，且拇指尖离食指根（5 号点）> 0.45 × 掌长。

**整体结论**（`fingers_result`）：任一只手有手指报问题 → 有问题；没报问题但检测到的手比该视角应有的少（如双手视角只看到一只）
→ 无法判断（`None`，`quality_ok = False`）；否则正常，置信度取各手中最低的。

**局限**：手指弯曲时投影会变短，所以要求五指张开再采集；关键点模型会“脑补”出缺失的手指，此时长度比是正常的，
这种情况长度比法发现不了，需要下面的轮廓复核和 L2。

### 4.2 手指：轮廓复核（`fingers.contour`）

**思路**：关键点会被“脑补”，但皮肤像素不会。另外做一个只依赖像素的**手部掩码**，检查每根手指最末一节的关键点是否真的落在皮肤上。
如果指尖段大部分落在掩码外，说明那里其实是背景，关键点很可能是编出来的。

**采样**（`contour.tip_coverage`）：

```
最末一节：食指到小指是 DIP → TIP，拇指是 IP → TIP
在这一节上 t = 0.2、0.45、0.7 处各取一个采样点（避开指尖边缘）
每个采样点周围 (2r+1)×(2r+1) 的小窗口里只要有一个像素在掩码内就算“命中”，r = max(2, 0.03 × 掌长)
覆盖率 = 命中数 ÷ 3，只可能是 0、1/3、2/3、1
覆盖率 < tip_on_mask_min（0.5）→ 该指记“关键点脑补”（tip_off_mask）
```

也就是说三个点里最多只命中一个时报问题。置信度：报问题时 `0.5 + 0.5 × (1 − 覆盖率)`（覆盖率 1/3 时为 0.833，为 0 时为 1.0）；
全部正常时 `0.5 + 0.5 × 最低覆盖率`。

**三种掩码后端**（流水线配置的 `contour:`）：

| 后端 | 原理 | 运行环境 | 弱点 |
| --- | --- | --- | --- |
| `skin` | 以手掌中央的颜色为参照，找颜色相近的像素 | 纯 OpenCV，CPU | 背景接近肤色（木纹、米色墙）时会把背景也算成手 |
| `mpseg` | MediaPipe 多类别分割的“身体皮肤”类 | CPU | 模型按自拍场景训练，手很小时边缘粗糙 |
| `sam2` | SAM 2.1，用手腕、食指到小指的指根（5、9、13、17）和掌心中点作提示点，取得分最高的分割结果 | GPU（RTX 6000） | 需要 torch 和权重；尚未实测 |

**`skin` 后端的步骤**（`SkinMask.hand_mask`）：

1. **取参照色**：关键点 0、1、5、9、13、17 围成的手掌多边形，向中心收缩到 60%（避开边缘），取其中像素 Lab 颜色的中位数。
   像素少于 20 个时退回取掌心中点周围 7×7 的小块。
2. **划 ROI**：所有关键点的外接矩形，四周各扩 0.5 × 掌长。
3. **按色差分类**：ROI 内每个像素与参照色的距离 `d = √((0.5·ΔL)² + Δa² + Δb²)`，`d < skin_delta_e`（20）算皮肤。
   亮度差 ΔL 只算一半权重，是因为同一只手上的阴影、高光主要改变亮度而不太改变颜色。
4. **形态学清理**：闭运算（核大小 `max(3, 0.04 × 掌长)`，取奇数）补上指缝里的小洞，再用 3×3 开运算去掉零星噪点。
5. **只保留手**：做连通域分析，只保留包含手掌关键点（0、1、5、9、13、17）的那几块，去掉背景里颜色碰巧接近的块。
6. **膨胀**：向外扩 `mask_dilate_px`（3）像素，容忍关键点稍微落在皮肤边缘外。

**局限**：指尖涂了深色甲油、指尖处有强阴影时，指尖段可能落在掩码外，造成误报；背景接近肤色时掩码会过大，造成漏检。
这正是需要比较三种后端、并且一定要有 L2 的原因。

### 4.3 多路 L1 合并（`core.merge_any`）

同一检测项有多个 L1 判定（如手指的长度比和轮廓复核）时，按“任一路报问题即报问题”合并：

| 各路结果 | 合并后 | 置信度 |
| --- | --- | --- |
| 至少一路有问题 | 有问题，部位取有问题各路的并集 | 有问题各路中最高的 |
| 全部正常 | 正常 | 各路中最低的 |
| 其他（有一路无法判断且没有一路报问题） | 无法判断 | 0 |

`quality_ok` 要求每一路都为真。用真实图片跑的例子见第 11.2 节：长度比 0.943、轮廓复核 1.0，合并后正常、置信度 0.943。

### 4.4 掌心还是手背（`geometry.facing`）

**为什么需要**：掌心、手背两个步骤要确认受检者真的翻了手；美甲只能在手背朝前时看。

**原理**：取两个向量，腕→食指根（0→5）和腕→小指根（0→17），算叉积的符号 `orient_sign`：

```
orient_sign = sign( (x5 − x0)·(y17 − y0) − (y5 − y0)·(x17 − x0) )
```

手在画面里怎么转（旋转）都不改变这个符号；只有“翻面”（掌心换手背）、换另一只手、或画面镜像才会让符号反过来。
所以知道了是哪只手，就能从符号判断朝向：

| 手 | 朝向镜头 | 未镜像画面的 `orient_sign` | 镜像画面 |
| --- | --- | --- | --- |
| 右手 | 手背 | +1 | −1 |
| 右手 | 掌心 | −1 | +1 |
| 左手 | 手背 | −1 | +1 |
| 左手 | 掌心 | +1 | −1 |

直观验证：受检者举起右手、掌心朝镜头、手指向上时，拇指一侧（食指根 5）在画面右侧，小指根 17 在画面左侧。
画面坐标 y 向下，0→5 指向右上、0→17 指向左上，叉积为负，对应“右手掌心 = −1”。

代码里统一写成 `s = orient_sign × (右手 ? 1 : −1) × (镜像 ? −1 : 1)`，`s > 0` 为手背，`s < 0` 为掌心。
这组符号已用 MediaPipe 官方测试图（`right_hands.jpg`、`left_hands.jpg`）核对，见 `tests/test_mediapipe_smoke.py`。

### 4.5 左右手（`keypoints.assign_sides`）

1. 双手视角检测到两只手，模型给出的左右标签**互不相同**且置信度**都 ≥ 0.8** → 采用模型的标签。
2. 否则按手腕在画面中的 x 坐标：未镜像画面里**靠左的是右手**（镜像画面反过来）。
3. 单手视角（`hand`）且数据里标了 `side` 时，直接用标注。

实测 MediaPipe Tasks 版的 `HandLandmarker` 对**未镜像**照片给出的就是本人的左右（与一些旧文档“默认输入是自拍镜像”的说法不同），
所以只有输入本身是镜像画面时才取反。双手交叉时第 2 条会认错，报告里会注明每只手的左右是按模型（`model`）还是按位置（`x-order`）判定的。

### 4.6 美甲：指甲与肤色的色差（`nails.delta_e`）

**只对手背朝前的手判断**，没有手背朝前的手时返回 `None`（不参与仲裁）。

**取色位置**（`geometry.nails_delta_e`）：

- 肤色：关键点 0、5、17 的重心（手背中央），半径 `max(4, 0.08 × 掌长)` 的方块内取平均色。
- 指甲：从指尖向 DIP 方向走 30% 的位置，半径 `max(3, 0.18 × 最末一节长度)` 的方块内取平均色。

**判定**：两种颜色都转成 Lab，算 ΔE。任一根手指 ΔE > `delta_e`（22）即判这只手“疑似美甲”。

**置信度**：设最大色差为 w、阈值为 22。疑似美甲时 `min(1, 0.55 + (w − 22) / 22)`；正常时 `min(1, 0.55 + (22 − w) / 22)`。

**算例**：

| 颜色 | RGB | Lab | 与肤色 ΔE | 结论 |
| --- | --- | --- | --- | --- |
| 手背肤色 | (224, 172, 140) | (74.3, 14.8, 23.8) | — | 参照 |
| 红色甲油 | (200, 40, 60) | (44.4, 61.6, 29.5) | 55.9 | 疑似美甲，置信度 1.0 |
| 自然指甲 | (235, 190, 175) | (80.5, 13.9, 13.7) | 11.9 | 正常，置信度 1.0 |

真实照片（MediaPipe 测试图）上，自然指甲的最大 ΔE 是 20.4，离阈值 22 很近，正常的置信度只有 0.623，低于 0.8，
仲裁结果是“需人工复核”（第 11.2 节）。这说明色差法对自然指甲误报偏多，透明、裸色甲油基本识别不了，是 M0 校准时要优先处理的项目。

### 4.7 四肢：可见度与左右比（`limbs.pose`）

**思路**：肢体缺失时，Pose 模型仍会在画面里给出那几个关节的“猜测点”，但可见度很低；肢体明显短缩时，左右同一肢段长度差别很大。

**步骤**（`geometry.limbs_result`）：

1. 没检测到人 → 无法判断。
2. 左右肩、左右髋这 4 个躯干点有任一可见度 < `min_visibility`（0.55）→ 无法判断（没站进画面或背对镜头）。
3. 关节落在画面边缘 1% 以外 → 视为“没拍全”，记为未完整入框（无法判断，应重拍），**不算疑似**。
4. 每条肢体（臂：肩、肘、腕；腿：髋、膝、踝）三个关节的最低可见度 v < 0.55 → 疑似缺失或被遮挡（`low_visibility`），置信度 `0.6 × (1 − v)`。
5. 每个肢段的左右长度比（左 ÷ 右）超出 `limb_symmetry` [0.75, 1.33] → 左右长度差异大（`asymmetric`），置信度 0.6；比值 < 1 记在左侧，> 1 记在右侧。
6. 都没问题时正常，置信度 `min(1, 最低可见度 ÷ 0.55 × 0.8)`。

**哪些视角评估什么**：上肢在 `stand`、`arms`、`arms_up` 都评估；下肢只在 `stand` 评估（其他视角腿部不一定在画面里）。
岗位画像的 `params: {upper, lower}` 可以关掉上肢或下肢，例如精密装配岗位只评估上肢。

**算例**：左前臂缺失，模型给出的左腕可见度 0.2 → 左臂报问题，置信度 0.6 × 0.8 = 0.48。置信度低没关系，任何一路报问题都会进入人工确认（第 6 节）。

### 4.8 发色（`hair.color`，粗略基线）

1. 用 `selfie_multiclass` 分割出头发像素，少于 `min_pixels`（400）→ 无法判断。
2. 取头发像素 Lab 的中位数，算彩度 `chroma = √(a² + b²)`。
3. `chroma ≥ dyed_chroma_min`（18）→ 疑似染色。
4. 置信度 `min(0.9, 0.5 + |chroma − 18| / 36)`，最高 0.9。例如彩度 25 → 0.694；彩度 5 → 0.861。

黑、棕、灰白头发彩度低，染成红、黄、蓝等颜色彩度高。这是很粗的启发式（挑染、深棕染发都分不清），只作基线记录；
发色在没有书面依据前是 `record_only`。

### 4.9 画质门控（`geometry.quality`）

| 检查 | 怎么算 | 默认范围 | 是否影响判定 |
| --- | --- | --- | --- |
| 亮度 | **只看受检部位**：手部关键点或身体关节周围的小块像素（半径为画面短边的 1%）的平均亮度 | 60–215 | 是：过暗或过亮时把本视角所有 L1 结果标为 `quality_ok = False` |
| 模糊 | 整图拉普拉斯方差 | ≥ 80 | 否：阈值尚未实测校准，只记录 |

只看受检部位，是为了避免“白墙背景很亮，但手本身光线正常”被误判为过亮。L2 的画质来自大模型自己的 `image_quality`，为 `poor` 时 `quality_ok = False`。

---

## 5. L2 视觉大模型：提示词、解析与护栏

### 5.1 哪些检测项交给大模型

`prompts.l2_checks_for(view, enabled)` = 该视角能看到的检测项 ∩ 岗位启用的检测项 ∩ 提示词文件里有规则的检测项。
例如精密装配岗位：`palms` 只问手指；`backs` 问手指、美甲、纹身、疤痕；`stand` 问四肢、纹身、疤痕、发色。
一个视角只发一次请求，一次问多个检测项。

### 5.2 请求长什么样

OpenAI 兼容格式，一条 system 消息加一条 user 消息（文字 + 图片）：

```json
{
  "model": "Qwen/Qwen3-VL-8B-Instruct",
  "temperature": 0,
  "max_tokens": 1024,
  "messages": [
    {"role": "system", "content": "你是工厂招聘远程检测的辅助复核员。只根据图片判断“本次要求评估的项目”……只输出一个 JSON 对象"},
    {"role": "user", "content": [
      {"type": "text", "text": "图片：受检者双手掌心朝向镜头、五指张开的近景。\n本次只评估：手指（fingers）……"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    ]}
  ]
}
```

system 提示词的要点：只判断要求的项目，**不描述、不推断性别、年龄、民族、肤色、体型、宗教**；说明画面未镜像（画面左侧是受检者右侧）；
看不清就把 `uncertain` 设为 true，不要猜。

user 提示词由 `build_user_prompt` 按三部分拼装：视角描述、“本次只评估：……不要输出其他项目”、按检测项编号的规则加输出示例。
`palms` 视角只问手指时，实际生成的文字是：

```
图片：受检者双手掌心朝向镜头、五指张开的近景。
本次只评估：手指（fingers）。不要输出其他项目。
按以下规则输出 items，每条都要有 check、part、label、confidence（0 到 1），可选 fingers、value、uncertain：
1. fingers：对画面中每只手各输出 1 条，part 为 "right_hand" 或 "left_hand"；label 为 "normal"（五指完整）、
   "missing_or_short"（有手指缺失或明显短缩）或 "not_visible"（这只手看不清或不在画面中）；fingers 列出缺失或短缩的手指
   （thumb/index/middle/ring/little），没有则为 []；value 为这只手可见的手指数量。图上如有关键点连线只作参考，
   它可能把缺失的手指“补”出来，请以真实的皮肤轮廓为准。
image_quality 为 good、fair 或 poor；reason 不超过 50 字。
输出格式示例：{"items": [{"check": "fingers", "part": "…", "label": "…", "fingers": [], "value": null, "confidence": 0.9, "uncertain": false}], ...}
```

（规则一行很长，这里为了排版折行显示。）各检测项的规则和标签在 `configs/prompts/v1.json`：

| 检测项 | “有问题”标签 | “正常”标签 | 是否逐手 |
| --- | --- | --- | --- |
| fingers | missing_or_short | normal | 是 |
| nails | polish、tips、extension | none | 是 |
| tattoo | tattoo | none | 否 |
| scar | scar | none | 否 |
| hair | dyed、highlights | natural | 否 |
| limbs | missing_or_abnormal | normal | 否 |

另外 `not_visible`、`uncertain`、`unknown` 三个标签任何检测项都可以用，表示无法判断。

### 5.3 JSON Schema 模式

设 `VLM_SCHEMA=1`（或命令行 `--vlm-schema`）时，请求里会带上 `response_format: {type: json_schema, ...}`，
由服务端在生成时强制输出符合格式的 JSON（`prompts.schema`）。服务端不支持时会返回 400，
客户端记录一条“服务端不支持 JSON Schema，已退回普通模式”并重发普通请求（`vlm.VLMClient.review`）。

### 5.4 解析与护栏（`prompts.parse_reply`）

大模型的输出不一定规整，解析分两步：

1. **抽取 JSON**（`_extract_json`）：去掉 Markdown 代码块标记，从第一个 `{` 开始按括号配对（会跳过字符串里的括号）找到完整的对象。
2. **逐条清洗**，每一条问题都记进 `errors`，最终写进报告：

| 情况 | 处理 | 目的 |
| --- | --- | --- |
| 输出了本次没要求的检测项（例如自己加了 `gender`） | 丢弃这一条 | 隐私护栏：模型不能借机评价无关属性 |
| `label` 不在该检测项的白名单内 | 当作 `uncertain` | 不让模型自创结论 |
| `confidence` 超出 0–1 或不是数字 | 截断到 0–1；不是数字按 0 | 防止异常值影响仲裁 |
| `fingers` 里有不认识的名字 | 去掉 | — |
| `image_quality` 不是 good / fair / poor | 按 fair | — |
| `reason` 过长 | 截到 200 字 | — |
| 完全找不到 JSON，或没有 `items` 数组 | 整个输出作废，所有检测项为“无法判断” | — |

### 5.5 转换成统一的判定结果（`prompts.to_check_results`）

每个检测项按下面的顺序决定：

1. 有任一条是“有问题”标签 → **有问题**，置信度取这些条中最高的；其中 `uncertain: true` 的条目置信度**减半但仍算有问题**（宁可多送人工，也不放过）。
2. 一条都没返回 → 无法判断，备注“模型未返回该项”。
3. 有任一条是未知标签，或“正常”标签但 `uncertain: true` → 无法判断，备注“模型表示无法确定”。
4. 全部是“正常”且确定 → **正常**，置信度取最低的。逐手的检测项还要求返回的手数够（双手视角两只都要有），否则无法判断，备注“只返回了 1 只手”。

**完整算例**：模型返回了下面这段（带了代码块标记，还多输出了一个没要求的 `gender`）：

```json
{"items": [
  {"check": "fingers", "part": "right_hand", "label": "missing_or_short", "fingers": ["index"], "value": 4, "confidence": 0.86, "uncertain": false},
  {"check": "fingers", "part": "left_hand", "label": "normal", "fingers": [], "value": 5, "confidence": 0.93, "uncertain": false},
  {"check": "gender", "part": "all", "label": "male", "confidence": 0.9}],
 "image_quality": "good", "reason": "右手食指仅见一节"}
```

解析结果：`gender` 被丢弃并记录“输出了未要求的项目 'gender'，已丢弃”；手指判为**有问题**，置信度 0.86，部位 `right_hand.index`，
`quality_ok = True`，备注“右手食指仅见一节”。

### 5.6 失败时怎么办

网络不通、超时（默认 90 秒）、返回格式不对时 `review()` 不抛异常，把错误记进 `errors`，所有检测项为“无法判断”，
仲裁按“这一路无法判断”处理（第 6 节）。网页 demo 里没配置大模型时 `/api/review` 返回 503，页面跳过 L2，只用 L1 仲裁。

---

## 6. 仲裁：L1 和 L2 怎么合成一个结论

### 6.1 原则

- 任何“有问题”的结论只给 `fail_suggested`（疑似问题·待人工确认），**系统从不输出最终“不通过”**。
- 只有所有可用来源都认为正常、置信度都 ≥ `high_conf`（0.8）、画质合格时才自动“满足”；
  两路都认为正常时，一路 ≥ `high_conf`、另一路 ≥ `agree_min_conf`（0.7）也自动“满足”。
- 两路矛盾或拿不准时送人工；画质不够或都无法判断时，还有重做机会就先要求重做。

### 6.2 判定顺序（`arbiter.judge`，即默认策略 `default`）

输入是 L1、L2 两路结果（某一路没有时为空，例如纹身没有 L1），以及参数 `high_conf`、`low_conf`、`max_redo` 和当前尝试次数 `attempt`。
`还能重做` = `attempt < max_redo`。

```
1. 两路都没有                                  → review   “没有任何模型结果”
2. 有一路画质不合格，且还能重做                   → redo     “画质或姿势不足以判断”
3. 所有来源都无法判断                           → 能重做 redo，否则 review
4. 有一路认为有问题：
   4a. 两路都有结论且一正一反：
       能重做且较低的置信度 < low_conf           → redo     “两路不一致且有一方置信度低”
       否则                                    → review   “两路不一致：L1=有问题 / L2=正常”
   4b. 否则（都说有问题，或只有一路有结论）        → fail_suggested
5. 都认为正常（设 conf 为各路置信度中最低的）：
   5a. 有一路画质不合格（已经不能重做）           → review   “认为正常但画质不足”
   5b. 有一路无法判断：conf ≥ high_conf          → pass，否则 review
   5c. conf ≥ high_conf                       → pass
   5c'. 两路都有结论、较高一路 ≥ high_conf、conf ≥ agree_min_conf → pass（未配置 agree_min_conf 时跳过）
   5d. 能重做                                  → redo     “认为正常但置信度低”
   5e. 否则                                    → review   “重做后置信度仍低”
```

注意两点：报问题不看置信度（第 4b 步，哪怕置信度只有 0.48 也送人工确认）；两路都说有问题但指向不同的手，结论仍是 `fail_suggested`，只是 `agree` 记为不一致。

### 6.3 决策表（实际运行代码得到）

`high_conf = 0.8`、`low_conf = 0.5`、`max_redo = 1`。第一次尝试 `attempt = 0`，最后一次尝试 `attempt = 1`。

| L1 | L2 | 第一次尝试 | 最后一次尝试 |
| --- | --- | --- | --- |
| 有问题 0.9 | 有问题 0.8 | fail_suggested | fail_suggested |
| 有问题 0.9 | 正常 0.9 | review（两路不一致） | review |
| 有问题 0.4 | 正常 0.9 | redo（不一致且一方置信度低） | review |
| 正常 0.95 | 正常 0.9 | pass | pass |
| 正常 0.7 | 正常 0.9 | redo（置信度低） | review |
| 正常 0.9 | 无法判断 | pass | pass |
| 正常 0.7 | 无法判断 | review | review |
| 无法判断 | 无法判断 | redo | review |
| 正常 0.9，画质不合格 | 正常 0.9 | redo（画质不足） | review |
| 无 | 有问题 0.6 | fail_suggested | fail_suggested |
| 无 | 正常 0.85 | pass | pass |
| 无 | 无 | review | review |

### 6.4 判定模式和按项覆盖（`arbiter.arbitrate`）

`arbitrate()` 在 `judge()` 外面再包一层：

- `mode = off`：不计算，结论和影子结论都是 `not_assessed`。
- `mode = record_only`：照常算出影子结论，对外结论改为 `not_assessed`，理由前加“只记录（书面依据未落实）；按判定模式应为 ……”。
- `arbitration.per_check` 可以按检测项覆盖参数。默认疤痕要求 `high_conf = 0.9`（大模型对疤痕误报多）。
  例如疤痕只有 L2、正常、置信度 0.85、只记录模式、最后一次尝试 → 对外“未评估”，影子结论 `review`（0.85 < 0.9 且不能再重做）。
- `arbitration.strategy` 指定策略名，可以换成自己注册的策略（EXTENDING.md 第 2 节）。

### 6.5 离线评测为什么按“最后一次尝试”仲裁

线上检测时，系统可以让受检者重做一次再判；离线评测面对的是已经拍好的图，没法真的重做。如果按第一次尝试算，
很多样本会停在 `redo`，指标就失真了。所以离线评测和网页 demo 的最终结论都按 `attempt = max_redo` 计算，
另外记录按 `attempt = 0` 算出的 `shadow_first_attempt`，用来报告“首次尝试会被要求重做的比例”。

### 6.6 会话级汇总

结论的严重程度：`fail_suggested`（4）> `review`（3）> `redo`（2）> `pass`（1）> `not_assessed`（0）。

- 同一检测项在多个视角都有结论时，取最严重的（`arbiter.worst`）。例如手指在掌心视角 `pass`、手背视角 `review` → 会话的手指结论是 `review`。
- 网页 demo 的整体结论 `overall` 取所有 `judge` 模式检测项中最严重的。
- 报告里动作类步骤（握拳、对捏、深蹲等）完成为“满足”，未完成或超时为“需人工复核”。

---

## 7. 指标怎么算

代码在 `limbcheck/metrics.py`，指标定义和目标值见 METRICS.md。这里讲计算细节。

### 7.1 统计单位：按图片和按会话

- **按图片**：每张截图对每个检测项是一个单位。
- **按会话**：同一会话（`session_id`）的各张图合成一个单位：真值、L1、L2 都是“任一张有问题即有问题，全部正常才正常，其余为无法判断”；
  结论取最严重的。

验收标准主要看**按会话**，因为对用人方有意义的是“这个人有没有被发现”，而不是“这张图有没有被发现”。

### 7.2 单一模型的指标（`model_stats`）

对 L1、L2 分别统计（只看那一路自己的 `positive`，不看仲裁）：

```
检出率 recall      = 判为有问题的正样本 ÷ 全部正样本         （无法判断的正样本按漏检计，保守）
误报率 fpr         = 判为有问题的负样本 ÷ 全部负样本
特异度 specificity = 判为正常的负样本 ÷ 全部负样本
精确率 precision   = 判为有问题的正样本 ÷ 判为有问题的全部样本
覆盖率 coverage    = 给出明确结论的样本 ÷ 有标注的样本
```

未标注（真值为 `None`）的样本不参与这些比例。

### 7.3 系统（仲裁后）的指标（`system_stats`）

用影子结论统计，所以只记录的项目也能评估。`fail_suggested`、`review`、`redo` 都算“被拦下（flagged）”，因为它们都不会被自动放过：

```
检出率 recall          = 被拦下的正样本 ÷ 全部正样本
漏检率 miss_rate       = 被判“满足”的正样本 ÷ 全部正样本
误报率 false_alarm     = 被拦下的负样本 ÷ 全部负样本
一致率 accuracy        = （被拦下的正样本 + 被判“满足”的负样本）÷ 全部
自动通过率             = 结论为 pass 的比例
需人工比例 manual_rate = 结论为 fail_suggested 或 review 的比例
首次重做率             = 首次尝试结论为 redo 的比例
```

### 7.4 L1 / L2 一致性与 Cohen's κ（`agreement_stats`）

只统计两路都有明确结论的单位。设一致率为 po，L1、L2 判为有问题的比例分别为 p1、p2：

```
碰巧一致的概率 pe = p1 × p2 + (1 − p1) × (1 − p2)
κ = (po − pe) ÷ (1 − pe)
```

**算例**：100 个单位，两路都说有问题 8 个、都说正常 80 个、只有 L1 说有问题 4 个、只有 L2 说有问题 8 个。
一致率 po = 88%，看起来很高；但大部分样本是正常的，两路“都说正常”本来就容易碰上：p1 = 0.12，p2 = 0.16，
pe = 0.12 × 0.16 + 0.88 × 0.84 = 0.758，κ = (0.88 − 0.758) ÷ (1 − 0.758) = **0.50**，属于中等一致。这就是只看一致率会高估的原因。

### 7.5 公平性分组（`group_stats`）

按样本的 `meta` 字段分组（如 `skin_tone`、`lighting`、`device`、`scenario`），对每组算系统检出率和误报率，
再算组间最大差（`recall_gap`、`false_alarm_gap`）。规则：

- 不分组的字段：`note`、`side`、`job`、`source_image`、`subject_id`。
- 一个字段的取值要有 2–12 种才分组（太多说明不是分组字段）。
- 只有样本数 ≥ 10 的组参与算差距，避免一两个样本造成假的大差距。

### 7.6 逐指统计（`finger_level`）

每只手每根手指算一个单位，看“指出的是不是那根手指”。L2 只说了“某只手有问题”而没说哪根时，视为这只手五根都报了（会拉高逐指误报率，这是故意的保守处理）。

### 7.7 置信区间：Wilson 方法（`wilson`）

样本少时，比例的点估计很不可靠。所有比例都附 95% Wilson 区间（z = 1.96，n 为样本数，p = k / n）：

```
中心 = (p + z²/(2n)) ÷ (1 + z²/n)
半宽 = z × √( p(1−p)/n + z²/(4n²) ) ÷ (1 + z²/n)
区间 = [中心 − 半宽, 中心 + 半宽]，截断到 0–1
```

| 结果 | 点估计 | 95% 区间 | 说明 |
| --- | --- | --- | --- |
| 35 个缺指会话全部检出 | 100% | 90.1%–100% | 这是“区间下限 ≥ 90%”所需的最少样本 |
| 30 个检出 28 个 | 93.3% | 78.7%–98.2% | 点估计达标，下限不达标 |
| 73 个健全会话 0 误报 | 0% | 0–5.0% | 证明“误报率 ≤ 5%”的最少样本 |
| 188 个健全会话 0 误报 | 0% | 0–2.0% | 证明“≤ 2%”的最少样本 |
| 10 个 0 误报 | 0% | 0–27.8% | 10 个样本什么都证明不了 |

按三法则粗算，0/73 的上限约为 3/73 = 4.1%，0/188 约为 1.6%；Wilson 稍保守一些，代码用 Wilson。

### 7.8 体验与防作弊指标（`session_stats`）

来自网页 demo 导出的会话记录：

- 完成率：有步骤记录且没有任何步骤超时的会话比例；一次通过率：超时步骤数为 0 的会话比例；另有每次会话平均超时步骤数。
- 完成时长中位数、P90；端侧推理 P95（每次会话取 P95，再在会话之间取 P95）；L2 单次延迟 P95。
- 随机挑战：真值“攻击类型”为回放、照片等的会话，挑战结果不是“完成”就算**拦截成功**；攻击类型为“无”的真人会话，挑战结果不是“完成”就算**误拦**。

### 7.9 验收判定（`evaluate_acceptance`）

`configs/acceptance.yaml` 每条写一个指标路径（如 `fingers.session.system.recall`）、比较符、目标值、可选的 `use`（`value` / `ci_low` / `ci_high`）和 `min_n`。
路径以检测项名开头时到 `metrics.checks` 下查找。结果有五种：

| 状态 | 中文 | 条件 |
| --- | --- | --- |
| `pass` | 通过 | 满足比较条件 |
| `fail` | 未达标 | 不满足 |
| `insufficient` | 样本不足 | 样本数 < `min_n`（不论数值多好） |
| `record` | 仅记录 | `op: record`，只展示数值 |
| `missing` | 无数据 | 指标不存在或为空（例如没配大模型就没有 L2 指标） |

---

## 8. 网页 demo 的内部逻辑

### 8.1 `server.py`（只用 Python 标准库）

启动时把仓库根目录加入 `sys.path`、设置 `LIMBCHECK_CONFIG_DIR`，然后直接导入 `limbcheck` 里不依赖第三方库的模块
（`core`、`prompts`、`vlm`、`arbiter`、`registry`），保证网页和离线评测用同一套提示词、解析和仲裁。

| 接口 | 作用 |
| --- | --- |
| 静态文件 | `index.html`、`app.js`、`vendor/`（MediaPipe 的 JS 与 wasm，已本地化） |
| `GET /models/<文件名>` | 指向仓库根目录的 `models/`；只取文件名部分，`../` 之类的路径穿越无效 |
| `GET /api/config` | 是否启用大模型、模型名、是否 Schema 模式、提示词版本，以及 `config.generated.json` 的全部内容 |
| `POST /api/review` | 请求 `{view, enabled, images}`；按 `l2_checks_for` 确定要问的检测项，调用大模型，返回 `Review.to_dict()`；没配置大模型时返回 503 |
| `POST /api/arbitrate` | 请求 `{job, frames: [{view, l1, l2}]}`；返回 `by_view`（每视角每项的结论，含 `shadow_first_attempt`）、`by_check`（每项取最严重）、`overall` |

大模型配置：只设了 `VLM_API_KEY` 时默认用百炼（DashScope）的 `qwen3.7-plus`；设了 `VLM_BASE_URL` 和 `VLM_MODEL` 时用指定的服务。请求体上限 40 MB。

### 8.2 `app.js` 主循环

```
loop()（每一帧，requestAnimationFrame）
  grabFrame()          把当前视频帧画到离屏画布 grab 上 —— 本帧的检测、截图、取色都用这同一帧
  按当前步骤 needs 运行 Pose 或 Hand 模型（只跑需要的那个，省算力），记录耗时
  readPose / readHands 整理结果；assignSides 确定左右手
  每 15 帧更新一次受检部位亮度
  runStep(step, now)   推进当前步骤的状态机
```

`readPose` 数人数时只计“主要关节平均可见度 > 0.5”且躯干中心与已计入的人相距 > 0.15（归一化坐标）的人，
避免背景误检或同一个人被重复检测成两个人。

### 8.3 步骤状态机

每个步骤是 `STEP_LIB` 里的一个对象，约定这些字段：

| 字段 | 作用 |
| --- | --- |
| `title` | 步骤名 |
| `needs` | `['pose']` 或 `['hands']`，决定每帧跑哪个模型 |
| `text(st)` | 横幅文字和语音提示（可随阶段变化） |
| `init(st)` | 步骤开始时初始化（可选） |
| `check(st, now)` | 每帧调用，返回 `{hint}`（未到位，给提示）、`{ok: true}`（到位）或 `{done: true, status, note}`（直接结束） |
| `hold` | 到位后需保持的毫秒数，默认 `capture.hold_ms`（1000） |
| `onHold(st, now)` | 保持够时间后调用：采集截图，或进入下一阶段（`{done: false}`），或结束（`{done: true}`） |
| `timeout(st)` / `timeoutFactor` | 自定义超时 / 默认超时（25 秒）的倍数 |
| `onTimeout(st)` | 超时时调用。有画面就按当前画面采集并返回 `review`，否则 `timeout` |

`runStep` 的逻辑：`check` 返回到位就开始计时，不到位就清零；计时满 `hold` 调 `onHold`；整步超过时限调 `onTimeout`。
步骤结果有三种：`pass`（完成）、`review`（需人工复核）、`timeout`（未完成）。

语音提示做了去重：同一句话 6 秒内不重复，任意两句之间至少隔 3.5 秒（强制播报的除外）。

### 8.4 各步骤做什么

| 步骤 id | 到位条件 | 采集视角 |
| --- | --- | --- |
| `env` 环境准备 | 检测到人、只有一个人、受检部位亮度在范围内、全身 13 个关节可见度都 > 0.55 且在画面内 | — |
| `stand` 正面站立 | 全身入框，双手腕低于肩 | `stand`；记录各肢段左右比和腿长基线 |
| `back` 背对镜头 | 躯干四点可见度 > 0.3，且鼻子不可见（< 0.5）或左右肩在画面中的顺序与正面相反 | `back` |
| `arms` 侧平举 → 上举 | 第一阶段肘角 ≥ 150°、手腕与肩高度差 ≤ 0.3 × 躯干长；第二阶段手腕高于鼻子、肘角 ≥ 140° | `arms`、`arms_up` |
| `legs` 单腿抬起 | 膝盖抬起使 (膝 y − 髋 y) ÷ 站立时大腿长 ≤ 0.5，左右各一次 | — |
| `squat` 深蹲 | 髋部下降超过腿长的 25%，再站回到 8% 以内 | — |
| `palms` / `backs` | 双手入镜且够大（掌长 ≥ 0.07 × 画面高），朝向正确（第 4.4 节），伸直手指数达到要求 | `palms` / `backs` |
| `flex` 握拳 → 张开 | 每只手伸直手指 ≤ 1，再张开 | — |
| `pinch` 拇指对捏 | 双手拇指依次碰到食指、中指、无名指、小指尖，超时为默认的 1.5 倍 | — |
| `challenge` 随机挑战 | 见 8.5 节 | — |

**照顾缺指者的放宽规则**（`needOpen`）：手部步骤开始后的前 6 秒要求五指全部伸直，之后放宽为 3 指，避免缺指的人永远“到不了位”。
这时页面仍提示“请尽量张开五指”。

### 8.5 随机手势挑战（防回放）

- 从 1–4 中随机抽 `challenge_rounds`（2）个不重复的数字，每轮要求“用任意一只手伸出 N 根手指”。
- 每轮限时 `challenge_round_ms`（5 秒），数对并保持 0.6 秒算完成，记录响应时间；超时记一轮失败，进入下一轮。
- 整个步骤的时限 = 轮数 × 5 秒 + 3 秒。有任一轮失败，步骤结果为 `review`（“疑似回放或配合不足”）。
- 挑战被**随机插在第一个手部步骤之后的某个位置**，深蹲和单腿抬起的先后也随机交换（`makeSteps`）。

原理：录像是事先录好的，不可能知道这次会随机要求几根手指、在什么时候要求，所以回放会在挑战中超时。
它防不住实时换脸、虚拟摄像头注入等攻击，正式方案需要实人认证 SDK 或原生 App（见 README“已知限制”）。

### 8.6 采集与 L2 并行

`captureView(st, view)` 每个视角只采集一次：

1. 用当前帧算该视角的 L1（`l1ForView`），受检部位过暗或过亮时把 L1 标为画质不合格。
2. 截一张**干净截图**（不带关键点叠加）：手部视角裁剪到双手外接框四周扩 30%，长边最多 1280 像素，JPEG 质量 0.88；身体视角用整帧。
3. 截一张带关键点叠加的 480 像素缩略图，放进报告当证据。
4. 记录每只手的左右、判定方式（`model` / `x-order`）、朝向。
5. 立即把干净截图发给 `/api/review`，**不等结果**，流程继续往下走。

全部步骤结束后 `endRun()` 等所有大模型请求返回，把结果按视角配上，一次性调用 `/api/arbitrate`，再生成报告。

### 8.7 验证模式与导出

勾选“验证模式”后，测试员在右侧填写受检者编号、缺失手指、美甲、纹身、疤痕、发色、四肢、攻击类型、光线、设备等（`readValidation`）。
`configs/checks.json` 里新增的检测项会自动多一个“有 / 无 / 未标注”选择框（`addExtraLabelFields`）。
导出的 JSON 格式为 `limbcheck.demo_report/v2`（第 11.4 节），用 `limbcheck import-demo` 导入。

---

## 9. 合成缺指样本

真实缺指样本难找。在它们到位之前，可以在健全手的图片上把一根手指“抹掉”，先比较各模型对“手指不见了”的反应
（尤其是关键点模型会不会把手指脑补回来）。代码在 `limbcheck/synth.py`，命令是 `limbcheck synth-missing`。

**两种切法**（`CUTS`）：

| 切法 | 含义 | 抹掉的范围 |
| --- | --- | --- |
| `mcp` | 整指缺失 | 食指到小指：从指根往 PIP 方向 18% 处起，经 PIP、DIP 到指尖；拇指：从 MCP 起 |
| `pip` | 远端两节缺失 | 食指到小指：从 PIP 起；拇指：从 IP 起 |

**步骤**：

1. **画要抹掉的区域**（`finger_mask`）：沿上述关键点画一条粗线，线宽为 0.21 × 掌长（拇指 0.26）；指尖再沿 DIP→TIP 方向外延 45%，
   末端画一个圆盖住指甲和指腹边缘；最后再膨胀 5×5。
2. **只让背景参与修复**（`remove_finger`）：把要抹掉的区域向外膨胀（核大小约 0.25 × 掌长）得到外围一圈，用放宽阈值的肤色掩码（ΔE 35）
   找出这一圈里属于相邻手指的像素，先替换成这一圈里背景像素的中位色。否则修复算法会把旁边手指的肤色“抹”进来，留下一根模糊的假手指。
3. **修复**：OpenCV Telea 算法，半径 max(3, 0.06 × 掌长)。
4. **补噪声、羽化**：修复区域往往过于平滑，按背景亮度波动补一点高斯噪声（标准差最多 4）；边缘用 σ = 1.5 的高斯模糊过渡。
5. **生成样本**：只用手部视角的图片，跳过真值里已标有缺失手指的；每张图把“哪只手 × 哪根手指 × 哪种切法”的组合打乱（随机种子固定，可复现），
   最多取 `--max-per-image` 个；原图另存一份（`__orig`）作对照；每个合成样本单独作为一个会话，真值写成缺失的那根手指。

**局限**：合成图有修复痕迹，和真实残肢外观不同（真实残端有皮肤褶皱、疤痕，没有“被擦掉”的背景）。合成结果只能作参考，
最终指标必须用真实缺指样本验证（VALIDATION.md 第 8 节）。

---

## 10. 代码逐模块导读

### 10.1 依赖关系

所有模块都从 `configs/` 读配置（经 `core.config_dir()`）。下表“依赖”指导入了本包里的哪些模块：

| 模块 | 依赖（本包内） | 第三方库 |
| --- | --- | --- |
| `core` | — | 无（只用标准库） |
| `registry` | — | 无 |
| `prompts` | core | 无 |
| `vlm` | core、prompts | 无 |
| `arbiter` | core、registry | 无 |
| `keypoints` | — | numpy |
| `geometry` | core、keypoints | numpy；模糊度用 OpenCV |
| `appearance` | core、geometry | numpy |
| `config` | core、prompts | PyYAML |
| `adapters/*` | keypoints、geometry、config、core | mediapipe、rtmlib、torch + sam2（各自用到时才导入） |
| `l1checks` | core、geometry、appearance、keypoints、registry | numpy |
| `pipeline` | registry、arbiter、config、core、geometry、l1checks、prompts、vlm | numpy、OpenCV、PyYAML |
| `dataset` | core | numpy；读写图片用 OpenCV |
| `metrics` | core、arbiter、dataset | — |
| `report` | core | 无 |
| `synth` | core、dataset、geometry、keypoints、adapters.contour | OpenCV |
| `viz` | core、keypoints（传入的流水线对象提供模型） | OpenCV |
| `bench` | metrics | — |
| `cli` | 以上全部（在各命令里按需导入） | tqdm |

前五个只用标准库，网页 demo 的 `server.py` 不用 pip 安装就能直接导入它们。模型相关的重依赖都在用到时才导入，
没装的依赖只在真正用到对应节点时报错。

### 10.2 `limbcheck/` 包

| 文件 | 做什么 | 关键函数 / 类 |
| --- | --- | --- |
| `core.py` | 从 `checks.json` 读出检测项和视角；定义结论常量、`Finding`、`CheckResult`、多路合并 | `config_dir()`、`VIEW_CHECKS`、`EXPECTED_HANDS`、`Finding`、`CheckResult.to_dict/from_dict`、`merge_any` |
| `keypoints.py` | 统一不同模型的关键点格式；左右手分配 | `HAND_FINGER_IDS`、`MP33`、`COCO17`、`WB133_*_HAND`、`Hand`、`Body`、`body_from_array`、`assign_sides` |
| `geometry.py` | L1 的几何与颜色公式 | `palm_len`、`finger_extended`、`orient_sign`、`facing`、`analyze_fingers`、`fingers_result`、`rgb_to_lab`、`nails_delta_e`、`nails_result`、`limbs_result`、`quality` |
| `appearance.py` | 发色基线 | `hair_stats`、`hair_result` |
| `l1checks.py` | L1 判定的上下文和内置判定注册 | `L1Context`（关键点按需计算并缓存，`_timed` 记耗时）、`fingers_keypoints`、`fingers_contour`、`nails_delta_e`、`limbs_pose`、`hair_color` |
| `adapters/base.py` | 关键点模型的接口约定 | `KeypointModel`（`hands(img)`、`bodies(img)`） |
| `adapters/__init__.py` | 内置模型节点的注册 | `mediapipe`、`rtmw`、`rtmw+hand`、`skin`、`mpseg`、`sam2` |
| `adapters/mediapipe_adapter.py` | MediaPipe Hand / Pose Landmarker | `MediaPipeKeypoints`（`close()` 释放资源） |
| `adapters/rtmw_adapter.py` | RTMW 全身 133 点、RTMPose-hand | `RTMWKeypoints`（rtmlib 要 BGR 输入；手部 21 点平均分 < 0.3 视为没看到） |
| `adapters/segmenter.py` | MediaPipe 多类别分割 | `SelfieMulticlass`（同一张图只分割一次，缓存持有图片引用）、类别常量 `HAIR`、`BODY_SKIN` |
| `adapters/contour.py` | 手指轮廓复核 | `SkinMask`、`SegmenterMask`、`Sam2Mask`、`tip_coverage`、`contour_result` |
| `prompts.py` | 提示词加载、拼装、JSON Schema、输出解析与转换 | `load_prompts`、`l2_checks_for`、`schema`、`build_user_prompt`、`build_messages`、`parse_reply`、`to_check_results` |
| `vlm.py` | OpenAI 兼容接口客户端（urllib） | `VLMClient.from_env`、`chat`、`review`、`Review`、`file_to_data_url` |
| `arbiter.py` | 仲裁 | `judge`（默认策略）、`arbitrate`、`agreement`、`worst`、`SEVERITY` |
| `pipeline.py` | 组装节点、对一张图跑 L1 + L2 + 仲裁 | `Pipeline.from_spec`、`Pipeline.run`、`load_pipeline_spec`（支持 `${VAR:-默认值}`）、`encode_jpeg` |
| `config.py` | 读阈值、岗位、验收标准；导出网页配置 | `load_thresholds`、`normalize_profile`、`load_job_profiles`、`check_mode`、`web_config`、`export_web_config`、`model_path` |
| `registry.py` | 可替换节点的注册表 | `register`、`get`、`create`、`load_plugins`、`names`、`close_all` |
| `dataset.py` | 数据集格式、真值规则、两种导入方式 | `Sample`、`load_samples`、`truth_positive`、`write_labels_template`、`import_folder`、`import_demo_reports` |
| `metrics.py` | 指标计算与验收判定 | `wilson`、`rate`、`image_units`、`session_units`、`model_stats`、`system_stats`、`agreement_stats`、`group_stats`、`compute`、`evaluate_acceptance` |
| `report.py` | 把指标渲染成 `report.md`；多次运行对比 | `render`、`compare`、`STATUS_CN` |
| `synth.py` | 合成缺指样本 | `finger_mask`、`remove_finger`、`synthesize` |
| `viz.py` | 把关键点、手部掩码和结论画到图上 | `draw` |
| `bench.py` | L1 耗时吞吐、L2 并发压测 | `bench_pipeline`、`bench_vlm` |
| `cli.py` | 命令行入口 | `cmd_eval`、`cmd_score`、`cmd_compare` 等，`main()` 结束时调用 `close_all()` |

### 10.3 一次离线评测的调用链（`limbcheck eval`）

```
cli.cmd_eval
  ├─ dataset.load_samples(data/manifest.jsonl)
  ├─ cli._pipeline → pipeline.load_pipeline_spec(yaml) → Pipeline.from_spec
  │     registry.load_plugins → registry.create("segmenter" / "contour" / "keypoints", spec) → make_vlm
  ├─ 对每个样本：Pipeline.run(img, view, meta)
  │     enabled_checks（视角能看到 ∩ 岗位未关闭）
  │     L1Context → 每个检测项依次调用注册的 L1 函数 → merge_any
  │     vlm.review（如果配置了大模型）
  │     quality（受检部位亮度）→ 过暗过亮时把 L1 标为画质不合格
  │     arbitrate(attempt = max_redo) 和 arbitrate(attempt = 0)
  │     单个节点抛异常只记进 errors，不中断评测
  ├─ 写 predictions.jsonl、run.json（流水线描述、git 版本、提示词版本、阈值）
  └─ _write_report → metrics.compute → evaluate_acceptance → metrics.json、report.md
```

`limbcheck score` 跳过“跑模型”这一步，直接对已有的预测文件算指标；数据是网页 demo 导入的、运行目录里又没有预测文件时，
自动使用 `predictions/demo_browser.jsonl`（浏览器端的结果）。

### 10.4 命令一览

| 命令 | 作用 |
| --- | --- |
| `eval` | 在数据集上跑流水线、出报告 |
| `score` | 对已有预测重新算指标（改了验收标准，或评估浏览器端结果时用） |
| `compare` | 多次运行横向对比 |
| `import-demo` | 导入网页 demo 导出的报告（同一会话重复导入会跳过） |
| `import-folder` / `labels-template` | 导入图片文件夹 + 标注表 / 生成标注表模板（按文件名猜视角） |
| `synth-missing` | 合成缺指样本 |
| `bench` / `bench-vlm` | L1 耗时吞吐压测 / L2 并发压测 |
| `vlm-probe` | 对一张图调用一次大模型，看原始输出和解析结果（`--show-prompt` 显示提示词） |
| `viz` | 画出关键点、掩码和结论，`--only misses` 只画漏检 |
| `export-web-config` | 由 `configs/` 生成网页配置 |
| `list` | 列出所有可替换节点、检测项、岗位、流水线、提示词版本 |

容量估算（`bench.py` 文件头）：单卡可支撑的实时路数 ≈ 单卡吞吐（张/秒）÷ 每路每秒要分析的帧数（建议 5–10）。
`bench-vlm` 的 `ok_rate` 是返回了非空文字的比例，`json_ok_rate` 是能解析出 JSON 的比例，后者更能反映大模型在高并发下是否稳定。

### 10.5 环境变量

| 变量 | 作用 |
| --- | --- |
| `LIMBCHECK_CONFIG_DIR` | 配置目录，默认仓库的 `configs/` |
| `LIMBCHECK_MODELS_DIR` | 模型文件目录，默认仓库的 `models/` |
| `LIMBCHECK_PROMPT_VERSION` | 提示词版本，默认见 `checks.json` 的 `default_prompt_version` |
| `VLM_BASE_URL` / `VLM_MODEL` / `VLM_API_KEY` / `VLM_SCHEMA` | 大模型地址 / 模型名 / 密钥 / 是否用 JSON Schema |
| `PORT` / `HOST` | 网页 demo 的端口（默认 8000）和监听地址（默认 127.0.0.1） |

---

## 11. 数据格式示例

### 11.1 `CheckResult`（某一路对一个检测项的判断）

```json
{
  "check": "fingers",
  "source": "L2:Qwen/Qwen3-VL-8B-Instruct",
  "positive": true,
  "confidence": 0.86,
  "parts": ["right_hand.index"],
  "quality_ok": true,
  "findings": [
    {"check": "fingers", "part": "right_hand", "label": "missing_or_short", "positive": true, "confidence": 0.86,
     "source": "L2:Qwen/Qwen3-VL-8B-Instruct", "value": ["index"], "note": ""},
    {"check": "fingers", "part": "left_hand", "label": "normal", "positive": false, "confidence": 0.93,
     "source": "L2:Qwen/Qwen3-VL-8B-Instruct", "value": 5, "note": ""}
  ],
  "note": "右手食指仅见一节"
}
```

`findings` 是逐条的具体判断，`value` 视来源而定：长度比、ΔE、可见度、头发颜色统计，或大模型返回的手指列表。

### 11.2 `predictions.jsonl` 的一行

下面是用 `mediapipe-skin` 流水线（不接大模型、精密装配岗位）跑 MediaPipe 官方测试图 `right_hands.jpg` 的真实输出。
这张图里有两只右手，这里删掉了第二只手的重复内容；`id`、`session_id` 由 `limbcheck eval` 从清单里补上，示例中的值是虚构的：

```json
{
  "id": "right_hands", "session_id": "right_hands",
  "view": "hand",
  "pipeline": "mediapipe-skin",
  "results": {
    "fingers": {
      "l1": {"check": "fingers", "source": "L1:mediapipe+contour-skin", "positive": false, "confidence": 0.943,
             "parts": [], "quality_ok": true,
             "findings": [{"part": "right_hand", "label": "normal", "positive": false, "confidence": 0.943, "source": "L1:mediapipe",
                           "value": {"thumb": 0.945, "index": 0.909, "middle": 1.0, "ring": 0.927, "little": 0.738}}]},
      "l2": null,
      "check": "fingers", "decision": "pass", "shadow": "pass",
      "reason": "仅 L1:mediapipe+contour-skin：正常且置信度高",
      "agree": null, "mode": "judge", "shadow_first_attempt": "pass"
    },
    "nails": {
      "l1": {"check": "nails", "source": "L1:deltaE", "positive": false, "confidence": 0.623, "parts": [], "quality_ok": true,
             "findings": [{"part": "right_hand", "label": "none", "positive": false, "confidence": 0.623, "source": "L1:deltaE",
                           "value": {"thumb": 20.4, "index": 16.3, "middle": 16.6, "ring": 14.7, "little": 15.1}}]},
      "l2": null,
      "check": "nails", "decision": "review", "shadow": "review",
      "reason": "重做后置信度仍低（0.62）",
      "agree": null, "mode": "judge", "shadow_first_attempt": "redo"
    }
  },
  "latency_ms": {"keypoints": 62.8, "contour": 438.8},
  "quality": {"brightness": 113.0, "blur": 708.8, "ok": true, "issues": []},
  "detail": {"hands": [{"side": "right", "score": 0.943}], "tip_coverage": {"right": {"thumb": 1.0, "index": 1.0, "middle": 1.0, "ring": 1.0, "little": 1.0}}},
  "errors": []
}
```

可以看到：手指的两路 L1（长度比、轮廓复核）都正常，合并后置信度取较低的 0.943，自动满足；美甲色差最大 20.4，接近阈值 22，
置信度 0.623，第一次尝试会要求重做，最后一次尝试送人工复核。

### 11.3 `manifest.jsonl` 的一行（一个样本）

```json
{"id": "S20261020-0931_backs", "image": "images/S20261020-0931_backs.jpg", "view": "backs",
 "session_id": "S20261020-0931", "subject_id": "P007",
 "truth": {"fingers": {"right": ["index"], "left": []}, "nails": {"right": false, "left": true},
           "tattoo": false, "scar": null, "hair": "natural", "limbs": false, "attack": "none"},
 "meta": {"scenario": "藏一根手指", "lighting": "正常", "device": "iPhone 13", "source": "demo", "job": "precision"}}
```

真值里 `null` 表示没标注，不参与该项统计。`fingers` 列出缺失或明显短缩的手指；`hair` 取 `natural` / `dyed` / `highlights`；
`attack` 取 `none` / `replay` / `photo` / `virtual_camera` / `other`。

### 11.4 网页导出的报告 `limbcheck.demo_report/v2`（结构）

```
{
  "schema": "limbcheck.demo_report/v2",
  "session_id", "subject_id", "at", "validation",
  "job": {"id", "name"},
  "truth": {...}, "meta": {...},                       ← 验证模式下测试员填写的内容
  "client": {"ua", "platform", "language", "video": {"w", "h"}},
  "config": {"prompt_version", "thresholds_version", "vlm_model"},
  "perf": {"pose_ms": {"p50", "p95", "n"}, "hand_ms": {...}, "fps", "delegate"},
  "steps": [{"id", "title", "status", "note", "ms", "metrics", "views"}],
  "frames": [{"view", "step", "img"（干净截图 data URL）, "l1": {检测项: CheckResult}, "brightness", "at_ms", "hands"}],
  "reviews": [Review.to_dict()…],                       ← 大模型原始输出、解析结果、耗时
  "arbitration_by_view": {视角: {检测项: Arbitration}},
  "arbitration": {"by_check", "overall", "error"},
  "duration_ms"
}
```

`import-demo` 把每个 `frames` 元素拆成一个样本（图片解码保存到 `images/`），把 L1、L2、仲裁结果写进 `predictions/demo_browser.jsonl`，
把步骤、挑战结果、端侧性能写进 `sessions.jsonl`。

### 11.5 `metrics.json` 的结构（指标路径）

```
metrics
├─ summary                      样本数、有预测的样本数、会话数
├─ checks
│  └─ fingers
│     ├─ image / session
│     │  ├─ n、n_labeled、n_pos、n_neg
│     │  ├─ l1 / l2             recall、fpr、specificity、precision、coverage（每项含 value、k、n、ci_low、ci_high）
│     │  ├─ system              recall、miss_rate、false_alarm_rate、accuracy、auto_pass_rate、manual_rate、redo_rate_first_attempt、decisions
│     │  ├─ agreement           rate、kappa
│     │  ├─ fairness            各分组字段的 groups、recall_gap、false_alarm_gap
│     │  └─ misses / false_alarms   漏检、误报的样本 id（最多 50 个）
├─ finger_level                 l1 / l2 的逐指 recall、fpr
├─ latency                      各阶段 p50_ms、p95_ms、mean_ms
├─ sessions                     completion_rate、duration_median_s、device_hand_ms_p95 等
└─ liveness                     attack_interception、attack.<类型>.interception、genuine_false_reject
```

验收标准里的 `fingers.session.system.recall` 就是沿着这棵树往下找。

---

## 12. 常见疑问

**为什么系统不输出“不通过”？**
远程拍摄条件差，模型会出错；残疾和健康信息属于敏感个人信息，自动淘汰在法律上风险很高。所以系统只给“疑似问题·待人工确认”，
最终结论由人作出，并提供线下替代和申诉渠道（HANDOFF.md 第 7 章）。

**为什么只要有一路报问题就送人工，不看置信度？**
漏掉一个真实问题的代价（把不能胜任的人放进精密岗位，或反过来让人在入职后才被发现）比多让复核员看一眼高得多。
送人工的比例单独统计（需人工比例），作为体验和成本指标去优化，而不是靠调高报问题的门槛。

**为什么大模型说“不确定但可能有问题”也算有问题？**
同上，宁可多送人工。置信度减半，让报告里能看出这是一条不太确定的判断。

**关键点模型为什么会“脑补”手指？**
它是在大量完整的手上训练的，学到的是“手应该长什么样”，输出永远是 21 个点。手指缺失时，它会根据其余手指的位置把缺失的那根“推测”出来，
而且常常推测得很像。M0 的小样本合成实验里，关键点单独只检出了 30%，加上轮廓复核到 60–93%（README“验证状态”）。

**为什么无法判断的正样本按漏检计，而不是排除？**
排除会让指标虚高：一个模型只要对难的样本说“不知道”，检出率就好看了。按漏检计，同时单独报告覆盖率，两者一起看才公平。

**`record_only` 的项目既然不判定，为什么还要算？**
为了在拿到书面依据之前就积累数据：影子结论说明“如果判定会是什么结果”，等依据落实、改为 `judge` 时，已经知道它的检出率和误报率。

**离线评测和网页上的结论会不一样吗？**
仲裁规则、提示词、解析代码是同一份，阈值来自同一个配置文件，同一张截图、同样的 L1 和 L2 结果会得到相同结论。
不同之处在于离线评测可以换模型（如 RTMW、SAM 2、不同大小的 Qwen-VL），这正是离线评测的用途。

**改了阈值，网页没变化？**
网页读的是生成的 `config.generated.json`，要运行 `limbcheck export-web-config` 重新生成；CI 会检查两者是否同步。

**为什么受检部位过暗只影响 L1，不影响 L2？**
L1 的颜色判定（美甲、肤色掩码）直接依赖像素亮度，暗了一定不准；大模型有自己的画质判断（`image_quality`），由它自己报告 `poor`。

**左右手判断错了会怎样？**
左右错了，手指、美甲的部位会记反，但“有没有问题”的结论不变。仲裁比较两路是否一致时只比到“哪只手”，左右记反会让 `agree` 变成不一致，
送人工复核。报告里注明了每只手是按模型还是按位置判定的，复核员可以据此核对。

**置信度 0.8 的意思是 80% 的把握吗？**
不是。见第 3.6 节：它是按规则算出的分数，只用来和仲裁阈值比较。要让它接近真实概率，需要用 M0 数据做校准。
