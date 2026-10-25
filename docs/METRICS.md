# 落地指标与各阶段需要的结果

本文回答两个问题：**这个项目要落地，需要看哪些指标；每个阶段要拿出什么结果才能往下走。**
指标的计算代码在 `limbcheck/metrics.py`，通过标准在 `configs/acceptance.yaml`（可直接改），
`limbcheck eval` / `limbcheck score` 会逐条判定并写进报告。

> 所有目标值都是**建议值**，需要和业务方、法务确认后写进 `configs/acceptance.yaml`。
> 交接文档第 11 章第 5 条“验收标准”仍是 P0 待定项。

---

## 1. 指标总览

六类指标。M0 是技术验证（内部志愿者），试点是一家工厂一个岗位（M4）。

### A. 判定准确性（核心）

| 指标 | 定义 | M0 目标 | 试点目标 | 指标路径 |
| --- | --- | --- | --- | --- |
| 手指缺失检出率（系统） | 有缺指的会话中，没有被自动放行的比例（判为“疑似问题 / 需复核 / 需重做”都算检出） | ≥ 90% | ≥ 95%，置信区间下限 ≥ 90% | `fingers.session.system.recall` |
| 健全手误报率（系统） | 手指正常的会话中，没有被自动放行的比例 | ≤ 5% | ≤ 3% | `fingers.session.system.false_alarm_rate` |
| 漏检率 | 有缺指却被自动判“满足”的比例（= 1 − 检出率，单独列出便于追责） | 记录 | ≤ 2% | `fingers.session.system.miss_rate` |
| L1 / L2 单独检出率 | 关键点 + 轮廓复核（L1）、视觉大模型（L2）各自的检出率，用来定位短板 | 记录基线 | 记录 | `fingers.image.l1.recall` / `.l2.recall` |
| 缺失手指定位准确率 | 逐指统计：报出的手指与真实缺失手指是否一致 | 记录 | 记录 | `finger_level.l1.recall` 等 |
| 四肢判定一致率 | 四肢“有无缺失 / 明显异常”与人工标注一致的比例 | ≥ 95% | ≥ 97% | `limbs.session.system.accuracy` |
| 动作到位一致率 | 系统判定“动作完成”与人工观看录像的结论一致的比例 | ≥ 95% | ≥ 95% | 人工抽查（见 VALIDATION.md 第 6 节） |
| 美甲检出率 / 误报率 | 有美甲被检出的比例；自然指甲被误报的比例 | 记录基线 | 检出 ≥ 90%（不透明甲油、甲片），误报 ≤ 5% | `nails.session.system.recall` / `.false_alarm_rate` |
| 纹身 / 疤痕 / 发色基线 | record_only 模式下照常计算，用来积累数据、评估将来能否转为判定 | 记录基线 | 有书面依据后另定 | `tattoo.session.system.recall` 等 |
| 需人工比例 | 被送去人工确认或复核的会话比例，直接决定复核工作量 | 记录 | ≤ 30%（视复核人力定） | `fingers.session.system.manual_rate` |
| 首次需重做比例 | 第一次拍摄就因画质或置信度低需要重做的比例 | 记录 | ≤ 15% | `fingers.session.system.redo_rate_first_attempt` |
| L1 / L2 一致性 | 两路结论一致的比例与 Cohen's κ | 记录 | κ ≥ 0.6 | `fingers.session.agreement.kappa` |
| 公平性差距 | 不同肤色分组、光线、设备之间检出率和误报率的最大差值（每组至少 10 例才计算） | 记录 | 误报率差距 ≤ 5 个百分点 | `fingers.session.fairness.<分组>.false_alarm_gap` |

为什么检出率按“没有被自动放行”算：系统从不输出最终“不通过”，“疑似问题·待人工确认”“需人工复核”“需重做”都会挡住自动放行，交给人看。真正的风险是**有问题却被自动判“满足”**，即漏检。

### B. 防作弊

| 指标 | 定义 | M0 目标 | 试点目标 | 指标路径 |
| --- | --- | --- | --- | --- |
| 回放攻击拦截率 | 对着摄像头播放预录视频时，被随机手势挑战拦下的比例 | 100%（≥ 10 次） | 100% | `liveness.attack.replay.interception` |
| 照片攻击拦截率 | 举照片时被拦截的比例 | 记录 | 100% | `liveness.attack.photo.interception` |
| 虚拟摄像头注入检出率 | 用虚拟摄像头软件推送视频时被识别的比例。浏览器内无法彻底识别，如实记录 | 如实记录 | 依赖采购的实人认证 SDK | `liveness.attack.virtual_camera.interception` |
| 真人误拦率 | 真人正常配合却没通过随机挑战的比例 | ≤ 5% | ≤ 3% | `liveness.genuine_false_reject` |

### C. 体验

| 指标 | 定义 | M0 目标 | 试点目标 | 指标路径 |
| --- | --- | --- | --- | --- |
| 完成率 | 所有步骤都没有超时的会话比例 | ≥ 90% | ≥ 95% | `sessions.completion_rate` |
| 完成时长 | 从开始到报告生成的时长中位数 / 90 分位 | 中位数 ≤ 180 秒 | 中位数 ≤ 180 秒，P90 ≤ 300 秒 | `sessions.duration_median_s` / `duration_p90_s` |
| 平均超时步数 | 每次检测平均有几步超时 | 记录 | ≤ 0.3 | `sessions.timeout_steps_mean` |
| 一次通过率 | 没有任何步骤超时的比例 | 记录 | ≥ 85% | `sessions.first_pass_rate` |
| 主观满意度 | 检测后问卷（1–5 分）：指引是否清楚、是否感到被冒犯 | 记录 | 平均 ≥ 4 分 | 问卷，不在代码里 |

### D. 性能与容量

| 指标 | 定义 | M0 目标 | 试点目标 | 指标路径 / 命令 |
| --- | --- | --- | --- | --- |
| 端侧推理延迟 | 浏览器里手部 / 姿态关键点每帧推理耗时 P95 | ≤ 100 ms（中端手机） | ≤ 100 ms | `sessions.device_hand_ms_p95` / `device_pose_ms_p95` |
| L2 单次延迟 | 一个视角发给大模型到拿到结果的 P95 | ≤ 5 秒 | 在下一步结束前返回 | `latency.l2.p95_ms`；压测用 `limbcheck bench-vlm` |
| L2 输出合规率 | 返回内容能解析为约定 JSON 的比例 | ≥ 98% | ≥ 99.5% | `limbcheck bench-vlm` 的 `json_ok_rate` |
| L1 服务端吞吐 | 单张 RTX 6000 上 RTMW + SAM 2 每秒处理张数 | 实测记录 | 支撑峰值并发 | `limbcheck bench` |
| 并发路数 | 单卡同时支撑多少路检测（L1 RTX 6000；L2 昇腾） | 实测记录 | ≥ 业务峰值 × 1.5 | `bench` / `bench-vlm --concurrency` |

### E. 业务效果（试点阶段才有数据）

| 指标 | 定义 | 试点目标 |
| --- | --- | --- |
| 人工复核工时 | 每位候选人平均人工复核分钟数 | 比现行线下面检节省 ≥ 50% |
| 复核推翻率 | 人工复核推翻系统建议的比例（按方向分开统计） | 记录，用于再训练 |
| 入职抽检不一致率 | 入职当天线下抽检与远程结论不一致的比例 | ≤ 2% |
| 申诉率 / 申诉成立率 | 候选人申诉比例；申诉后结论被改的比例 | 记录 |
| 单次成本 | 算力 + 带宽 + SDK + 复核人力 / 检测次数 | 低于现行流程 |

### F. 合规与治理（必须项，不达标不能上线）

| 项目 | 要求 |
| --- | --- |
| 单独同意 | 敏感个人信息单独同意，可随时中止；同意率与中止率记录在案 |
| 外观项书面依据 | 纹身、疤痕、发色只有在岗位画像里写明书面依据（`basis`）后才能从 record_only 改为 judge，代码会强制检查 |
| 不自动淘汰 | 系统只给建议；所有“疑似问题”必须人工确认；提供线下替代渠道和申诉 |
| 数据最小化与删除 | 视频不落盘；证据帧加密、按角色授权、到期删除（期限待定）；删除执行有日志 |
| 大模型护栏 | 只输出约定 JSON；只评估本次启用的项目；记录模型和提示词版本；换模型或提示词前跑回归 |
| 法务审查 | 检测范围、同意书、报告措辞、数据处理规则经劳动法律师审阅 |
| 公平性 | 各分组误报率差距在目标内；差距超标时先停用相关判定 |

---

## 2. 每个阶段需要的结果

| 阶段 | 必须拿出的结果 | 出口标准 |
| --- | --- | --- |
| **M0 技术验证**（第 1–2 周） | ① M0 评测报告（`limbcheck eval` 生成的 `report.md`），A–D 类指标每项都有实测数字和置信区间；② 模型对比表（MediaPipe vs RTMW + SAM 2；Qwen-VL 8B / 32B / 72B），`limbcheck compare` 生成；③ 容量报告：单卡并发路数、P50 / P95 延迟；④ 失败样本清单与原因分析（`limbcheck viz` 画图）；⑤ 用人方访谈纪要：目标岗位、判定标准、外观项依据 | `configs/acceptance.yaml` 中 M0 判定项全部“通过”，或差距有明确改进方案并获业务方认可 |
| **M1 核心能力**（第 3–8 周） | 服务端实时推理 + 推流；手指、四肢判定上线到内部环境；身份核验与防作弊 v1；内部 50 人次测试报告 | 手指检出率 ≥ 95%（区间下限 ≥ 90%），误报 ≤ 5%；回放拦截 100% |
| **M2 外观能力**（第 7–14 周） | 美甲、发色、纹身、疤痕的验证集与指标；有书面依据的项目给出判定阈值 | 有依据的外观项在验证集达标；没依据的保持 record_only |
| **M3 复核台与报告**（第 11–16 周） | 复核台、报告模板、同意书、数据处理规则 | 律师审查通过 |
| **M4 单厂试点**（第 17–22 周） | 试点评估报告：A–F 全部指标，尤其 E 类业务效果和人工推翻率 | 推广决策会通过 |

**M0 报告的最低内容**（缺一项都不算完成）：

1. 数据说明：人数、会话数、各场景数量、正负样本数、设备和光线分布。
2. 验收标准对照表（报告自动生成），“样本不足”的项要写明还差多少样本。
3. 每个检测项：L1、L2、仲裁后三行的检出率、误报率、置信区间；需人工比例；一致率。
4. 漏检和误报样本逐个列出，并附可视化图和原因（关键点“脑补”、光线、遮挡、模型输出不合规等）。
5. 防作弊：每类攻击的次数和拦截数。
6. 体验：完成率、时长分布、哪一步最容易超时。
7. 性能：端侧 P95、L2 P95、单卡并发路数。
8. 结论与下一步：哪些项达标、哪些不达标、改什么、需要什么数据。

---

## 3. 统计方法

### 3.1 按会话还是按图片

- **按会话**：一次检测（一个人走完一遍）是一个单位，任一视角报了问题就算这个会话报了问题。业务看的是这个，验收标准默认按会话。
- **按图片**：每个视角的截图是一个单位，用来分析是哪个视角、哪一路模型出的问题。

### 3.2 置信区间

报告里每个比例都附 95% Wilson 置信区间。样本少时区间很宽：30 个缺指会话全部检出，区间下限也只有 88.7%。
所以验收时既看点估计，也看区间；`acceptance.yaml` 里可以对某项加 `use: ci_low`，要求区间下限达标。

### 3.3 需要多少样本

| 想证明 | 0 次失误时需要的样本数（Wilson） | 粗算（三分之一法则：上限 ≈ 3 / n） |
| --- | --- | --- |
| 检出率区间下限 ≥ 85% | 22 个正样本 | 20 |
| 检出率区间下限 ≥ 90% | 35 个正样本 | 30 |
| 检出率区间下限 ≥ 95% | 73 个正样本 | 60 |
| 误报率区间上限 ≤ 5% | 73 个负样本 | 60 |
| 误报率区间上限 ≤ 2% | 188 个负样本 | 150 |

允许少量失误时样本要更多，例如 60 个正样本漏 1 个，下限 91.1%；漏 3 个，下限 86.3%。
M0 的建议是缺指正样本 ≥ 50 个会话、健全负样本 ≥ 75 个会话、回放攻击 ≥ 10 次（见 VALIDATION.md 的场景表）。

### 3.4 样本不足时

`acceptance.yaml` 每条可以设 `min_n`。样本数不到 `min_n` 时，报告给出“样本不足”而不是“通过”或“未达标”，避免用 5 个样本得出 100% 的结论。

### 3.5 离线评测按“最后一次尝试”仲裁

线上流程中，置信度低或画质差时会请候选人当场重做一次。离线评测没法真的重做，所以按最后一次尝试的规则仲裁（低置信直接转人工），
另外单独统计“首次尝试就会要求重做”的比例（`redo_rate_first_attempt`），用来估计线上的重做次数。

---

## 4. 指标路径一览（写 `acceptance.yaml` 时用）

```
<检测项>.<image|session>.<l1|l2>.<recall|fpr|specificity|precision|coverage>
<检测项>.<image|session>.system.<recall|miss_rate|false_alarm_rate|accuracy|auto_pass_rate|manual_rate|redo_rate_first_attempt>
<检测项>.<image|session>.agreement.<rate|kappa>
<检测项>.<image|session>.fairness.<分组字段>.<recall_gap|false_alarm_gap>
finger_level.<l1|l2>.<recall|fpr>
latency.<keypoints|contour|segment|l2>.<p50_ms|p95_ms|mean_ms>
sessions.<completion_rate|duration_median_s|duration_p90_s|timeout_steps_mean|first_pass_rate|device_pose_ms_p95|device_hand_ms_p95|l2_demo_p95_ms>
liveness.<attack_interception|genuine_false_reject>
liveness.attack.<replay|photo|virtual_camera|other>.interception
```

检测项：`fingers` `limbs` `nails` `tattoo` `scar` `hair`，以及 `configs/checks.json` 里新增的项目。
比例类指标的值是一个对象（`value` `k` `n` `ci_low` `ci_high`），`acceptance.yaml` 用 `use:` 选择比较哪一个。

---

## 5. 怎么得到这些数字

```bash
# 网页 demo 验证模式导出的报告（可以一次导入多份）
limbcheck import-demo reports/*.json -o data/m0
# 浏览器端的结果直接出指标
limbcheck score --data data/m0 --run runs/m0-browser
# 用离线流水线重跑同一批截图（换模型、换大模型都在这里比）
limbcheck eval --data data/m0 --pipeline configs/pipelines/mediapipe-skin.yaml
limbcheck eval --data data/m0 --pipeline configs/pipelines/rtmw-sam2-gpu.yaml --vlm-base-url http://昇腾服务器:8001/v1 --vlm-model Qwen/Qwen3-VL-8B-Instruct
limbcheck compare runs/m0-browser runs/<上一步生成的目录> ... -o runs/compare.md
# 性能
limbcheck bench --data data/m0 --pipeline configs/pipelines/rtmw-sam2-gpu.yaml
limbcheck bench-vlm --image data/m0/images/<任一手背截图>.jpg --view backs --concurrency 8 --requests 200
```

验证流程与场景设计见 [VALIDATION.md](VALIDATION.md)。
