# 扩展与替换：预留的可更新、可替换、可修改节点

需求会变：换模型、改阈值、改提示词、加岗位、加检测项、加拍摄步骤。本仓库把这些都做成了**配置项或注册节点**，
大多数改动只改 `configs/`，不碰核心代码。本文逐个说明改哪里、要不要写代码、改完怎么验证。
每个阈值在判定里怎么起作用、各模块的代码结构，见 [GUIDE.md](GUIDE.md)。

## 总览

| 想做的事 | 改哪里 | 要写代码吗 | 本文 |
| --- | --- | --- | --- |
| 调阈值（手指长度比、色差、置信度…） | `configs/thresholds.yaml` | 否 | 第 1 节 |
| 改仲裁规则的参数 / 按检测项单独设置 | `configs/thresholds.yaml` 的 `arbitration` | 否 | 第 2 节 |
| 换一套仲裁策略 | 插件里 `@register("arbiter", …)` | 少量 | 第 2 节 |
| 改大模型提示词 | `configs/prompts/v2.json`（新版本） | 否 | 第 3 节 |
| 换大模型（DashScope / 昇腾 vLLM / 本地） | 环境变量或流水线配置 `vlm:` | 否 | 第 4 节 |
| 换关键点 / 轮廓 / 分割模型 | 流水线配置 `configs/pipelines/*.yaml` | 已内置的不用；新模型写适配器 | 第 5 节 |
| 加一个 L1 判定算法 | 插件里 `@register("l1", …)` | 是 | 第 6 节 |
| 加岗位、改岗位要求 | `configs/job_profiles/*.yaml` | 否 | 第 7 节 |
| 外观项从“只记录”改为“判定” | 岗位画像的 `basis` + `mode` | 否 | 第 7 节 |
| **加一个检测项** | `configs/checks.json` + 提示词 + 岗位画像 | 否（只靠 L2 时）；要 L1 时写判定 | 第 8 节 |
| 加一个拍摄视角 / 引导步骤 | `configs/checks.json` 的 `views` + 网页 `STEP_LIB` | 网页步骤要写 | 第 9 节 |
| 改验收标准 | `configs/acceptance.yaml` | 否 | 第 10 节 |
| 改数据格式 | `limbcheck/dataset.py`，带版本号 | 是 | 第 11 节 |
| 配置放到别处 / 模型放到别处 | `LIMBCHECK_CONFIG_DIR` / `LIMBCHECK_MODELS_DIR` | 否 | 第 12 节 |

`limbcheck list` 会列出当前所有可用节点、检测项、视角、岗位、流水线和提示词版本。

**任何改动之后都做三件事**：

1. `pytest`：单元测试（含“配置与网页同步”“只改配置新增检测项”的测试）。
2. 改了 `configs/` 里的阈值、岗位、检测项后运行 `limbcheck export-web-config`，网页 demo 才会用上新配置。
3. 在回归集上跑 `limbcheck eval`，用 `limbcheck compare` 对比改动前后（见 VALIDATION.md 第 9 节）。

---

## 1. 阈值：`configs/thresholds.yaml`

所有数值门限集中在这里，网页（经 `config.generated.json`）和离线评测读同一份：

| 段 | 含义 |
| --- | --- |
| `capture` | 动作保持时长、每步超时、随机挑战轮数和每轮时限 |
| `quality` | 受检部位亮度范围、单人、手的最小尺寸 |
| `pose` | 关节可见度门限、四肢左右长度比范围 |
| `fingers` | 各指相对中指的参考长度比 `ref`、判为短缩的系数 `short_factor`、伸直角度、对捏距离 |
| `contour` | 指尖落在手部掩码内的最低比例、肤色阈值 |
| `nails` | 指甲与手背色差 ΔE 门限 |
| `hair` | 判为染发的彩度门限 |
| `arbitration` | 见第 2 节 |

改完把 `version` 加一，评测报告和网页导出的报告里都会记录版本号。
阈值只在校准集上调，测试集上报告（VALIDATION.md 第 7 节）。

---

## 2. 仲裁：参数、按项覆盖、自定义策略

默认策略（`limbcheck/arbiter.py` 的 `default`）：

- 任何一路认为有问题 → “疑似问题·待人工确认”（系统不输出最终“不通过”）；两路矛盾 → 人工复核或重做。
- 只有所有来源都认为正常、置信度都 ≥ `high_conf`、画质合格时才自动“满足”。
- 置信度低或画质差时，第一次请候选人重做（`max_redo`），之后转人工。

参数在 `thresholds.yaml`：

```yaml
arbitration:
  strategy: default
  high_conf: 0.8
  low_conf: 0.5
  max_redo: 1
  per_check:
    scar: {high_conf: 0.9}     # 疤痕要求更高置信度才自动通过
```

**换策略**：在自己的模块里注册一个函数，签名 `fn(l1, l2, cfg, attempt) -> (结论, 理由)`，`l1` / `l2` 是 `CheckResult` 或 `None`：

```python
# mycompany_ext/arbiters.py
from limbcheck.core import FAIL_SUGGESTED, PASS, REVIEW
from limbcheck.registry import register

@register("arbiter", "l2_veto")
def l2_veto(l1, l2, cfg, attempt):
    """大模型说有问题就送人工，否则沿用 L1。"""
    if l2 is not None and l2.positive:
        return FAIL_SUGGESTED, "L2 认为有问题"
    if l1 is not None and l1.positive is False and l1.confidence >= cfg["high_conf"]:
        return PASS, "L1 正常且置信度高"
    return REVIEW, "证据不足"
```

然后 `thresholds.yaml` 写 `strategy: l2_veto`，流水线配置 `plugins: [mycompany_ext.arbiters]`。
网页 demo 的仲裁也走 `server.py` 里的同一个函数；自定义策略要在 demo 里生效，需让 `server.py` 能 import 到该模块（放进仓库或设 `PYTHONPATH`）并在 `server.py` 顶部 import 一次。

---

## 3. 提示词：带版本号，可回滚

- 提示词全部在 `configs/prompts/<版本>.json`：系统提示（护栏）、各视角描述、各检测项的规则和合法标签（`positive` / `negative`）。
- 输出格式、解析、校验、转换成 `CheckResult` 的代码在 `limbcheck/prompts.py`，**改提示词不用改代码**。
- 解析有护栏：只保留本次启用的检测项；标签不在约定范围内按“不确定”处理；输出不是 JSON 视为无法判断（转人工）。

改提示词的流程：

1. 复制 `v1.json` 为 `v2.json`，把 `version` 改成 `v2`，修改规则。
2. `LIMBCHECK_PROMPT_VERSION=v2 limbcheck vlm-probe --image 某张图.jpg --view backs --show-prompt` 看生成的提示词和模型回复。
3. `LIMBCHECK_PROMPT_VERSION=v2 limbcheck eval --data data/regression ...`，与 v1 的运行对比。
4. 达标后把 `configs/checks.json` 的 `default_prompt_version` 改为 `v2`，运行 `limbcheck export-web-config`。

评测报告、网页报告、每条 L2 记录都带提示词版本号，线上出问题可以追溯和回滚。

---

## 4. 大模型：任何 OpenAI 兼容接口

L2 客户端（`limbcheck/vlm.py`）只依赖 OpenAI 兼容的 `/chat/completions`，换服务商只改地址和模型名：

| 部署 | `VLM_BASE_URL` | `VLM_MODEL` 示例 |
| --- | --- | --- |
| 昇腾 vLLM-Ascend（内网，推荐） | `http://<昇腾服务器>:8001/v1` | `Qwen/Qwen3-VL-8B-Instruct`、`Qwen/Qwen3-VL-32B-Instruct` |
| RTX 6000 上的 vLLM | `http://<RTX 服务器>:8001/v1` | 同上 |
| 阿里云百炼 DashScope | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen3.7-plus` |
| 本地 Ollama | `http://localhost:11434/v1` | `qwen2.5vl:7b` |

- 网页 demo：启动 `server.py` 前设环境变量。离线评测：环境变量，或命令行 `--vlm-base-url --vlm-model`，或流水线配置里 `vlm: {base_url: ..., model: ...}`（支持 `${VAR}` 引用环境变量）。
- `VLM_SCHEMA=1`（或 `--vlm-schema`）让 vLLM 用 JSON Schema 约束输出；服务不支持时自动退回普通模式。
- 部署脚本见 `deploy/ascend/` 和 `deploy/rtx/`。

---

## 5. 模型节点：关键点 / 轮廓复核 / 分割

流水线配置（`configs/pipelines/*.yaml`）组装节点，节点名来自注册表：

```yaml
name: rtmw-sam2
keypoints: {type: rtmw+hand, device: cuda, mode: balanced}   # mediapipe / rtmw / rtmw+hand / 包.模块:类
contour: {type: sam2, device: cuda}                          # skin / mpseg / sam2 / none
segmenter: {type: mpseg}                                     # mpseg / none
l1: [fingers.keypoints, fingers.contour, nails.delta_e, limbs.pose, hair.color]
vlm: env
plugins: []
```

命令行可以临时覆盖：`limbcheck eval --data ... --keypoints rtmw --contour skin`。

**接入新模型**只要实现对应接口，再注册或直接写完整路径：

| 节点 | 接口 | 说明 |
| --- | --- | --- |
| keypoints | `name`；`hands(img) -> list[Hand]`；`bodies(img) -> list[Body]` | `img` 为 RGB uint8；`Hand.pts` 为 21×2 像素坐标（MediaPipe 手部点序），`side` 为受检者本人的 `right` / `left`（不知道就 `None`，会按画面位置推断）；`Body.joints` 用 `limbcheck/keypoints.py` 里的关节名 |
| contour | `name`；`hand_mask(img, hand, th) -> np.ndarray` | 返回该手的二值掩码（H×W，uint8），用于检查指尖是否真的有皮肤 |
| segmenter | `name`；`categories(img)` / `mask(img, classes)` | 头发、皮肤等语义分割，用于发色和 `mpseg` 轮廓 |

```python
# mycompany_ext/models.py
from limbcheck.keypoints import Hand
from limbcheck.registry import register

@register("keypoints", "my-hand-model")
class MyHandModel:
    name = "my-hand-model"
    def __init__(self, weights="/models/my_hand.onnx", device="cuda"):
        ...
    def hands(self, img):
        return [Hand(pts=pts, score=s, side=None) for pts, s in self._infer(img)]
    def bodies(self, img):
        return []
    def close(self):          # 可选：评测结束时自动调用，释放显存
        ...
```

流水线写 `keypoints: {type: my-hand-model, device: cuda}` 和 `plugins: [mycompany_ext.models]`；
或不注册，直接写 `keypoints: {type: "mycompany_ext.models:MyHandModel"}`。

**网页端**的端侧模型是 MediaPipe（`apps/web-demo/models/` 由 `server.py` 指向仓库根目录 `models/`）。
换端侧模型需要改 `app.js` 的 `init()` 和 `readHands` / `readPose`，只要输出同样的 21 点 / 33 点结构，其余判定逻辑不用动。

---

## 6. L1 判定算法

每个检测项可以挂多个 L1 判定，任一路报问题即报问题（`core.merge_any`）。在插件里注册：

```python
# mycompany_ext/l1.py
from limbcheck.core import CheckResult, Finding
from limbcheck.registry import register

@register("l1", "nails.classifier", check="nails", needs="keypoints")
def nails_classifier(ctx):              # ctx: limbcheck.l1checks.L1Context
    if not ctx.hands:
        return None                     # None 表示这一路不适用
    findings = []
    for h in ctx.hands:
        crop = ...                      # 用 h.pts 从 ctx.img 裁指甲
        p = my_model(crop)
        findings.append(Finding("nails", f"{h.side}_hand", "polish" if p > 0.5 else "none", p > 0.5, float(abs(p - 0.5) * 2), "L1:nails-cls"))
    pos = [f for f in findings if f.positive]
    return CheckResult("nails", "L1:nails-cls", bool(pos), max(f.confidence for f in findings),
                       [f.part for f in pos], True, findings)
```

`L1Context` 提供：`img`、`view`、`th`（阈值）、`profile`、`meta`、按需计算并缓存的 `hands` / `bodies`、
`contour` / `segmenter` 节点、`check_params(检测项)`（岗位画像里的 `params`）、`_timed(名称, 函数)`（计时写进报告）。

流水线配置的 `l1:` 列表加上 `nails.classifier`，`plugins:` 加上模块名。单个判定抛异常时会被记录到该样本的 `errors`，不影响其他判定。

---

## 7. 岗位画像：`configs/job_profiles/*.yaml`

```yaml
id: precision
name: 精密装配
description: 需双手十指完成抓握与捏取等精细操作
requirements: [双手十指能完成抓握与捏取, 上肢活动正常, 入职前需卸除美甲]
steps: [env, stand, back, arms, palms, backs, flex, pinch, challenge]   # 网页引导步骤（顺序；挑战会随机插入手部步骤之间）
checks:
  limbs:    {enabled: true, mode: judge, params: {upper: true, lower: false}}
  fingers:  {enabled: true, mode: judge, params: {pinch: true}}
  nails:    {enabled: true, mode: judge}
  tattoo:   {enabled: true, mode: record_only}
  scar:     {enabled: true, mode: record_only}
  hair:     {enabled: true, mode: record_only}
  liveness: {enabled: true, mode: judge}
basis: []
```

- `mode`：`judge` 给出结论；`record_only` 照常计算并保存，对外显示“只记录”（评测里仍按判定模式统计，用来积累基线）；`off` 完全不做。
- **外观项（纹身、疤痕、发色）没有书面依据时不能设为 `judge`**：`basis` 里没有覆盖该项时，加载配置会自动降级为 `record_only` 并给出警告。有依据后这样写：

```yaml
basis:
  - {checks: [tattoo], doc: "XX 厂员工手册（2025 版）", clause: "第 3.2 条：食品车间员工手臂不得有可见纹身"}
checks:
  tattoo: {enabled: true, mode: judge}
```

- 新增岗位：复制一个文件改 `id` 等字段即可；网页下拉框和离线 `--job` 自动出现。

---

## 8. 新增一个检测项

例：增加“首饰”（jewelry），只靠 L2 大模型判断。**不用改任何 Python / JS 代码**（`tests/test_extending.py` 用这个例子验证了全链路）：

1. `configs/checks.json` 的 `checks` 里加：
   ```json
   "jewelry": {"name": "首饰", "views": ["backs", "hand"], "basis_required": true, "appearance": true}
   ```
   `views` 决定在哪些截图上评估；与岗位操作无关的项目建议 `basis_required: true`。
2. `configs/prompts/v1.json`（或新版本）的 `checks` 里加规则和合法标签：
   ```json
   "jewelry": {"rule": "jewelry：对每只手各输出 1 条，part 为 \"right_hand\" 或 \"left_hand\"；label 为 \"none\" 或 \"ring\"。",
               "positive": ["ring"], "negative": ["none"], "per_hand": true}
   ```
3. 需要的岗位画像里启用：`jewelry: {enabled: true, mode: record_only}`。
4. `limbcheck export-web-config`。

之后自动生效的部分：

- 网页：L2 提示词带上新项；报告“岗位要求核对”和各视角表格多一行；验证模式多一个“首饰：有 / 无 / 未标注”的标注框；导出的真值里有 `jewelry`。
- 离线：标注表加一列 `jewelry`（是 / 否）即可导入；评测报告和指标路径 `jewelry.session.system.recall` 等自动出现。
- 验收：在 `acceptance.yaml` 加一条即可。

需要 L1 时，再按第 6 节注册一个 `check="jewelry"` 的判定。

---

## 9. 新增拍摄视角或引导步骤

**视角**（一张证据截图对应一个视角）：`configs/checks.json` 的 `views` 里加一项（`kind` 为 `body` 或 `hand`，手部视角写 `expected_hands`），
在需要的检测项的 `views` 里引用它，提示词文件的 `view_desc` 里写一句画面描述。

**网页引导步骤**：在 `apps/web-demo/app.js` 的 `STEP_LIB` 里加一项。一个步骤包括：

```js
side_raise: {
  title: '侧身站立', needs: ['pose'],                 // needs：本步要跑的模型（pose / hands）
  text: () => '请向左转 90 度，侧身站立',             // 语音和横幅提示
  check(st, now) { ...; return { ok: true } },        // 每帧判断：{ok} 到位；{hint} 纠正提示；{done} 直接结束
  onHold(st) { captureView(st, 'side'); return { done: true }; },   // 保持到位后：采集视角截图（自动做 L1、发起 L2）
  onTimeout: captureOnTimeout('side', 'pose'),         // 超时：有人就按当前画面采集并标记需复核
},
```

再把步骤 id 写进岗位画像的 `steps`。离线评测不关心步骤，只关心视角。

---

## 10. 验收标准：`configs/acceptance.yaml`

```yaml
- id: fingers_recall
  name: 手指缺失检出率（仲裁后，按会话）
  metric: fingers.session.system.recall    # 指标路径，见 METRICS.md 第 4 节
  op: ">="                                 # >= / <= / record（只记录）
  target: 0.90
  use: ci_low                              # 可选：比较置信区间下限（更严格）；默认比较点估计
  min_n: 30                                # 样本不足时判“样本不足”
```

不同阶段可以各放一份（如 `acceptance-pilot.yaml`），评测时用 `--acceptance` 指定。

---

## 11. 数据格式与版本

| 数据 | 格式 | 版本 |
| --- | --- | --- |
| 单路结论 | `CheckResult` / `Finding`（`limbcheck/core.py`），网页 `app.js` 里同构 | 加字段向后兼容；`from_dict` 忽略未知字段 |
| 数据集 | `manifest.jsonl`（每行一张图：`id` `image` `view` `session_id` `subject_id` `truth` `meta`） | 字段说明见 `limbcheck/dataset.py` 开头 |
| 预测 | `predictions.jsonl`（每行一张图的 `results`：每项的 `l1` `l2` `decision` `shadow` `reason` `agree` `mode`） | 同上 |
| 网页导出报告 | `schema: limbcheck.demo_report/v2` | 导入时校验；改结构时升为 v3，并在 `import_demo_reports` 里保留 v2 的读取 |

`meta` 里的任意字段（光线、设备、场景、肤色分组…）都会自动成为公平性分组的维度，不用改代码。

---

## 12. 部署相关的可替换项

| 环境变量 | 作用 |
| --- | --- |
| `LIMBCHECK_CONFIG_DIR` | 配置目录（默认仓库 `configs/`）。可以给每个客户一份配置，代码不变 |
| `LIMBCHECK_MODELS_DIR` | 模型文件目录（默认仓库 `models/`），网页和离线共用 |
| `LIMBCHECK_PROMPT_VERSION` | 临时切换提示词版本 |
| `VLM_BASE_URL` `VLM_MODEL` `VLM_API_KEY` `VLM_SCHEMA` | 大模型服务 |

---

## 13. 已知需要代码改动的地方（尚未做成配置）

- 网页端的引导步骤逻辑（第 9 节），因为每个动作的到位判断不同。
- 端侧模型替换（第 5 节末）。
- 服务端实时推流（WebRTC）与复核台：属于 M1 / M3 的工程，本仓库目前只到“验证”。
