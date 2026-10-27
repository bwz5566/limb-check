# 手部检测：完整性 + 纹身 / 疤痕录用标准

本文说明第 1 步（手部）怎么判、标准写在哪、怎么部署和调优。进度见 [PROGRESS.md](PROGRESS.md)。

## 1. 采集流程（网页 demo / 正式端相同）

| 步骤 | 视角 | 看什么 | 尺寸怎么量 |
| --- | --- | --- | --- |
| 正面站立、背面、侧平举、上举（约 2 米） | stand / back / arms / arms_up | 四肢、大臂、背部纹身疤痕 | 大臂、小臂关键点长度 |
| **露出脖子（左右转头，约 1 米）** | neck（两张图一起送大模型） | 脖子纹身 | 不需要（脖子有纹身即不录用） |
| **双臂近景：小臂横放胸前，先外侧后内侧（约 1 米）** | arms_out / arms_in | 小臂、手腕、手背纹身疤痕；自伤划痕多在小臂内侧 | 小臂关键点长度 |
| 双手掌心、双手手背（走近） | palms / backs | 手指、手掌、手背完整性；手指、手背、手掌纹身疤痕 | 手掌长、宽 |
| 握拳张开、拇指对捏、随机手势 | — | 手指功能、防作弊 | — |

岗位画像的 `steps` 决定启用哪些步骤，四个岗位都已加上 `neck` 和 `arms_close`。

## 2. 手掌 / 手背 / 手指完整性

三路证据，任一路有问题就交人工确认：

1. **关键点长度比**（原有）：各指长度 / 中指长度低于参考值 × 0.6 → 疑似缺失或短缩。
2. **轮廓复核**：关键点模型在手指或掌面缺失时会"脑补"出完整手形，所以用独立的皮肤掩码交叉检查——
   指尖段（原有）和**掌面 13 个采样点**（新增，`contour.palm_on_mask_min: 0.75`）落在掩码外即报疑似。
   肤色参照改为掌面 + 各指近节的多点投票，局部缺失时不会把背景当成肤色。
3. **大模型**：提示词 v2 增加 `deformed`（手指、手掌、手背畸形、残缺、并指多指、严重挛缩）。

另外，手部检测项要求**掌心、手背两个视角都拍到**（`checks.json` 的 `required_views`），缺一个就"需人工复核"，
保证报告上的"手掌手背均正常"有两面的证据。

## 3. 纹身 / 疤痕录用标准

标准全部写在 [`configs/policies/marks.yaml`](../configs/policies/marks.yaml)，判定代码在 `limbcheck/marks.py`。
每处标记判为四种之一：**pass** 符合标准、**review** 需人工核验、**block** 不符合标准（系统显示"疑似问题·待人工确认"并列出条款）、
**clean** 疑似污渍请清洁重拍。一个检测项取最严重的那处。系统始终不输出最终"不通过"。

### 3.1 招聘方标准与配置的对应

| 招聘方条款 | 配置项 | 默认动作 |
| --- | --- | --- |
| 攻击性图案（牛鬼蛇神、龙虎豹猛兽含睡眠闭眼、骷髅、刀具、帮派恐吓）无论大小不录用 | `content.*.action` | block |
| 辱骂文字、帮派符号、涉暴力图腾、低俗违禁 | `content.abusive_text / gang / violent_totem / vulgar` | block |
| 无文字或无明显符号可录用（普通文字、符号需看内容） | `content.text_neutral / symbol` | review |
| 小星星、叶片、云朵、小花、波浪线条 | `content.benign` | pass |
| 脖子有纹身，即使面积小也不录用 | `tattoo.regions.neck.action` | block |
| 大臂、小臂：面积 < 身份证 85.6×54 mm 放行 | `regions.upper_arm/forearm.size_ref: id_card` | pass |
| 大臂、小臂：面积 ≥ 身份证不录用…… | `large` | block |
| ……但洗过、色素明显变淡、无攻击性的放行 | `large_faded` | pass |
| 颜色过深，最终以核验为准 | `regions.*.dark` | 手臂 review；手指、手背 block |
| 手指：面积比 1 分硬币（直径 1.8 cm），≥ 硬币不录用 | `regions.finger.size_ref: coin`、`large` | block |
| 手指：整体洗过褪色的淡痕可录 | `regions.finger.large_faded` | pass |
| 手指纹身 ≤ 1 处可录、≥ 2 处不录用 | `regions.finger.max_count: 1` | block（按"哪只手哪根手指"跨视角去重） |
| 手背多处纹身叠加 | `regions.hand_back.max_count: 1` | block |
| 手背属外露高频部位，作业时持续外露 | `regions.hand_back.single` | **review**（见 3.3） |
| 纹身位置仅限大臂、小臂 | `regions.face/other.action` | review |
| 多条整齐平行划痕（自伤风险） | `scar.parallel_multiple` | block |
| 长条手术刀疤，岗位为精细作业 | `scar.surgical_long_fine_work`（岗位 `scar.params.fine_work: true`） | block；非精细作业 review |
| 严重增生影响手部 / 手臂灵活操作 | `scar.hypertrophic_severe` | block |
| 手背多处陈旧疤痕 | `scar.hand_back_multiple` | block |
| 疤痕同时搭配纹身 | `scar.with_tattoo`、`tattoo.covers_scar` | block（同一部位跨检测项也会合并判断） |
| 大面积长条疤痕 | `scar.large`（长 ≥ 80 mm 或面积 ≥ 身份证） | block |
| 仅单条短小、孤立、无增生的意外伤疤可录用 | `scar.short_single`（长 < 30 mm） | pass |
| 疑似污渍：清洁提醒 | `stain.action: clean` | 请清洁后重拍；重拍后仍在 → review |

长度、面积阈值（30 / 50 / 80 mm、临界 ±30%）是建议初值，需要用实测样本校准。

### 3.2 面积怎么估

1. 大模型对每处纹身、疤痕给出框（0–1000 相对坐标）。
2. 用画面里的人体当尺子算"毫米 / 像素"：手部视角用手掌长（腕→中指根，按 100 mm）和掌宽（食指根→小指根，按 70 mm），
   身体视角用小臂（250 mm）、大臂（300 mm）；取透视缩短最少的那段（偏保守）。成人平均值写在 `measure.anatomy_mm`。
3. 框的宽 × 高换算成 cm²，与身份证 46.2 cm² 或 1 分硬币 2.54 cm² 比较；在 ±30% 以内算临界 → 人工核验。
4. 估不出比例时，用大模型目测的厘米数；再不行用它给的"比参照物小 / 相近 / 大"。测量与目测方向相反时也交人工。

个体差异约 ±15%（面积约 ±30%），所以临界区不自动判。需要更准时可以让候选人把身份证举在纹身旁边，
在 `measure` 下接一个"参照物"比例来源即可（接口已预留：`meta.scale_source`）。

### 3.3 需要招聘方确认的两处解读

- **手背单处、小面积、浅色、无攻击性的纹身**：表格允许，但"不允许录用"清单写了"手背属于外露高频部位"。
  默认交人工核验；如果应直接放行，把 `regions.hand_back.single` 改成 `pass`，应直接拦截改成 `block`。
- **大臂、小臂颜色过深**：表格写"颜色过深 拦截，最终以核验为准"，默认交人工核验（不自动拦截）。

### 3.4 纹身、疤痕、污渍怎么区分

提示词要求大模型按外观区分：纹身是皮下色素、边缘清晰、表面平整；疤痕是皮肤凸起或凹陷、偏粉白、有纹理；
污渍和笔迹浮在表面、边缘模糊或有涂抹痕；痣、胎记、淤青不算。拿不准时 `uncertain: true` → 人工核验。
疑似污渍时，网页报告会显示清洁提醒和"已清洁，重拍"按钮，只重做拍到污渍的那一步，重新复核和仲裁。

## 4. 部署

- 网页 demo：`cd apps/web-demo && VLM_BASE_URL=... VLM_MODEL=... python server.py`，只依赖 Python 标准库。
  标准和阈值改完后运行 `limbcheck export-web-config` 重新生成网页配置。
- 大模型：任何 OpenAI 兼容接口（vLLM / vLLM-Ascend 部署 Qwen3-VL、阿里云百炼等）。框坐标默认按 Qwen3-VL 的 0–1000 相对坐标；
  若换用输出像素坐标的模型，把 `measure.bbox_format` 改为 `abs`。建议开启 `VLM_SCHEMA=1`（vLLM 支持）强制输出格式。
- 离线评测：`limbcheck eval --data ...` 时流水线会自动把比例传给大模型结果；`limbcheck vlm-probe --image 手背.jpg --view backs` 可单张调试。
- 标注：标注表 `tattoo` / `scar` 列填 `none` / `pass` / `review` / `block`（或 无 / 符合 / 核验 / 不符合）；网页验证模式也已改为这四档。

## 5. 后续优化空间

- 颜色深浅：在框内计算纹身与周围皮肤的亮度差，作为客观量化，替代大模型的"深 / 浅"判断。
- 纹身 / 疤痕分割模型（L1）：输出同样的 `Mark` 格式即可接入规则引擎，与大模型两路交叉。
- 参照物比例：检测画面中的身份证（长宽比 1.586）直接得到毫米 / 像素。
- 跨视角去重：手背、手臂同一处在多个视角出现时，目前按单视角最多处数计，可用关键点坐标对齐后精确去重。
