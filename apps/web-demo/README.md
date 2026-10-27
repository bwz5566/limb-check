# 网页验证 Demo

候选人在浏览器里按语音提示完成一组动作；浏览器内用 MediaPipe 实时检测关键点，**只用来引导动作和决定什么时候截图**。
每个视角的高清证据截图交 `server.py` 用 Python 流水线做 L1 判定（服务端判定，见下文），可选地再发给视觉大模型做 L2 复核，
最后按与离线评测相同的规则仲裁，生成报告。
**验证模式**下，测试员可以记录受检者的真实情况，导出报告后用 `limbcheck` 计算检出率、误报率等指标。

## 运行

只需要 Python 3.10 及以上。不装任何包也能跑（`server.py` 基础功能只用标准库），但这时只有浏览器端结果，误判多；
建议先在仓库根目录 `pip install -e ".[mediapipe,rtmw]"` 启用服务端判定（见下文）。
模型文件在仓库根目录 `models/`，已随仓库提供，无需联网下载。

```bash
cd apps/web-demo
python server.py            # Windows 也可以双击 start.bat；macOS 双击 start.command
```

用 Chrome 或 Edge 打开 http://localhost:8000 。排查误判时打开 http://localhost:8000/?debug ：画面上会实时标出每根手指（中节 / 末节）、指缝、肢体是否"确认看到"（绿 = 确认，红 = 没确认，灰 = 无法判断）以及掌长、亮度、背景是否接近肤色。必须经 `server.py` 访问，直接双击 `index.html` 打不开。

浏览器只允许在 `localhost` 或 https 页面上打开摄像头。用手机测试时，可以把电脑上的服务通过 https 反向代理出去，
或在手机上用 USB 调试转发端口；同一局域网直接访问 `http://电脑IP:8000` 会被浏览器拒绝打开摄像头。

## 服务端判定（L1）

浏览器里的小模型只能看 1280×720 的实时画面，被遮挡、蜷起的手指也会"脑补"出关键点，用来做最终判定误判太多。
所以每张证据截图都会上传到 `server.py`，用 `limbcheck` 的 Python 流水线重新判定（关键点 + 手指肤色轮廓 + 人体分割复核），
仲裁时优先用服务端结果；服务端没装依赖或出错时，自动退回浏览器端结果（报告里会注明）。

```bash
pip install -e ".[mediapipe]"           # 最低配置：服务端用 MediaPipe（与浏览器同一模型，但看的是截图，规则更全）
pip install -e ".[mediapipe,rtmw]"      # 推荐：RTMW 全身 133 点 + RTMPose 手部（CPU 可跑，每张图预计 1–3 秒，未实测）
```

- 默认 `LIMBCHECK_PIPELINE=auto`：先试 `rtmw-skin-cpu`，加载不了再用 `mediapipe-skin`。启动时终端会打印用的是哪一个，页面右上角也会显示"服务端判定：已启用（…）"。
- RTMW 首次启动会从 `download.openmmlab.com` 下载 ONNX 模型（几百 MB，存到 `~/.cache/rtmlib/hub`），之后离线可用。
- 有 GPU 服务器时用 `LIMBCHECK_PIPELINE=rtmw-sam2-gpu`（RTMW + SAM 2.1 分割，见 `deploy/rtx`）。
- 设为 `off` 只用浏览器端结果；页面上也可以取消勾选"服务端判定"。

## 启用大模型复核（L2）

把 `apps/web-demo/.env.example` 复制一份，改名为 `.env`，填上 Key 后重新启动 `server.py`（任何 OpenAI 兼容接口都可以）：

```ini
# 阿里云百炼：只填这一项（默认 qwen3.7-plus；截图会发到外部服务，需在授权书里写明）
VLM_API_KEY=sk-...
# 可选：换模型
VLM_MODEL=qwen3.8-max

# 或内网部署的 Qwen-VL（vLLM / vLLM-Ascend，见 deploy/）
# VLM_BASE_URL=http://10.0.0.8:8001/v1
# VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct
# VLM_SCHEMA=1
```

`.env` 已加入 `.gitignore`，不会提交到 git。也可以不建文件，启动前用环境变量设置同名项（`set VLM_API_KEY=...` / `export VLM_API_KEY=...`），环境变量优先。
页面右上角会显示“大模型复核：已启用（模型名）”。

## 流程

步骤由岗位画像（`configs/job_profiles/*.yaml` 的 `steps`）决定，例如精密装配：

| 步骤 | 采集的视角 | 用于 |
| --- | --- | --- |
| 环境准备 | — | 光线（只看受检部位）、单人、全身入框 |
| 正面站立 | `stand` | 四肢完整（可见度、双脚、关节弯折、人体轮廓、左右长度比）；纹身、疤痕 |
| 转身背对镜头 | `back` | 背侧纹身、疤痕；发色 |
| 双臂侧平举 → 上举 | `arms`、`arms_up` | 上肢完整；协调性（外展角、左右对称）；手臂纹身、疤痕 |
| 交叉摸肩 | — | 协调性（跨中线动作，左右交替、节律） |
| 露出脖子（左右转头） | `neck` | 颈部纹身 |
| 双臂近景（外侧 → 内侧） | `arms_out`、`arms_in` | 手臂纹身、疤痕的大小估算 |
| 单腿抬起、原地高抬腿、深蹲（仓储岗位） | — | 协调性（抬腿高度、平衡、交替节律、下蹲深度） |
| 双手掌心朝前 | `palms` | 手指数量与长度、掌面完整 |
| 翻转为手背朝前 | `backs` | 手指、手背完整；美甲；手背、手指纹身 |
| 握拳 → 张开 | — | 手指屈伸 |
| 拇指依次对捏（精密岗位） | — | 精细操作 |
| 随机手势挑战（随机插在手部步骤之间） | — | 防回放录像 |

- 动作到位并保持约 1 秒自动进入下一步；不到位时语音纠正；超时后按当前画面采集并标记“需人工复核”。
- 手部步骤前 6 秒要求五指全部张开，之后放宽，避免缺指的人卡住。
- 左右手：模型给出的左右可信时直接采用，否则按画面位置（受检者右手在原始画面左侧）。报告里会注明每只手是怎么判定的。
- 每个视角采集一张干净截图（关键点检测和截图用的是同一帧），在浏览器里算 L1，同时并行发给大模型，不阻塞流程。

## 报告

- **岗位要求核对**：每个检测项的结论（满足 / 不符合 / 需人工复核 / 疑似问题·待人工确认），只记录的项目显示“只记录”和按判定模式应得的结论；随机挑战。
- **纹身 / 疤痕核验**：每处的裁剪小图、估算尺寸、命中的录用标准条款；疑似污渍时提醒清洁并一键重拍。
- **肢体协调性（动作分析）**：每个动作的指标和结论。
- **各视角证据与判定**：每张截图的 L1、L2 结果和仲裁理由。
- 分步结果、手指长度比、指甲色差、大模型原始输出、端侧推理耗时。
- 能明确判断的直接给“满足 / 不符合”（阈值随页面上选的筛选严格度变化），置信度不足的交人工。

## 验证模式（采集带标注的数据）

1. 勾选右侧“验证模式”，填写受检者编号（不要填姓名）、场景、缺失手指、美甲、纹身、疤痕、发色、四肢、是否攻击测试、光线、设备。
2. 完成检测后核对标注（检测结束后仍可修改），点“导出报告 JSON”。
3. 导入并计算指标：

```bash
limbcheck import-demo limbcheck-S2026....json -o data/m0
limbcheck score --data data/m0 --run runs/m0-browser
limbcheck eval --data data/m0          # 用离线流水线重跑同一批截图，对比不同模型
```

完整的验证方案见 [docs/VALIDATION.md](../../docs/VALIDATION.md)。

## 配置从哪来

页面不写死任何阈值、岗位、检测项：`server.py` 把 `config.generated.json`（由 `limbcheck export-web-config` 从 `configs/` 生成）交给页面。
改了 `configs/` 里的阈值或岗位后要重新生成，否则单元测试 `test_web_config_in_sync` 会失败提醒。
提示词、输出解析和仲裁在 `server.py` 里调用 `limbcheck.prompts` / `limbcheck.vlm` / `limbcheck.arbiter`，与离线评测完全一致。

URL 参数：`?job=warehouse` 预选岗位；`?fast=1` 缩短超时（自测用）。

## 文件

| 文件 | 说明 |
| --- | --- |
| `server.py` | 静态文件、`/models/`（指向仓库 `models/`）、`/api/config`、`/api/review`（L2）、`/api/arbitrate`（仲裁） |
| `app.js` | 引导状态机、L1 判定（与 `limbcheck/geometry.py` 同一套公式）、报告、导出 |
| `index.html` / `style.css` | 页面 |
| `config.generated.json` | 自动生成的配置，请勿手改 |
| `vendor/` | MediaPipe Tasks Vision 0.10.14（JS 与 wasm，本地化以免依赖外网 CDN） |

## 已知限制

- 浏览器无法可靠识别虚拟摄像头注入，正式方案需要实人认证 SDK 或原生 App。
- 关键点模型在手指缺失时可能“脑补”出不存在的手指，这是 L1 只作初判、必须有 L2 和人工复核的原因。
- 色差法美甲判断受光线和肤色影响，透明、裸色甲油基本识别不了。
- 截图和标注只保存在本机浏览器内存和导出的文件里，demo 不做服务端存储；正式方案的加密存储和到期删除在 M1 实现。
