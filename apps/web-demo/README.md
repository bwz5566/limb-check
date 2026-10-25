# 网页验证 Demo

候选人在浏览器里按语音提示完成一组动作；浏览器内用 MediaPipe 实时检测关键点并做 L1 判定，
每个视角的证据截图可选地发给视觉大模型做 L2 复核，最后由 `server.py` 按与离线评测相同的规则仲裁，生成报告。
**验证模式**下，测试员可以记录受检者的真实情况，导出报告后用 `limbcheck` 计算检出率、误报率等指标。

## 运行

只需要 Python 3.10 及以上，不用 pip 安装任何包（`server.py` 只用标准库，直接复用仓库里的 `limbcheck` 代码）。
模型文件在仓库根目录 `models/`，已随仓库提供，无需联网下载。

```bash
cd apps/web-demo
python server.py            # Windows 也可以双击 start.bat；macOS 双击 start.command
```

用 Chrome 或 Edge 打开 http://localhost:8000 。必须经 `server.py` 访问，直接双击 `index.html` 打不开。

浏览器只允许在 `localhost` 或 https 页面上打开摄像头。用手机测试时，可以把电脑上的服务通过 https 反向代理出去，
或在手机上用 USB 调试转发端口；同一局域网直接访问 `http://电脑IP:8000` 会被浏览器拒绝打开摄像头。

## 启用大模型复核（L2）

启动前设置环境变量（任何 OpenAI 兼容接口都可以）：

```bash
# 内网部署的 Qwen-VL（vLLM / vLLM-Ascend，见 deploy/）
export VLM_BASE_URL=http://10.0.0.8:8001/v1
export VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct
export VLM_SCHEMA=1            # 可选：vLLM 支持用 JSON Schema 约束输出

# 或阿里云百炼（截图会发到外部服务，需在授权书里写明）
export VLM_API_KEY=sk-...      # 只设这一项时默认 DashScope + qwen-vl-max
python server.py
```

Windows 命令行用 `set VLM_API_KEY=sk-...`。页面右上角会显示“大模型复核：已启用（模型名）”。

## 流程

步骤由岗位画像（`configs/job_profiles/*.yaml` 的 `steps`）决定，例如精密装配：

| 步骤 | 采集的视角 | 用于 |
| --- | --- | --- |
| 环境准备 | — | 光线（只看受检部位）、单人、全身入框 |
| 正面站立 | `stand` | 四肢完整与左右长度比；发色；纹身、疤痕 |
| 转身背对镜头 | `back` | 发色；背侧纹身、疤痕 |
| 双臂侧平举 → 上举 | `arms`、`arms_up` | 上肢完整与活动；手臂纹身、疤痕 |
| 单腿抬起、深蹲（仓储岗位） | — | 下肢活动 |
| 双手掌心朝前 | `palms` | 手指数量与长度 |
| 翻转为手背朝前 | `backs` | 手指；美甲；手背纹身、疤痕 |
| 握拳 → 张开 | — | 手指屈伸 |
| 拇指依次对捏（精密岗位） | — | 精细操作 |
| 随机手势挑战（随机插在手部步骤之间） | — | 防回放录像 |

- 动作到位并保持约 1 秒自动进入下一步；不到位时语音纠正；超时后按当前画面采集并标记“需人工复核”。
- 手部步骤前 6 秒要求五指全部张开，之后放宽，避免缺指的人卡住。
- 左右手：模型给出的左右可信时直接采用，否则按画面位置（受检者右手在原始画面左侧）。报告里会注明每只手是怎么判定的。
- 每个视角采集一张干净截图（关键点检测和截图用的是同一帧），在浏览器里算 L1，同时并行发给大模型，不阻塞流程。

## 报告

- **岗位要求核对**：每个检测项的结论（满足 / 需人工复核 / 疑似问题·待人工确认），只记录的项目显示“只记录”和按判定模式应得的结论；动作完成情况；随机挑战。
- **各视角证据与判定**：每张截图的 L1、L2 结果和仲裁理由。
- 分步结果、手指长度比、指甲色差、大模型原始输出、端侧推理耗时。
- 系统不输出最终“不通过”，所有疑似项都交人工确认。

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
