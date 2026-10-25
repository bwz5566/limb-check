"""把指标渲染成 Markdown 报告；多次运行横向对比。"""
from __future__ import annotations

from .core import CHECK_CN, DECISION_CN

STATUS_CN = {"pass": "通过", "fail": "未达标", "insufficient": "样本不足", "record": "仅记录", "missing": "无数据"}
STAGE_CN = {"l1": "L1 专用模型", "l2": "L2 大模型"}


def pct(v) -> str:
    return "—" if v is None else f"{v * 100:.1f}%"


def fr(r: dict | None, ci: bool = True) -> str:
    """格式化比例：92.0%（46/50，CI 81.2–97.1%）"""
    if not r or r.get("value") is None:
        return "—"
    s = f"{pct(r['value'])}（{r['k']}/{r['n']}"
    if ci and r.get("ci_low") is not None:
        s += f"，CI {r['ci_low'] * 100:.0f}–{r['ci_high'] * 100:.0f}%"
    return s + "）"


def _fmt_value(row: dict) -> str:
    v = row.get("value")
    if v is None:
        return "—"
    if _is_ratio(row["metric"]):
        s = pct(v)
        if row.get("ci_low") is not None:
            s += f"（CI {row['ci_low'] * 100:.0f}–{row['ci_high'] * 100:.0f}%）"
        return s
    return f"{v}"


def _is_ratio(metric: str) -> bool:
    last = metric.rsplit(".", 1)[-1]
    return not (last.endswith("_s") or "_ms" in last or "kappa" in last or "mean" in last)


def _target(row: dict) -> str:
    if row.get("op") == "record":
        return "记录基线"
    t = row.get("target")
    if isinstance(t, (int, float)) and _is_ratio(row["metric"]):
        t = pct(t)
    return f"{row.get('op')} {t}"


def acceptance_table(rows: list[dict]) -> str:
    lines = ["| 项目 | 指标 | 目标 | 实测 | 样本数 | 结论 |", "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r.get('name') or r['id']} | `{r['metric']}` | {_target(r)} | {_fmt_value(r)} | "
                     f"{r.get('n') if r.get('n') is not None else '—'} | {STATUS_CN[r['status']]} |")
    return "\n".join(lines)


def check_section(check: str, block: dict) -> str:
    out = [f"### {CHECK_CN[check]}（{check}）"]
    for level in ("session", "image"):
        b = block.get(level)
        if not b:
            continue
        out.append(f"\n**{'按会话' if level == 'session' else '按图片'}**：共 {b['n']} 个，已标注 {b['n_labeled']} 个"
                   f"（有问题 {b['n_pos']}，正常 {b['n_neg']}）\n")
        out.append("| 来源 | 召回率（检出） | 误报率 | 精确率 | 有结论的比例 |")
        out.append("| --- | --- | --- | --- | --- |")
        for k in ("l1", "l2"):
            s = b.get(k)
            if s:
                out.append(f"| {STAGE_CN[k]} | {fr(s['recall'])} | {fr(s['fpr'])} | {fr(s['precision'], False)} | {fr(s['coverage'], False)} |")
        sy = b["system"]
        out.append(f"| **仲裁后（系统）** | {fr(sy['recall'])} | {fr(sy['false_alarm_rate'])} | — | — |")
        out.append("")
        dec = "、".join(f"{DECISION_CN.get(k, k)} {v}" for k, v in sorted(sy["decisions"].items()))
        out.append(f"- 漏检率（有问题却自动通过）：{fr(sy['miss_rate'])}")
        out.append(f"- 自动通过率 {fr(sy['auto_pass_rate'], False)}；需人工（待确认 + 复核）{fr(sy['manual_rate'], False)}；"
                   f"首次尝试会要求重做 {fr(sy['redo_rate_first_attempt'], False)}")
        out.append(f"- 结论分布：{dec or '—'}")
        ag = b.get("agreement")
        if ag:
            out.append(f"- L1 / L2 一致率 {fr(ag['rate'], False)}，Cohen's κ = {ag['kappa'] + 0.0 if ag['kappa'] is not None else '—'}")
        if b.get("misses"):
            out.append(f"- 漏检样本：{', '.join(b['misses'][:20])}{' …' if len(b['misses']) > 20 else ''}")
        if b.get("false_alarms"):
            out.append(f"- 误报样本：{', '.join(b['false_alarms'][:20])}{' …' if len(b['false_alarms']) > 20 else ''}")
        for key, g in (b.get("fairness") or {}).items():
            cells = "；".join(f"{v}: 召回 {fr(s['recall'], False)} / 误报 {fr(s['false_alarm_rate'], False)}" for v, s in g["groups"].items())
            gap = f"召回差 {pct(g['recall_gap'])}，误报差 {pct(g['false_alarm_gap'])}" if g["recall_gap"] is not None or g["false_alarm_gap"] is not None else f"各组样本不足 {g['min_n']}，不计算差值"
            out.append(f"- 分组 `{key}`：{cells}（{gap}）")
    return "\n".join(out)


def render(metrics: dict, acceptance_rows: list[dict] | None, run_info: dict, dataset: str) -> str:
    s = metrics["summary"]
    pipe = run_info.get("pipeline")
    lines = [
        f"# 评测报告：{run_info.get('name', '')}",
        "",
        f"- 数据集：`{dataset}`（{s['samples']} 张图，{s['sessions']} 个会话，已预测 {s['predicted']} 张）",
        f"- 流水线：`{pipe.get('name') if isinstance(pipe, dict) else pipe}`",
        f"- 时间：{run_info.get('created_at', '')}；代码版本：{run_info.get('git', '—')}；提示词版本：{run_info.get('prompt_version', '—')}",
        "",
        "> 括号里的 CI 是 95% Wilson 置信区间。样本少时区间很宽，判断是否达标要看区间而不只看点估计。",
        "> “仲裁后（系统）”的召回率把“待人工确认 / 人工复核 / 重做”都算作检出，因为这些情况都不会被自动放行。",
        "> record_only 的外观项也按判定模式计算，用来积累基线；对外结论仍是“未评估”。",
        "",
    ]
    if acceptance_rows:
        counts = {k: sum(r["status"] == k for r in acceptance_rows) for k in STATUS_CN}
        lines += ["## 验收标准对照", "",
                  "、".join(f"{STATUS_CN[k]} {v} 项" for k, v in counts.items() if v), "", acceptance_table(acceptance_rows), ""]
    lines += ["## 各检测项", ""]
    for c, block in metrics["checks"].items():
        lines += [check_section(c, block), ""]
    fl = metrics.get("finger_level")
    if fl:
        lines += ["## 逐指定位", "", "| 来源 | 逐指召回 | 逐指误报 |", "| --- | --- | --- |"]
        for k, v in fl.items():
            lines.append(f"| {STAGE_CN[k]} | {fr(v['recall'])} | {fr(v['fpr'])} |")
        lines.append("")
    if metrics.get("latency"):
        lines += ["## 耗时（每张图）", "", "| 环节 | p50 | p95 | 平均 | 次数 |", "| --- | --- | --- | --- | --- |"]
        for k, v in metrics["latency"].items():
            lines.append(f"| {k} | {v['p50_ms']} ms | {v['p95_ms']} ms | {v['mean_ms']} ms | {v['n']} |")
        lines += ["", "离线评测的耗时受本机硬件影响，容量结论以 `limbcheck bench` / `bench-vlm` 在目标服务器上的结果为准。", ""]
    ex, lv = metrics.get("sessions"), metrics.get("liveness")
    if ex:
        lines += ["## 体验与端侧性能（网页 demo 会话）", "",
                  f"- 会话数 {ex['n']}；完成率 {fr(ex['completion_rate'])}；首次即完成（无超时步骤）{fr(ex['first_pass_rate'])}",
                  f"- 时长中位数 {ex['duration_median_s']} 秒，P90 {ex['duration_p90_s']} 秒；平均超时步骤 {ex['timeout_steps_mean']}",
                  f"- 端侧推理 P95：姿态 {ex['device_pose_ms_p95']} ms，手部 {ex['device_hand_ms_p95']} ms；L2 返回 P95 {ex['l2_demo_p95_ms']} ms", ""]
    if lv:
        lines += ["## 防作弊（随机手势挑战）", ""]
        if "attack_interception" in lv:
            lines.append(f"- 攻击会话被拦截（挑战未通过）：{fr(lv['attack_interception'])}")
            for a, v in lv.get("attack", {}).items():
                lines.append(f"  - {a}：{fr(v['interception'])}")
        if "genuine_false_reject" in lv:
            lines.append(f"- 真人会话被误拦：{fr(lv['genuine_false_reject'])}")
        lines.append("")
    return "\n".join(lines)


def compare(runs: list[tuple[str, dict]]) -> str:
    """多次运行横向对比（如 MediaPipe vs RTMW、8B vs 32B）。"""
    checks = sorted({c for _, m in runs for c in m["checks"]}, key=list(CHECK_CN).index)
    lines = ["# 运行对比", ""]
    for c in checks:
        lines += [f"## {CHECK_CN[c]}（按会话）", "",
                  "| 运行 | L1 召回 | L1 误报 | L2 召回 | L2 误报 | 系统召回 | 系统误报 | 需人工 | 一致率 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for name, m in runs:
            b = (m["checks"].get(c) or {}).get("session")
            if not b:
                continue
            l1, l2, sy, ag = b.get("l1") or {}, b.get("l2") or {}, b["system"], b.get("agreement") or {}
            lines.append(f"| {name} | {fr(l1.get('recall'))} | {fr(l1.get('fpr'))} | {fr(l2.get('recall'))} | {fr(l2.get('fpr'))} | "
                         f"{fr(sy['recall'])} | {fr(sy['false_alarm_rate'])} | {fr(sy['manual_rate'], False)} | {fr(ag.get('rate'), False)} |")
        lines.append("")
    lines += ["## 耗时 p95（ms）", "", "| 运行 | 关键点 | 轮廓 | 分割 | L2 |", "| --- | --- | --- | --- | --- |"]
    for name, m in runs:
        lt = m.get("latency", {})
        g = lambda k: (lt.get(k) or {}).get("p95_ms", "—")
        lines.append(f"| {name} | {g('keypoints')} | {g('contour')} | {g('segment')} | {g('l2')} |")
    return "\n".join(lines) + "\n"
