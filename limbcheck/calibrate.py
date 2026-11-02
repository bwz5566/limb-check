"""按严格度校准仲裁阈值（high_conf、auto_fail_conf），模型更换或微调后用。

做法：拿带标注的验证集和一次评测的预测（predictions.jsonl），不再调用模型，
对每一档严格度：纹身 / 疤痕先按该档的录用标准重新判定（marks.rejudge），再在阈值网格上重新仲裁，
统计自动放行、自动判不符合、转人工的比例，以及两类自动判错：
    漏放 wrong_pass = 不合格却被自动"满足"（分母：真值有问题的样本）
    误拦 wrong_fail = 合格却被自动"不符合"（分母：真值正常的样本）
在该档 calibration 约束（configs/strictness.yaml）内选自动化比例最高的一组阈值作为推荐值。
"""
from __future__ import annotations

from .arbiter import arbitrate
from .core import FAIL, PASS, CheckResult, VIEW_CHECKS
from .dataset import Sample, truth_positive
from .marks import POLICY_CHECKS, rejudge
from .strictness import arbitration_cfg, level_def, levels as all_levels

GRID = (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)


def _stats(rows: list[tuple[bool, str]]) -> dict:
    n = len(rows)
    pos = [d for t, d in rows if t]
    neg = [d for t, d in rows if not t]
    auto_pass = sum(d == PASS for _, d in rows)
    auto_fail = sum(d == FAIL for _, d in rows)
    r = lambda k, m: round(k / m, 4) if m else None
    return {"n": n, "auto_pass": r(auto_pass, n), "auto_fail": r(auto_fail, n), "manual": r(n - auto_pass - auto_fail, n),
            "wrong_pass": r(sum(d == PASS for d in pos), len(pos)), "wrong_fail": r(sum(d == FAIL for d in neg), len(neg)),
            "n_pos": len(pos), "n_neg": len(neg)}


def calibrate(samples: list[Sample], preds: dict, arbitration: dict, checks: list[str] | None = None,
              levels: list[str] | None = None, profile: dict | None = None, grid=GRID) -> dict:
    levels = levels or all_levels()
    attempt = arbitration.get("max_redo", 1)
    out: dict = {}
    for s in samples:
        res_all = (preds.get(s.id) or {}).get("results") or {}
        for c, res in res_all.items():
            if checks and c not in checks or c not in VIEW_CHECKS.get(s.view, ()):
                continue
            truth = truth_positive(s, c)
            if truth is None:
                continue
            l1 = CheckResult.from_dict(res.get("l1"))
            params = ((profile or {}).get("checks", {}).get(c) or {}).get("params") or {}
            for lv in levels:
                l2 = rejudge(c, res.get("l2"), s.view, lv, {c: params}) if c in POLICY_CHECKS else CheckResult.from_dict(res.get("l2"))
                base = arbitration_cfg(arbitration, lv)
                for hi in grid:
                    for af in grid:
                        cfg = {**base, "high_conf": hi, "auto_fail_conf": af,
                               "per_check": {k: {kk: vv for kk, vv in v.items() if kk not in ("high_conf", "auto_fail_conf")}
                                             for k, v in (base.get("per_check") or {}).items()}}
                        d = arbitrate(c, l1, l2, cfg, "judge", attempt).shadow
                        out.setdefault(c, {}).setdefault(lv, {}).setdefault((hi, af), []).append((truth, d))
    report: dict = {}
    for c, by_level in out.items():
        report[c] = {}
        for lv, by_cfg in by_level.items():
            lim = level_def(lv).get("calibration") or {}
            table = {f"{hi}/{af}": _stats(rows) for (hi, af), rows in by_cfg.items()}
            ok = [(k, v) for k, v in table.items()
                  if (v["wrong_pass"] or 0) <= lim.get("max_wrong_pass", 1) and (v["wrong_fail"] or 0) <= lim.get("max_wrong_fail", 1)]
            best = max(ok, key=lambda kv: (kv[1]["auto_pass"] + kv[1]["auto_fail"], -float(kv[0].split("/")[1]))) if ok else None
            cur = arbitration_cfg(arbitration, lv)
            now = f"{cur.get('high_conf')}/{cur.get('auto_fail_conf')}"
            report[c][lv] = {"limits": lim, "current": {"thresholds": now, **(table.get(now) or {})},
                             "recommended": {"thresholds": best[0], **best[1]} if best else None, "grid": table}
    return report


def render(report: dict) -> str:
    pct = lambda v: "—" if v is None else f"{v:.1%}"
    lines = ["# 严格度阈值校准", "", "阈值写法：high_conf / auto_fail_conf。漏放 = 不合格被自动满足；误拦 = 合格被自动判不符合。", ""]
    for c, by_level in report.items():
        lines += [f"## {c}", "", "| 严格度 | 约束（漏放 / 误拦） | 方案 | 阈值 | 自动放行 | 自动判不符合 | 转人工 | 漏放 | 误拦 | 样本（有问题 / 正常） |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for lv, r in by_level.items():
            lim = r["limits"]
            for name, row in (("当前", r["current"]), ("推荐", r["recommended"])):
                if not row or "n" not in row:
                    lines.append(f"| {lv} | ≤{pct(lim.get('max_wrong_pass'))} / ≤{pct(lim.get('max_wrong_fail'))} | {name} | 无满足约束的阈值 | | | | | | |")
                    continue
                lines.append(f"| {lv} | ≤{pct(lim.get('max_wrong_pass'))} / ≤{pct(lim.get('max_wrong_fail'))} | {name} | {row['thresholds']} | "
                             f"{pct(row['auto_pass'])} | {pct(row['auto_fail'])} | {pct(row['manual'])} | {pct(row['wrong_pass'])} | "
                             f"{pct(row['wrong_fail'])} | {row['n_pos']} / {row['n_neg']} |")
        lines.append("")
    lines += ["推荐值满足该档约束且自动判定比例最高。样本少时比例波动很大，正式采用前请保证每档每项至少数百个样本，",
              "并把推荐值写回 configs/strictness.yaml 对应档的 arbitration。"]
    return "\n".join(lines)
