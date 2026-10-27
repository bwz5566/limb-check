"""指标计算。定义见 docs/METRICS.md。

所有比例都附 95% Wilson 置信区间：样本少时区间很宽，这正是 M0 结论需要谨慎的原因。
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field

from .arbiter import worst
from .dataset import Sample, expected_sides, truth_finger_parts, truth_positive
from .core import CHECKS, FAIL, FAIL_SUGGESTED, FINGERS, IMAGE_CHECKS, PASS, REDO, REVIEW, VIEW_CHECKS

FLAGGED = (FAIL, FAIL_SUGGESTED, REVIEW, REDO)
GROUP_EXCLUDE = {"note", "side", "job", "source_image", "subject_id"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round(max(0.0, (c - r) / d), 4), round(min(1.0, (c + r) / d), 4)


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"value": round(k / n, 4) if n else None, "k": k, "n": n, "ci_low": lo, "ci_high": hi}


def percentile(values: list[float], p: float) -> float | None:
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    if len(v) == 1:
        return round(v[0], 1)
    pos = (len(v) - 1) * p / 100
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return round(v[lo] + (v[hi] - v[lo]) * (pos - lo), 1)


@dataclass
class Unit:
    id: str
    truth: bool | None
    l1: bool | None
    l2: bool | None
    decision: str | None            # shadow 结论（按判定模式）
    first: str | None               # 首次尝试的 shadow 结论
    agree: bool | None
    meta: dict = field(default_factory=dict)
    l1_parts: set = field(default_factory=set)
    l2_parts: set = field(default_factory=set)
    has_l1: bool = False
    has_l2: bool = False


def _pos(r: dict | None) -> bool | None:
    return None if r is None else r.get("positive")


def image_units(samples: list[Sample], preds: dict[str, dict], check: str) -> list[Unit]:
    out = []
    for s in samples:
        if check not in VIEW_CHECKS.get(s.view, ()):
            continue
        res = (preds.get(s.id) or {}).get("results", {}).get(check)
        if not res:
            continue
        l1, l2 = res.get("l1"), res.get("l2")
        out.append(Unit(
            id=s.id, truth=truth_positive(s, check), l1=_pos(l1), l2=_pos(l2),
            decision=res.get("shadow") or res.get("decision"), first=res.get("shadow_first_attempt"),
            agree=res.get("agree"), meta=s.meta, has_l1=l1 is not None, has_l2=l2 is not None,
            l1_parts=set((l1 or {}).get("parts") or []), l2_parts=set((l2 or {}).get("parts") or [])))
    return out


def _any(vals: list[bool | None]) -> bool | None:
    if any(v is True for v in vals):
        return True
    if vals and all(v is False for v in vals):
        return False
    return None


def session_units(samples: list[Sample], units: list[Unit]) -> list[Unit]:
    by_id = {s.id: s for s in samples}
    groups: dict[str, list[Unit]] = defaultdict(list)
    for u in units:
        groups[by_id[u.id].session_id].append(u)
    out = []
    for sid, us in groups.items():
        out.append(Unit(
            id=sid, truth=_any([u.truth for u in us]), l1=_any([u.l1 for u in us if u.has_l1]),
            l2=_any([u.l2 for u in us if u.has_l2]), decision=worst([u.decision for u in us]),
            first=worst([u.first for u in us]),
            agree=False if any(u.agree is False for u in us) else (True if all(u.agree for u in us) else None),
            meta=us[0].meta, has_l1=any(u.has_l1 for u in us), has_l2=any(u.has_l2 for u in us),
            l1_parts=set().union(*[u.l1_parts for u in us]), l2_parts=set().union(*[u.l2_parts for u in us])))
    return out


def model_stats(units: list[Unit], key: str) -> dict | None:
    present = [u for u in units if getattr(u, f"has_{key}")]
    if not present:
        return None
    pos = [u for u in units if u.truth is True]
    neg = [u for u in units if u.truth is False]
    tp = sum(getattr(u, key) is True for u in pos)
    fp = sum(getattr(u, key) is True for u in neg)
    tn = sum(getattr(u, key) is False for u in neg)
    labeled = pos + neg
    return {
        "recall": rate(tp, len(pos)),          # 无结果的正样本按漏检计（保守）
        "fpr": rate(fp, len(neg)),
        "specificity": rate(tn, len(neg)),
        "precision": rate(tp, tp + fp),
        "coverage": rate(sum(getattr(u, key) is not None for u in labeled), len(labeled)),
    }


def system_stats(units: list[Unit]) -> dict:
    us = [u for u in units if u.decision is not None and u.truth is not None]
    pos = [u for u in us if u.truth]
    neg = [u for u in us if not u.truth]
    flagged = lambda u: u.decision in FLAGGED
    counts = defaultdict(int)
    for u in us:
        counts[u.decision] += 1
    return {
        "recall": rate(sum(flagged(u) for u in pos), len(pos)),
        "miss_rate": rate(sum(u.decision == PASS for u in pos), len(pos)),
        "false_alarm_rate": rate(sum(flagged(u) for u in neg), len(neg)),
        "accuracy": rate(sum(flagged(u) for u in pos) + sum(u.decision == PASS for u in neg), len(us)),
        "auto_pass_rate": rate(counts[PASS], len(us)),
        "auto_fail_rate": rate(counts[FAIL], len(us)),
        "wrong_auto_fail_rate": rate(sum(u.decision == FAIL for u in neg), len(neg)),   # 合格却被自动判不符合（最需控制）
        "manual_rate": rate(counts[FAIL_SUGGESTED] + counts[REVIEW], len(us)),
        "redo_rate_first_attempt": rate(sum(u.first == REDO for u in us), len(us)),
        "decisions": dict(counts),
    }


def agreement_stats(units: list[Unit]) -> dict | None:
    both = [u for u in units if u.l1 is not None and u.l2 is not None]
    if not both:
        return None
    n = len(both)
    agree = sum(u.l1 == u.l2 for u in both)
    p1 = sum(u.l1 for u in both) / n
    p2 = sum(u.l2 for u in both) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    po = agree / n
    kappa = round((po - pe) / (1 - pe), 4) if pe < 1 else None
    return {"rate": rate(agree, n), "kappa": kappa}


def group_stats(units: list[Unit], min_n: int = 10) -> dict:
    keys = sorted({k for u in units for k, v in u.meta.items() if k not in GROUP_EXCLUDE and isinstance(v, (str, int, bool))})
    out = {}
    for k in keys:
        vals = defaultdict(list)
        for u in units:
            if k in u.meta:
                vals[str(u.meta[k])].append(u)
        if len(vals) < 2 or len(vals) > 12:
            continue
        g = {v: system_stats(us) for v, us in vals.items()}
        rec = [s["recall"]["value"] for s in g.values() if s["recall"]["n"] >= min_n and s["recall"]["value"] is not None]
        fa = [s["false_alarm_rate"]["value"] for s in g.values() if s["false_alarm_rate"]["n"] >= min_n and s["false_alarm_rate"]["value"] is not None]
        out[k] = {
            "groups": {v: {"recall": s["recall"], "false_alarm_rate": s["false_alarm_rate"]} for v, s in g.items()},
            "recall_gap": round(max(rec) - min(rec), 4) if len(rec) >= 2 else None,
            "false_alarm_gap": round(max(fa) - min(fa), 4) if len(fa) >= 2 else None,
            "min_n": min_n,
        }
    return out


def finger_level(samples: list[Sample], preds: dict[str, dict]) -> dict:
    """逐指统计（定位准不准）：每只手每根手指一个单位。"""
    rows = {"l1": [], "l2": []}
    for s in samples:
        if "fingers" not in VIEW_CHECKS.get(s.view, ()):
            continue
        res = (preds.get(s.id) or {}).get("results", {}).get("fingers")
        truth = truth_finger_parts(s)
        if not res or truth is None:
            continue
        for key in ("l1", "l2"):
            r = res.get(key)
            if r is None or r.get("positive") is None:
                continue
            parts = set(r.get("parts") or [])
            hands = {p.split(".")[0] for p in parts if "." not in p}   # L2 只报到"某只手"时视为整只手都报了
            for side in expected_sides(s):
                for f in FINGERS:
                    part = f"{side}_hand.{f}"
                    rows[key].append((part in truth, part in parts or f"{side}_hand" in hands))
    out = {}
    for key, rs in rows.items():
        if not rs:
            continue
        pos = [p for t, p in rs if t]
        neg = [p for t, p in rs if not t]
        out[key] = {"recall": rate(sum(pos), len(pos)), "fpr": rate(sum(neg), len(neg))}
    return out


def latency_stats(preds: dict[str, dict]) -> dict:
    vals = defaultdict(list)
    for p in preds.values():
        for k, v in (p.get("latency_ms") or {}).items():
            if v is not None:
                vals[k].append(float(v))
    return {k: {"p50_ms": percentile(v, 50), "p95_ms": percentile(v, 95), "mean_ms": round(statistics.fmean(v), 1), "n": len(v)}
            for k, v in vals.items()}


def session_stats(sessions: list[dict]) -> tuple[dict, dict]:
    """体验与防作弊指标（来自网页 demo 导出的会话记录）。"""
    if not sessions:
        return {}, {}
    n = len(sessions)
    complete = [s for s in sessions if s.get("steps") and all(st.get("status") != "timeout" for st in s["steps"])]
    durations = [s["duration_ms"] / 1000 for s in sessions if s.get("duration_ms")]
    timeouts = [sum(st.get("status") == "timeout" for st in s.get("steps") or []) for s in sessions]
    pose_ms = [s["perf"]["pose_ms"]["p95"] for s in sessions if (s.get("perf") or {}).get("pose_ms", {}).get("p95") is not None]
    hand_ms = [s["perf"]["hand_ms"]["p95"] for s in sessions if (s.get("perf") or {}).get("hand_ms", {}).get("p95") is not None]
    l2 = [v for s in sessions for v in s.get("l2_latency_ms") or []]
    exp = {
        "n": n,
        "completion_rate": rate(len(complete), n),
        "duration_median_s": percentile(durations, 50),
        "duration_p90_s": percentile(durations, 90),
        "timeout_steps_mean": round(statistics.fmean(timeouts), 2) if timeouts else None,
        "first_pass_rate": rate(sum(t == 0 for t in timeouts), n),
        "device_pose_ms_p95": percentile(pose_ms, 95),
        "device_hand_ms_p95": percentile(hand_ms, 95),
        "l2_demo_p95_ms": percentile(l2, 95),
    }
    attacks, genuine = defaultdict(list), []
    for s in sessions:
        ch = (s.get("challenge") or {}).get("status")
        if ch is None:
            continue
        a = (s.get("truth") or {}).get("attack")
        if a in (None, ""):
            continue
        (genuine if a == "none" else attacks[a]).append(ch != "pass")
    live = {}
    all_att = [x for v in attacks.values() for x in v]
    if all_att:
        live["attack_interception"] = rate(sum(all_att), len(all_att))
        live["attack"] = {a: {"interception": rate(sum(v), len(v))} for a, v in attacks.items()}
    if genuine:
        live["genuine_false_reject"] = rate(sum(genuine), len(genuine))
    return exp, live


def compute(samples: list[Sample], preds: dict[str, dict], sessions: list[dict] | None = None, min_group_n: int = 10) -> dict:
    checks = {}
    for c in IMAGE_CHECKS:
        iu = image_units(samples, preds, c)
        if not iu:
            continue
        su = session_units(samples, iu)
        block = {}
        for level, us in (("image", iu), ("session", su)):
            labeled = [u for u in us if u.truth is not None]
            block[level] = {
                "n": len(us), "n_labeled": len(labeled),
                "n_pos": sum(u.truth is True for u in us), "n_neg": sum(u.truth is False for u in us),
                "l1": model_stats(us, "l1"), "l2": model_stats(us, "l2"),
                "system": system_stats(us), "agreement": agreement_stats(us),
                "fairness": group_stats(us, min_group_n),
                "misses": sorted(u.id for u in us if u.truth is True and u.decision == PASS)[:50],
                "false_alarms": sorted(u.id for u in us if u.truth is False and u.decision in FLAGGED)[:50],
            }
        checks[c] = block
    exp, live = session_stats(sessions or [])
    return {
        "summary": {"samples": len(samples), "predicted": sum(1 for s in samples if s.id in preds),
                    "sessions": len({s.session_id for s in samples})},
        "checks": checks,
        "finger_level": finger_level(samples, preds),
        "latency": latency_stats({k: v for k, v in preds.items()}),
        "sessions": exp,
        "liveness": live,
    }


def resolve(metrics: dict, path: str):
    parts = path.split(".")
    node = metrics["checks"] if parts[0] in metrics.get("checks", {}) and parts[0] in CHECKS else metrics
    for p in parts:
        if not isinstance(node, dict) or p not in node:
            return None
        node = node[p]
    return node


def evaluate_acceptance(metrics: dict, acceptance: dict) -> list[dict]:
    ops = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, "<": lambda a, b: a < b}
    out = []
    for c in acceptance.get("criteria", []):
        node = resolve(metrics, c["metric"])
        row = {k: c.get(k) for k in ("id", "name", "metric", "op", "target", "use", "min_n")}
        if node is None:
            out.append({**row, "status": "missing", "value": None, "n": None})
            continue
        if isinstance(node, dict):
            value = node.get(c.get("use") or "value")
            n, ci = node.get("n"), (node.get("ci_low"), node.get("ci_high"))
        else:
            value, n, ci = node, None, (None, None)
        row.update({"value": value, "n": n, "ci_low": ci[0], "ci_high": ci[1]})
        if c.get("op") == "record":
            row["status"] = "record"
        elif value is None:
            row["status"] = "missing"
        elif c.get("min_n") and n is not None and n < c["min_n"]:
            row["status"] = "insufficient"
        else:
            row["status"] = "pass" if ops[c["op"]](value, c["target"]) else "fail"
        out.append(row)
    return out
