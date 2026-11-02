"""肢体正常综合判断：把各检测项的仲裁结论和动作步骤的完成情况汇总成一个总结论（仅标准库，网页 demo 的 server.py 也用）。

汇总方式在 configs/summary.yaml：按模块分组、哪些项目入职前可以整改、总结论的顺序和说明。
- 只汇总判定模式（judge）的检测项；只记录（record_only）的项目列出但不影响总结论。
- 可整改的项目（默认指甲）判"不符合"或"疑似问题"时不淘汰，给"有条件通过"并写明整改要求。
- 动作步骤没完成（超时、手动跳过）→ 该模块需人工复核。
"""
from __future__ import annotations

import json
from functools import lru_cache

from .arbiter import worst
from .core import CHECK_CN, DECISION_CN, FAIL, FAIL_SUGGESTED, NOT_ASSESSED, PASS, REDO, REVIEW, config_dir

CONDITIONAL = "conditional"        # 模块级：只有可整改的问题
_OVERRIDE: dict | None = None


@lru_cache(maxsize=1)
def _load_default() -> dict:
    path = config_dir() / "summary.yaml"
    try:
        import yaml
    except ImportError:   # 网页 demo 只用标准库：读 export-web-config 生成的 JSON
        web = config_dir().parent / "apps" / "web-demo" / "config.generated.json"
        return json.loads(web.read_text(encoding="utf-8"))["summary"]
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def summary_config() -> dict:
    return _OVERRIDE if _OVERRIDE is not None else _load_default()


def set_summary_config(c: dict | None) -> None:
    global _OVERRIDE
    _OVERRIDE = c


def _reason(e: dict) -> str:
    """检测项结论的说明：会话级备注 + 仲裁理由（去掉"只记录"前缀）。"""
    parts = list(e.get("notes") or [])
    r = e.get("reason") or ""
    if r.startswith("只记录"):
        r = r.split("：", 1)[-1]
    if r:
        parts.append(r)
    return "；".join(dict.fromkeys(p for p in parts if p))[:300]


def summarize(by_check: dict, steps: list[dict] | None = None, checks: list[str] | None = None,
              modes: dict | None = None, cfg: dict | None = None) -> dict:
    """by_check：arbitrate_session 的 by_check（每项 {mode, decision, shadow, notes, reason?, views}）；
    steps：动作步骤 [{id, title, status, note}]（status 为 pass / review / timeout ……）；
    checks：本次检测范围内的检测项（None 为全部）；modes：岗位画像里各检测项的模式（by_check 里没有的项目，如活体，按它判断）。

    返回 {outcome, name, text, conditions, manual, failed, groups}。"""
    cfg = cfg or summary_config()
    fixable = cfg.get("fixable") or {}
    steps = steps or []
    by_step = {s.get("id"): s for s in steps}
    groups, conditions, manual, failed, redo = [], [], [], [], []
    for g in cfg.get("groups") or []:
        items, judged = [], []
        for c in g.get("checks") or []:
            e = by_check.get(c)
            if e is None or (checks is not None and c not in checks):
                continue
            mode, d = e.get("mode", "judge"), e.get("decision")
            item = {"check": c, "name": CHECK_CN.get(c, c), "mode": mode, "decision": d, "shadow": e.get("shadow"),
                    "reason": f"{CHECK_CN.get(c, c)}：{_reason(e) or DECISION_CN.get(d, d or '')}"}
            items.append(item)
            if mode != "judge" or not d or d == NOT_ASSESSED:
                continue
            if c in fixable and d in (FAIL, FAIL_SUGGESTED):
                item["decision_shown"] = CONDITIONAL
                conditions.append({"group": g["name"], "check": c, "text": fixable[c], "reason": item["reason"],
                                   "confirmed": d == FAIL})
                judged.append(CONDITIONAL)
                continue
            judged.append(d)
            if d == FAIL:
                failed.append({"group": g["name"], "check": c, "reason": item["reason"]})
            elif d in (FAIL_SUGGESTED, REVIEW):
                manual.append({"group": g["name"], "check": c, "decision": d, "reason": item["reason"]})
            elif d == REDO:
                redo.append({"group": g["name"], "check": c, "reason": item["reason"]})
        # 动作步骤：只在模块里有判定模式的检测项（在检测范围内）时计入
        scoped = [c for c in g.get("checks") or [] if checks is None or c in checks]
        if any((by_check.get(c) or {}).get("mode", (modes or {}).get(c)) == "judge" for c in scoped):
            done = [by_step[s] for s in g.get("steps") or [] if s in by_step]
            for st in done:
                ok = st.get("status") == "pass"
                name = st.get("title") or st.get("id")
                items.append({"check": None, "step": st.get("id"), "name": name, "mode": "judge", "decision": PASS if ok else REVIEW,
                              "reason": "按要求完成" if ok else (st.get("note") or st.get("status") or "")[:300]})
                judged.append(PASS if ok else REVIEW)
                if not ok:
                    manual.append({"group": g["name"], "check": None, "decision": REVIEW,
                                   "reason": f"{name}未完成：{st.get('note') or st.get('status')}"[:300]})
        if not items:
            continue
        real = [d for d in judged if d != CONDITIONAL]
        decision = worst(real) if real else (CONDITIONAL if judged else None)
        if decision in (None, PASS) and CONDITIONAL in judged:
            decision = CONDITIONAL
        groups.append({"id": g["id"], "name": g["name"], "decision": decision or NOT_ASSESSED,
                       "record_only": not judged, "items": items})

    if failed:
        outcome = "fail"
    elif manual:
        outcome = "manual"
    elif redo:
        outcome = "redo"
    elif conditions:
        outcome = "conditional"
    elif any(g["decision"] == PASS for g in groups):
        outcome = "pass"
    else:
        outcome = "none"
    o = (cfg.get("outcomes") or {}).get(outcome) or {}
    return {"outcome": outcome, "name": o.get("name", outcome), "text": o.get("text", ""),
            "conditions": conditions, "manual": manual, "failed": failed, "redo": redo, "groups": groups}

