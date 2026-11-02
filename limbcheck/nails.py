"""指甲：美甲（甲油、甲片、延长甲）、过长、过尖。标准在 configs/policies/nails.yaml。

- L1（关键点 + 像素）只能看出有颜色的甲油：手背朝镜头时指甲与同一根手指中节皮肤的色差（limbcheck.geometry.nails_delta_e）。
  试过沿手指方向的颜色剖面量"指甲超出指腹的长度"：网络摄像头下指尖边缘模糊约 5 毫米，
  实测者的短指甲被量成过长，所以长度、形状不用像素法。
- L2（大模型）看指尖近景给出款式、长度、形状（提示词 v3 的 nails 规则），这里按标准逐只手判定。
- 没有大模型时，长度、形状交人工看报告里的指甲近景小图。
"""
from __future__ import annotations

import json
from functools import lru_cache

from .core import EXPECTED_HANDS, CheckResult, Finding, config_dir

PASS, REVIEW, BLOCK = "pass", "review", "block"
_RANK = {PASS: 0, REVIEW: 1, BLOCK: 2}

_OVERRIDE: dict | None = None


@lru_cache(maxsize=1)
def _load_default() -> dict:
    path = config_dir() / "policies" / "nails.yaml"
    try:
        import yaml
    except ImportError:   # 网页 demo 只用标准库：读 export-web-config 生成的 JSON
        web = config_dir().parent / "apps" / "web-demo" / "config.generated.json"
        return json.loads(web.read_text(encoding="utf-8"))["policies"]["nails"]
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def nails_policy() -> dict:
    return _OVERRIDE if _OVERRIDE is not None else _load_default()


def set_nails_policy(p: dict | None) -> None:
    """替换当前标准（测试用）；传 None 恢复为 configs/policies/nails.yaml。"""
    global _OVERRIDE
    _OVERRIDE = p


def worst(actions) -> str:
    acts = [a for a in actions if a]
    return max(acts, key=_RANK.__getitem__) if acts else PASS


def judge_hand(style: str, length: str | None, shape: str | None, pol: dict | None = None) -> tuple[str, list[dict]]:
    """一只手的款式 / 长度 / 形状 → (verdict, 命中的条款)。长度、形状没给出时不判。"""
    pol = pol or nails_policy()
    hits = []
    s = (pol.get("styles") or {}).get(style)
    if s:
        hits.append({"item": "style", "value": style, "action": s.get("action", PASS),
                     "text": s.get("text") or s.get("name", style)})
    for key, val in (("length", length), ("shape", shape)):
        sec = pol.get(key) or {}
        if val in (sec.get("actions") or {}):
            act = sec["actions"][val]
            name = (sec.get("names") or {}).get(val, val)
            hits.append({"item": key, "value": val, "action": act,
                         "text": f"指甲{name}" if key == "length" else f"指甲{name}"})
    verdict = worst(h["action"] for h in hits)
    return verdict, hits


def policy_result(items: list[dict], view: str, source: str, quality_ok: bool = True, reason: str = "",
                  pol: dict | None = None) -> CheckResult:
    """L2 的 nails 条目（每只手一条：label 为款式，attrs.length / attrs.shape）按标准判成 CheckResult。
    每条 Finding.value 有 verdict 和命中的条款，仲裁用 policy 策略读取。"""
    pol = pol or nails_policy()
    styles = pol.get("styles") or {}
    if not items:
        return CheckResult("nails", source, None, 0.0, [], quality_ok, [], "模型未返回该项")
    findings, unsure = [], False
    for i in items:
        side = i["part"] or "unknown"
        if i["label"] not in styles or i["uncertain"]:
            unsure = True
            findings.append(Finding("nails", side, i["label"], None, i["confidence"], source, None, "模型表示看不清指甲"))
            continue
        length, shape = (i["attrs"].get("length"), i["attrs"].get("shape"))
        verdict, hits = judge_hand(i["label"], length, shape, pol)
        bad = [h["text"] for h in hits if h["action"] == verdict and verdict != PASS]
        desc = "、".join(filter(None, [styles[i["label"]].get("name"), (pol.get("length", {}).get("names") or {}).get(length),
                                      (pol.get("shape", {}).get("names") or {}).get(shape)]))
        hand = {"right_hand": "右手", "left_hand": "左手"}.get(side, side)
        note = f"{hand}：{'；'.join(bad)}" if bad else f"{hand}：{desc or '自然指甲'}"
        findings.append(Finding("nails", side, i["label"], verdict != PASS, i["confidence"], source,
                                {"kind": "nail_hand", "verdict": verdict, "style": i["label"], "length": length, "shape": shape,
                                 "rules": hits}, note))
    bad = [f for f in findings if f.positive]
    note = "；".join(f.note for f in findings if f.note)
    if bad:
        return CheckResult("nails", source, True, round(max(f.confidence for f in bad), 3), sorted({f.part for f in bad}),
                           quality_ok, findings, note[:400])
    need = EXPECTED_HANDS.get(view, 2)
    if unsure or len({f.part for f in findings}) < need:
        return CheckResult("nails", source, None, 0.0, [], quality_ok, findings, note or f"只返回了 {len(findings)} 只手")
    return CheckResult("nails", source, False, round(min(f.confidence for f in findings), 3), [], quality_ok, findings,
                       note or reason)


def l1_with_verdicts(r: CheckResult | None, pol: dict | None = None) -> CheckResult | None:
    """给 L1 色差法的结果加上按标准的 verdict（疑似甲油 → styles.polish 的动作），与 L2 一起按 policy 策略仲裁。
    色差法只看颜色，长度、形状不判。"""
    if r is None:
        return None
    pol = pol or nails_policy()
    polish = (pol.get("styles") or {}).get("polish") or {}
    for f in r.findings:
        if f.label not in ("polish", "none"):
            continue
        verdict = polish.get("action", BLOCK) if f.label == "polish" else PASS
        if f.label == "polish":
            f.confidence = min(f.confidence, pol.get("l1_max_confidence", 0.8))
        f.value = {"kind": "nail_color", "verdict": verdict, "delta_e": f.value,
                   "rules": [{"item": "style", "value": f.label, "action": verdict, "text": polish.get("text", "疑似美甲")}] if f.label == "polish" else []}
    if r.positive:
        r.confidence = max(f.confidence for f in r.findings if f.positive)
    r.note = "；".join(filter(None, [r.note, "指甲长度、形状由大模型或人工看指甲近景判断"]))
    return r
