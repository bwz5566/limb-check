"""指甲标准（configs/policies/nails.yaml，limbcheck/nails.py）：大模型的款式 / 长度 / 形状 → 标准判定 → 仲裁。"""
import json

from limbcheck.arbiter import arbitrate
from limbcheck.config import load_thresholds
from limbcheck.core import FAIL, FAIL_SUGGESTED, PASS, REVIEW, CheckResult, Finding
from limbcheck.nails import judge_hand, l1_with_verdicts
from limbcheck.prompts import build_user_prompt, l2_checks_for, parse_reply, prompt_version, to_check_results

ARB = load_thresholds()["arbitration"]


def hand(side, label="none", length="short", shape="round", conf=0.9, uncertain=False):
    return {"check": "nails", "part": f"{side}_hand", "label": label, "confidence": conf, "uncertain": uncertain,
            "attrs": {"length": length, "shape": shape}}


def l2(*items, view="nails_close"):
    text = json.dumps({"items": list(items), "image_quality": "good", "reason": "测试"}, ensure_ascii=False)
    parsed, errs = parse_reply(text, ["nails"])
    assert not errs, errs
    return to_check_results(parsed, ["nails"], view, "L2:test")["nails"]


def test_judge_hand():
    assert judge_hand("none", "short", "round")[0] == "pass"
    assert judge_hand("none", "medium", "oval")[0] == "pass"          # 1–3 毫米很常见，不拦截
    assert judge_hand("none", "long", "round")[0] == "block"
    assert judge_hand("none", "short", "pointed")[0] == "block"
    v, hits = judge_hand("polish", "short", "round")
    assert v == "block" and any("入职前需卸除" in h["text"] for h in hits)
    assert judge_hand("none", None, None)[0] == "pass"                 # 没给长度、形状时不判


def test_prompt_v3_has_length_and_shape():
    assert prompt_version() == "v3"
    p = build_user_prompt("nails_close", l2_checks_for("nails_close"))
    assert "length" in p and "pointed" in p and "指甲近景" in p
    assert build_user_prompt("head_front", ["hair"])           # 头部近景视角有描述


def test_natural_nails_pass():
    r = l2(hand("right"), hand("left"))
    a = arbitrate("nails", None, r, ARB, "judge", 1)
    assert r.positive is False and a.decision == PASS and "右手" in a.reason


def test_long_or_pointed_nails_blocked():
    r = l2(hand("right", length="long"), hand("left"))
    assert r.positive is True and r.parts == ["right_hand"]
    a = arbitrate("nails", None, r, ARB, "judge", 1)
    assert a.decision == FAIL and "过长" in a.reason                   # 大模型置信度 0.9 ≥ 0.85 → 直接判定
    low = arbitrate("nails", None, l2(hand("right", shape="pointed", conf=0.7), hand("left")), ARB, "judge", 1)
    assert low.decision == FAIL_SUGGESTED and "尖形" in low.reason


def test_polish_and_extension():
    r = l2(hand("right", "polish"), hand("left", "extension", length="long"))
    assert r.positive is True and set(r.parts) == {"right_hand", "left_hand"}
    assert "入职前需卸除" in r.note


def test_unsure_or_missing_hand_is_unknown():
    assert l2(hand("right"), hand("left", uncertain=True)).positive is None
    assert l2(hand("right")).positive is None                         # 只返回一只手


def test_record_only_job():
    a = arbitrate("nails", None, l2(hand("right", "polish"), hand("left")), ARB, "record_only", 1)
    assert a.decision == "not_assessed" and a.shadow in (FAIL, FAIL_SUGGESTED)


def test_l1_polish_capped_to_human_confirmation():
    """色差法单独发现甲油：置信度上限 0.8，交人工确认（不直接判不符合）。"""
    f = Finding("nails", "right_hand", "polish", True, 0.95, "L1:deltaE", {"index": 40.0}, "疑似美甲：食指")
    ok = Finding("nails", "left_hand", "none", False, 0.9, "L1:deltaE", {"index": 5.0})
    r = l1_with_verdicts(CheckResult("nails", "L1:deltaE", True, 0.95, ["right_hand"], True, [f, ok]))
    assert r.confidence <= 0.8 and r.findings[0].value["delta_e"] == {"index": 40.0}
    a = arbitrate("nails", r, None, ARB, "judge", 1)
    assert a.decision == FAIL_SUGGESTED
    clean = l1_with_verdicts(CheckResult("nails", "L1:deltaE", False, 0.9, [], True, [ok]))
    assert arbitrate("nails", clean, None, ARB, "judge", 1).decision == PASS
    assert l1_with_verdicts(None) is None


def test_l1_and_l2_combined():
    """L1 没看出颜色（透明甲油、裸色甲片），L2 看出延长甲 → 按 L2 拦截。"""
    ok = [Finding("nails", s, "none", False, 0.9, "L1:deltaE", {}) for s in ("right_hand", "left_hand")]
    l1 = l1_with_verdicts(CheckResult("nails", "L1:deltaE", False, 0.9, [], True, ok))
    a = arbitrate("nails", l1, l2(hand("right", "extension", "long"), hand("left")), ARB, "judge", 1)
    assert a.decision in (FAIL, FAIL_SUGGESTED) and "延长甲" in a.reason
    assert arbitrate("nails", l1, None, ARB, "judge", 1).decision in (PASS, REVIEW)
