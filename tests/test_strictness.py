"""筛选严格度（严格 / 标准 / 宽松）与"高置信度直接判定、低置信度转人工"。"""
import json

from limbcheck.arbiter import arbitrate
from limbcheck.calibrate import calibrate, render
from limbcheck.config import load_job_profiles, load_thresholds
from limbcheck.core import CheckResult
from limbcheck.dataset import Sample
from limbcheck.marks import rejudge
from limbcheck.prompts import parse_reply, to_check_results
from limbcheck.strictness import arbitration_cfg, levels

ARB = load_thresholds()["arbitration"]
META = {"img_w": 1000, "img_h": 1000, "mm_per_px": 0.5, "scale_source": "测试"}


def tattoo(part, conf=0.95, bbox=(0, 0, 40, 40), **attrs):
    return {"check": "tattoo", "part": part, "label": "tattoo", "confidence": conf, "bbox": list(bbox),
            "attrs": {"content": "benign", "ink": "light", **attrs}}


def decide(items, level, view="backs"):
    text = json.dumps({"items": items, "image_quality": "good"}, ensure_ascii=False)
    parsed, _ = parse_reply(text, ["tattoo"])
    r = to_check_results(parsed, ["tattoo"], view, "L2:test", [], {**META, "strictness": level})["tattoo"]
    return arbitrate("tattoo", None, r, arbitration_cfg(ARB, level), "judge", 1), r


def test_levels_change_policy():
    hb = [tattoo("right_hand_back")]          # 手背单处、小、浅色、无攻击性
    assert decide(hb, "strict")[0].decision == "fail"
    assert decide(hb, "standard")[0].decision == "review"
    assert decide(hb, "lenient")[0].decision == "pass"
    near = [tattoo("left_forearm", bbox=(0, 0, 171, 108))]    # 面积 ≈ 身份证
    assert decide(near, "strict", "arms_out")[0].decision == "fail"
    assert decide(near, "standard", "arms_out")[0].decision == "review"
    assert decide(near, "lenient", "arms_out")[0].decision == "pass"


def test_aggressive_content_blocked_at_every_level():
    for lv in levels():
        a, _ = decide([tattoo("left_forearm", content="beast")], lv, "arms_out")
        assert a.decision == "fail", lv


def test_low_confidence_block_goes_to_human():
    item = [tattoo("neck", conf=0.8)]
    assert decide(item, "strict", "neck")[0].decision == "fail"               # 0.8 ≥ 严格档 0.75
    a, _ = decide(item, "standard", "neck")
    assert a.decision == "fail_suggested" and "不足以直接判定" in a.reason       # 0.8 < 标准档 0.85
    assert decide(item, "lenient", "neck")[0].decision == "fail_suggested"


def test_default_strategy_auto_fail_needs_two_sources():
    l1 = CheckResult("fingers", "L1:x", True, 0.9, ["left_hand.ring"])
    l2 = CheckResult("fingers", "L2:x", True, 0.92, ["left_hand"])
    cfg = arbitration_cfg(ARB, "standard")
    assert arbitrate("fingers", l1, l2, cfg, "judge", 1).decision == "fail"
    assert arbitrate("fingers", l1, None, cfg, "judge", 1).decision == "fail_suggested"   # 关键点单路会"脑补"，不直接判
    low = CheckResult("fingers", "L2:x", True, 0.7, ["left_hand"])
    assert arbitrate("fingers", l1, low, cfg, "judge", 1).decision == "fail_suggested"
    assert arbitrate("fingers", l1, l2, {**cfg, "auto_fail_conf": None}, "judge", 1).decision == "fail_suggested"


def test_confident_normal_passes_directly():
    ok = CheckResult("fingers", "L2:x", False, 0.75)
    assert arbitrate("fingers", None, ok, arbitration_cfg(ARB, "lenient"), "judge", 1).decision == "pass"     # 0.75 ≥ 0.7
    assert arbitrate("fingers", None, ok, arbitration_cfg(ARB, "standard"), "judge", 1).decision == "review"  # 0.75 < 0.8


def test_level_keeps_stricter_per_check_pass_threshold():
    cfg = arbitration_cfg(ARB, "lenient")
    assert cfg["per_check"]["scar"]["high_conf"] == 0.9 and cfg["per_check"]["scar"]["strategy"] == "policy"
    assert cfg["high_conf"] == 0.7 and cfg["auto_fail_conf"] == 0.92


def test_profiles_have_strictness():
    for p in load_job_profiles().values():
        assert p["strictness"] in levels()


def test_rejudge_without_calling_model():
    _, r = decide([tattoo("right_hand_back")], "standard")
    d = json.loads(json.dumps(r.to_dict()))      # 存进 predictions.jsonl 再读出来
    for lv, want in (("strict", "block"), ("standard", "review"), ("lenient", "pass")):
        rr = rejudge("tattoo", d, "backs", lv)
        assert [f.value["verdict"] for f in rr.findings] == [want], lv


def test_calibrate_report():
    samples, preds = [], {}
    for i, (part, truth) in enumerate([("right_hand_back", "review"), ("neck", "block"), ("left_forearm", "pass")] * 3):
        _, r = decide([tattoo(part)], "standard", "backs" if "hand" in part else "neck" if part == "neck" else "arms_out")
        view = "backs" if "hand" in part else "neck" if part == "neck" else "arms_out"
        samples.append(Sample(id=f"s{i}", image="x.jpg", view=view, session_id=f"S{i}", truth={"tattoo": truth}))
        preds[f"s{i}"] = {"results": {"tattoo": {"l1": None, "l2": r.to_dict()}}}
    rep = calibrate(samples, preds, ARB, ["tattoo"])
    assert set(rep["tattoo"]) == set(levels())
    std = rep["tattoo"]["standard"]
    assert std["current"]["n"] == 9 and std["current"]["wrong_pass"] == 0.0
    assert "严格度阈值校准" in render(rep)
