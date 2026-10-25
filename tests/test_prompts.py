import json

from limbcheck.prompts import build_messages, build_user_prompt, l2_checks_for, parse_reply, schema, to_check_results


def test_checks_for_views():
    assert l2_checks_for("palms") == ["fingers"]
    assert l2_checks_for("backs") == ["fingers", "nails", "tattoo", "scar"]
    assert l2_checks_for("stand", ["limbs", "hair"]) == ["limbs", "hair"]


def test_prompt_only_mentions_enabled_checks():
    p = build_user_prompt("backs", ["fingers", "nails"])
    assert "fingers" in p and "nails" in p and "tattoo：" not in p and "hair：" not in p
    msgs = build_messages("backs", ["fingers"], ["data:image/jpeg;base64,AAAA"])
    assert msgs[0]["role"] == "system" and "不要描述" in msgs[0]["content"]
    assert msgs[1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg")


def test_schema_restricts_checks():
    s = schema(["fingers"])
    assert s["properties"]["items"]["items"]["properties"]["check"]["enum"] == ["fingers"]


def _reply(items, quality="good"):
    return "好的，结果如下：\n```json\n" + json.dumps({"items": items, "image_quality": quality, "reason": "测试"}, ensure_ascii=False) + "\n```"


def test_parse_and_convert_positive():
    text = _reply([
        {"check": "fingers", "part": "right_hand", "label": "normal", "fingers": [], "value": 5, "confidence": 0.9},
        {"check": "fingers", "part": "left_hand", "label": "missing_or_short", "fingers": ["ring"], "value": 4, "confidence": 0.8},
        {"check": "tattoo", "part": "all_visible", "label": "none", "confidence": 0.9},   # 未要求 → 丢弃
    ])
    parsed, errs = parse_reply(text, ["fingers"])
    assert any("未要求" in e for e in errs)
    res = to_check_results(parsed, ["fingers"], "palms", "L2:test")["fingers"]
    assert res.positive is True and res.parts == ["left_hand.ring"] and res.confidence == 0.8


def test_negative_requires_both_hands():
    text = _reply([{"check": "fingers", "part": "right_hand", "label": "normal", "confidence": 0.95}])
    parsed, _ = parse_reply(text, ["fingers"])
    res = to_check_results(parsed, ["fingers"], "palms", "L2:test")["fingers"]
    assert res.positive is None and "只返回了 1 只手" in res.note


def test_uncertain_positive_still_positive_with_lower_conf():
    text = _reply([{"check": "scar", "part": "left_forearm", "label": "scar", "confidence": 0.8, "uncertain": True}])
    parsed, _ = parse_reply(text, ["scar"])
    res = to_check_results(parsed, ["scar"], "stand", "L2:test")["scar"]
    assert res.positive is True and res.confidence == 0.4


def test_bad_label_and_poor_quality():
    text = _reply([{"check": "hair", "part": "head", "label": "purple", "confidence": 0.9}], quality="poor")
    parsed, errs = parse_reply(text, ["hair"])
    assert any("不在约定范围" in e for e in errs)
    res = to_check_results(parsed, ["hair"], "stand", "L2:test")["hair"]
    assert res.positive is None and res.quality_ok is False


def test_unparseable():
    parsed, errs = parse_reply("抱歉，我无法判断。", ["fingers"])
    assert parsed is None and errs
    res = to_check_results(parsed, ["fingers"], "palms", "L2:test", errs)["fingers"]
    assert res.positive is None


def test_nested_braces_in_strings():
    text = '{"items": [{"check": "hair", "part": "head", "label": "natural", "value": "黑色 {偏棕}", "confidence": 0.9}], "image_quality": "good"} 其他文字 }'
    parsed, errs = parse_reply(text, ["hair"])
    assert parsed["items"][0]["value"] == "黑色 {偏棕}"
