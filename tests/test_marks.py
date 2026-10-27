"""纹身 / 疤痕录用标准（configs/policies/marks.yaml）逐条验证：大模型输出 → 标准判定 → 仲裁。"""
import json

import pytest

from limbcheck.arbiter import arbitrate
from limbcheck.config import load_thresholds
from limbcheck.marks import make_mark, measure, region_of, session_review
from limbcheck.prompts import parse_reply, to_check_results

ARB = load_thresholds()["arbitration"]
# 一张 1000×1000 的图，比例 0.5 mm/像素：框 [0,0,100,100] = 50×50 mm
META = {"img_w": 1000, "img_h": 1000, "mm_per_px": 0.5, "scale_source": "测试"}


def tattoo(part, content="benign", ink="light", bbox=None, vs_ref=None, fingers=None, conf=0.9, **attrs):
    return {"check": "tattoo", "part": part, "label": "tattoo", "confidence": conf, "fingers": fingers or [],
            "bbox": bbox, "attrs": {"content": content, "ink": ink, **({"vs_ref": vs_ref} if vs_ref else {}), **attrs}}


def scar(part, bbox=None, conf=0.9, **attrs):
    return {"check": "scar", "part": part, "label": "scar", "confidence": conf, "bbox": bbox,
            "attrs": {"pattern": "single", "hypertrophic": "none", **attrs}}


def run(items, view="arms_out", checks=("tattoo", "scar"), meta=META, attempt=1):
    text = json.dumps({"items": items, "image_quality": "good", "reason": "测试"}, ensure_ascii=False)
    parsed, errs = parse_reply(text, list(checks))
    assert not errs, errs
    res = to_check_results(parsed, list(checks), view, "L2:test", errs, meta)
    return {c: (res[c], arbitrate(c, None, res[c], ARB, "judge", attempt)) for c in checks}


def verdicts(r):
    return [f.value["verdict"] for f in r.findings if isinstance(f.value, dict) and "verdict" in f.value]


def test_region_names():
    assert region_of("left_forearm") == ("forearm", "left")
    assert region_of("right_wrist") == ("forearm", "right")
    assert region_of("neck") == ("neck", None)
    assert region_of("right_hand", "backs") == ("hand_back", "right")
    assert region_of("right_hand", "palms") == ("palm", "right")
    assert region_of("left_finger", fingers=["ring"]) == ("finger", "left")
    assert region_of("left_upper_arm") == ("upper_arm", "left")


def test_measure_rel1000():
    assert measure([100, 100, 271, 208], META) == {"w_mm": 85.5, "h_mm": 54.0, "area_mm2": 4617.0, "source": "测试"}
    assert measure([0, 0, 10, 10], {"mm_per_px": 1}) is None          # 缺图像尺寸
    assert make_mark(tattoo("left_forearm", size_cm=[3, 2]), "arms_out").size["area_mm2"] == 600


def test_small_benign_arm_tattoo_passes():
    (r, a), = [run([tattoo("left_forearm", bbox=[0, 0, 100, 60])], checks=("tattoo",))["tattoo"]]
    assert r.positive is False and a.decision == "pass" and "均符合录用标准" in a.reason


def test_neck_tattoo_blocked_even_if_small():
    r, a = run([tattoo("neck", bbox=[0, 0, 20, 20])], view="neck", checks=("tattoo",))["tattoo"]
    assert a.decision == "fail_suggested" and "脖子" in a.reason and "即使面积小" in a.reason


@pytest.mark.parametrize("content", ["beast", "deity_demon", "skull_horror", "weapon", "gang", "abusive_text"])
def test_aggressive_content_blocked_regardless_of_size(content):
    r, a = run([tattoo("right_upper_arm", content=content, bbox=[0, 0, 20, 20])], checks=("tattoo",))["tattoo"]
    assert a.decision == "fail_suggested" and "无论面积大小" in a.reason


def test_sleeping_tiger_is_beast():
    r, a = run([tattoo("right_forearm", content="beast", ink="faded", bbox=[0, 0, 20, 20])], checks=("tattoo",))["tattoo"]
    assert a.decision == "fail_suggested"      # 猛兽类无论睁眼、闭眼、是否洗淡都不录用


def test_large_arm_tattoo_blocked_unless_faded():
    big = [0, 0, 400, 300]          # 200×150 mm，远大于身份证
    r, a = run([tattoo("left_upper_arm", bbox=big)], checks=("tattoo",))["tattoo"]
    assert a.decision == "fail_suggested" and "面积 ≥ 身份证" in a.reason
    r, a = run([tattoo("left_upper_arm", bbox=big, ink="faded")], checks=("tattoo",))["tattoo"]
    assert a.decision == "pass"     # 面积大但洗过明显变淡、无攻击性 → 放行


def test_borderline_size_goes_to_review():
    r, a = run([tattoo("left_forearm", bbox=[0, 0, 171, 108])], checks=("tattoo",))["tattoo"]   # ≈ 身份证大小
    assert a.decision == "review" and "临界" in a.reason


def test_dark_arm_tattoo_review():
    r, a = run([tattoo("left_forearm", ink="dark", bbox=[0, 0, 60, 60])], checks=("tattoo",))["tattoo"]
    assert a.decision == "review" and "颜色过深" in a.reason


def test_unknown_size_uses_model_comparison():
    r, a = run([tattoo("left_forearm", vs_ref="smaller")], checks=("tattoo",), meta=None)["tattoo"]
    assert a.decision == "pass"
    r, a = run([tattoo("left_forearm")], checks=("tattoo",), meta=None)["tattoo"]
    assert a.decision == "review" and "无法估计面积" in a.reason


def test_measure_conflicting_with_model_is_review():
    r, a = run([tattoo("left_forearm", bbox=[0, 0, 400, 300], vs_ref="smaller")], checks=("tattoo",))["tattoo"]
    assert a.decision == "review"


def test_single_small_light_finger_tattoo_passes():
    small = [0, 0, 20, 20]          # 10×10 mm < 1 分硬币面积 2.54 cm²
    r, a = run([tattoo("right_finger", fingers=["index"], bbox=small)], view="backs", checks=("tattoo",))["tattoo"]
    assert a.decision == "pass"


def test_finger_rules():
    small = [0, 0, 20, 20]
    # 两根手指各一处 → 不录用
    r, a = run([tattoo("right_finger", fingers=["index"], bbox=small), tattoo("left_finger", fingers=["ring"], bbox=small)],
               view="backs", checks=("tattoo",))["tattoo"]
    assert a.decision == "fail_suggested" and "手指纹身 2 处" in a.reason
    # 面积 ≥ 1 分硬币 → 不录用
    r, a = run([tattoo("right_finger", fingers=["index"], bbox=[0, 0, 60, 60])], view="backs", checks=("tattoo",))["tattoo"]
    assert a.decision == "fail_suggested" and "1 分硬币" in a.reason
    # 颜色过深 → 不录用
    r, a = run([tattoo("right_finger", fingers=["index"], ink="dark", bbox=small)], view="backs", checks=("tattoo",))["tattoo"]
    assert a.decision == "fail_suggested" and "颜色过深" in a.reason
    # 整体洗过褪色的淡痕，即使不小 → 放行
    r, a = run([tattoo("right_finger", fingers=["index"], ink="faded", bbox=[0, 0, 60, 60])], view="backs", checks=("tattoo",))["tattoo"]
    assert a.decision == "pass"


def test_hand_back_rules():
    r, a = run([tattoo("right_hand_back", bbox=[0, 0, 40, 40])], view="backs", checks=("tattoo",))["tattoo"]
    assert a.decision == "review" and "外露高频" in a.reason            # 单处小面积 → 人工核验（可在配置里改为放行）
    r, a = run([tattoo("right_hand_back", bbox=[0, 0, 40, 40]), tattoo("right_hand_back", bbox=[500, 500, 540, 540])],
               view="backs", checks=("tattoo",))["tattoo"]
    assert a.decision == "fail_suggested" and "多处纹身叠加" in a.reason


def test_stain_asks_for_cleaning_then_review():
    stain = {"check": "tattoo", "part": "left_forearm", "label": "stain", "confidence": 0.8, "bbox": [0, 0, 50, 50]}
    r, a = run([stain], checks=("tattoo",), attempt=0)["tattoo"]
    assert r.positive is None and r.quality_ok is False
    assert a.decision == "redo" and "清洁" in a.reason
    r, a = run([stain], checks=("tattoo",), attempt=1)["tattoo"]
    assert a.decision == "review" and "清洁" in a.reason


def test_uncertain_tattoo_vs_scar_goes_to_review():
    item = tattoo("left_forearm", bbox=[0, 0, 40, 40])
    item["uncertain"] = True
    r, a = run([item], checks=("tattoo",))["tattoo"]
    assert a.decision == "review" and "不确定" in a.reason


def test_scar_rules():
    def one(**kw):
        return run([scar("left_forearm", **kw)], checks=("scar",))["scar"][1]

    assert one(bbox=[0, 0, 40, 10]).decision == "pass"                            # 单条 20 mm、无增生
    assert "自伤" in one(bbox=[0, 0, 100, 60], pattern="parallel_multiple", count=5).reason
    assert one(bbox=[0, 0, 100, 60], pattern="parallel_multiple", count=5).decision == "fail_suggested"
    assert one(bbox=[0, 0, 40, 10], hypertrophic="severe").decision == "fail_suggested"
    assert one(bbox=[0, 0, 200, 10]).decision == "fail_suggested"                 # 100 mm 大面积长条
    assert one(bbox=[0, 0, 80, 10]).decision == "review"                          # 40 mm 介于短小与长条之间
    assert one(bbox=[0, 0, 120, 10], surgical=True).decision == "fail_suggested"  # 60 mm 手术刀疤，默认按精细作业
    assert one(bbox=[0, 0, 40, 10], with_tattoo=True).decision == "fail_suggested"
    a = run([scar("left_forearm", bbox=[0, 0, 120, 10], surgical=True)], checks=("scar",),
            meta={**META, "params": {"scar": {"fine_work": False}}})["scar"][1]
    assert a.decision == "review"                                                 # 非精细作业岗位 → 人工核验
    a = run([scar("right_hand_back", bbox=[0, 0, 40, 10], count=3, pattern="multiple_scattered")], view="backs", checks=("scar",))["scar"][1]
    assert a.decision == "fail_suggested" and "手背多处陈旧疤痕" in a.reason


def test_scar_with_tattoo_same_region_in_one_view():
    out = run([tattoo("left_forearm", bbox=[0, 0, 40, 40]), scar("left_forearm", bbox=[100, 100, 140, 110])])
    assert out["scar"][1].decision == "fail_suggested" and "疤痕同时搭配纹身" in out["scar"][1].reason


def test_session_cross_view_finger_count():
    small = [0, 0, 20, 20]
    backs = run([tattoo("right_finger", fingers=["index"], bbox=small)], view="backs")
    palms = run([tattoo("left_finger", fingers=["ring"], bbox=small)], view="palms")
    assert backs["tattoo"][1].decision == "pass" and palms["tattoo"][1].decision == "pass"   # 单个视角都只有 1 处
    sr = session_review({"backs": {c: r for c, (r, _) in backs.items()}, "palms": {c: r for c, (r, _) in palms.items()}})
    t = sr["tattoo"]
    assert t.action == "block" and "手指纹身 2 处" in t.rules[0]["text"] and len(t.marks) == 2
    # 同一根手指在两个视角都拍到 → 只算 1 处
    palms2 = run([tattoo("right_finger", fingers=["index"], bbox=small)], view="palms")
    sr = session_review({"backs": {c: r for c, (r, _) in backs.items()}, "palms": {c: r for c, (r, _) in palms2.items()}})
    assert sr["tattoo"].action == "pass"


def test_session_reminders():
    stain = {"check": "tattoo", "part": "right_hand_back", "label": "stain", "confidence": 0.8}
    out = run([stain], view="backs", checks=("tattoo",))
    sr = session_review({"backs": {"tattoo": out["tattoo"][0]}})
    assert sr["tattoo"].reminders and "清洁" in sr["tattoo"].reminders[0]["message"]


def test_none_is_pass():
    r, a = run([{"check": "tattoo", "part": "all_visible", "label": "none", "confidence": 0.95}], checks=("tattoo",))["tattoo"]
    assert r.positive is False and a.decision == "pass"
