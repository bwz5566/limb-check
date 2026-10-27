"""流水线与可替换节点：用假关键点模型 + 合成手图跑通完整流程。"""
import numpy as np
import pytest

from conftest import FakeKeypoints, draw_hand, make_body, make_hand
from limbcheck import registry
from limbcheck.config import load_job_profiles
from limbcheck.core import CheckResult
from limbcheck.pipeline import Pipeline


def _two_hands():
    r = make_hand("right", "palm", center=(170, 420))
    l = make_hand("left", "palm", center=(470, 420))
    return r, l


def test_palms_normal_passes(thresholds):
    r, l = _two_hands()
    img = draw_hand(l.pts, shape=(560, 640)) & draw_hand(r.pts, shape=(560, 640))   # 两只手画在同一张白底图上
    p = Pipeline(thresholds, keypoints=FakeKeypoints([r, l]), contour=registry.create("contour", "skin"), l1=["fingers.keypoints", "fingers.contour"])
    rec = p.run(img, "palms")
    f = rec["results"]["fingers"]
    assert f["l1"]["positive"] is False and f["shadow"] == "pass"
    assert set(rec["latency_ms"]) >= {"keypoints", "contour"}


def test_hallucinated_finger_goes_to_human(thresholds):
    r, l = _two_hands()
    img = draw_hand(l.pts, shape=(560, 640), skip=("index",)) & draw_hand(r.pts, shape=(560, 640))
    p = Pipeline(thresholds, keypoints=FakeKeypoints([r, l]), contour=registry.create("contour", "skin"), l1=["fingers.keypoints", "fingers.contour"])
    f = p.run(img, "palms")["results"]["fingers"]
    assert f["l1"]["positive"] is True and "left_hand.index" in f["l1"]["parts"]
    assert f["shadow"] == "fail_suggested"


def test_profile_modes_and_limbs(thresholds):
    prof = load_job_profiles()["warehouse"]
    p = Pipeline(thresholds, keypoints=FakeKeypoints(bodies=[make_body()]), profile=prof, l1=["limbs.pose"])
    rec = p.run(np.full((640, 640, 3), 128, np.uint8), "stand")
    assert rec["results"]["limbs"]["decision"] == "pass"
    assert "tattoo" not in rec["results"]      # 没有 L1 也没有 L2 时不出结论


def test_plugin_l1_for_new_check(thresholds):
    @registry.register("l1", "hair.fake_test", check="hair")
    def fake_hair(ctx):
        return CheckResult("hair", "L1:fake", True, 0.9, ["head"])

    @registry.register("l1", "tattoo.fake_test", check="tattoo")
    def fake_tattoo(ctx):
        return CheckResult("tattoo", "L1:fake", True, 0.9, ["left_forearm"])

    prof = load_job_profiles()["precision"]
    p = Pipeline(thresholds, profile=prof, l1=["hair.fake_test", "tattoo.fake_test"])
    res = p.run(np.zeros((100, 100, 3), np.uint8), "stand")["results"]
    h = res["hair"]
    assert h["decision"] == "not_assessed" and h["shadow"] == "fail_suggested" and h["mode"] == "record_only"
    t = res["tattoo"]      # 纹身已有书面依据（录用标准）→ judge；没有逐处标准判定信息时按默认仲裁
    assert t["decision"] == "fail_suggested" and t["mode"] == "judge"


def test_custom_keypoint_model_by_path(thresholds):
    km = registry.create("keypoints", {"type": "conftest:FakeKeypoints", "hands": list(_two_hands())})
    assert len(km.hands(None)) == 2


def test_node_failure_is_recorded(thresholds):
    @registry.register("l1", "fingers.broken_test", check="fingers")
    def broken(ctx):
        raise RuntimeError("模型挂了")

    p = Pipeline(thresholds, keypoints=FakeKeypoints(list(_two_hands())), l1=["fingers.broken_test", "fingers.keypoints"])
    rec = p.run(np.full((560, 640, 3), 250, np.uint8), "palms")
    assert any("模型挂了" in e for e in rec["errors"]) and "fingers" in rec["results"]


def test_unknown_node_lists_choices():
    with pytest.raises(KeyError) as e:
        registry.get("contour", "nope")
    assert "skin" in str(e.value)
