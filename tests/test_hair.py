"""发色 / 发长（configs/policies/hair.yaml，limbcheck/appearance.py）：用合成的分割图和关键点验证，不依赖模型。"""
import copy

import numpy as np
import pytest

from limbcheck.adapters.segmenter import BACKGROUND, BODY_SKIN, CLOTHES, FACE_SKIN, HAIR, OTHERS
from limbcheck.appearance import classify, hair_policy, hair_result, wb_gains
from limbcheck.arbiter import arbitrate
from limbcheck.config import load_thresholds
from limbcheck.core import FAIL_SUGGESTED, PASS, REVIEW
from limbcheck.geometry import rgb_to_lab
from limbcheck.keypoints import Body

H, W = 480, 640
SKIN = (205, 160, 135)
BLACK = (28, 26, 27)
COLORS = {"red": (175, 45, 35), "purple": (120, 50, 140), "green": (40, 140, 60), "blonde": (215, 180, 95),
          "blue": (40, 80, 170)}


class FakeSeg:
    def __init__(self, cat):
        self.cat = cat

    def categories(self, img):
        return self.cat


def scene(hair=BLACK, hair_bottom=170, streaks=None, cast=None, others_hair=False):
    """一个人的头肩像：头发（y 60–hair_bottom，长发会垂到肩下）、脸（y 110–210）、脖子、衣服、双肩在 y≈250。"""
    img = np.full((H, W, 3), 225, np.uint8)
    cat = np.full((H, W), BACKGROUND, np.uint8)
    yy, xx = np.mgrid[:H, :W]
    hair_m = (((xx - 320) / 95) ** 2 + ((yy - 140) / 85) ** 2 <= 1) & (yy < 140)
    hair_m |= (np.abs(xx - 320) <= 95) & (np.abs(xx - 320) >= 70) & (yy >= 120) & (yy < hair_bottom)
    face_m = ((xx - 320) / 68) ** 2 + ((yy - 160) / 55) ** 2 <= 1
    neck_m = (np.abs(xx - 320) <= 35) & (yy >= 200) & (yy < 250)
    body_m = (np.abs(xx - 320) <= 170) & (yy >= 240)
    for m, c, k in [(body_m, (60, 90, 140), CLOTHES), (neck_m, SKIN, BODY_SKIN), (face_m & ~hair_m, SKIN, FACE_SKIN)]:
        img[m], cat[m] = c, k
    img[hair_m], cat[hair_m] = hair, OTHERS if others_hair else HAIR
    if streaks is not None:   # 竖条挑染
        s = hair_m & ((xx // 10) % 4 == 0)
        img[s] = streaks
    if cast is not None:      # 摄像头偏色：每个通道乘一个倍数
        img = np.clip(img * np.asarray(cast), 0, 255).astype(np.uint8)
    body = Body(joints={"left_shoulder": np.array([430.0, 250.0]), "right_shoulder": np.array([210.0, 250.0]),
                        "left_eye": np.array([345.0, 150.0]), "right_eye": np.array([295.0, 150.0])},
                vis={"left_shoulder": 0.9, "right_shoulder": 0.9, "left_eye": 0.9, "right_eye": 0.9})
    return img, cat, body


def run(gender=None, **kw):
    img, cat, body = scene(**kw)
    return hair_result(img, FakeSeg(cat), body, {"gender": gender} if gender else {})


def finding(r, kind):
    return next(f for f in r.findings if f.value.get("kind") == kind)


def test_classify_colors():
    pol = hair_policy()
    names = list(pol["colors"])
    lab = rgb_to_lab(np.array([BLACK, (95, 65, 45), *COLORS.values()], float))
    got = [names[i] for i in classify(lab, pol["colors"])]
    assert got[:2] == ["black", "brown"]
    assert got[2:] == ["red", "pink_purple", "green", "blonde", "blue"]


def test_black_hair_passes():
    r = run()
    assert r.positive is False and r.confidence >= 0.8
    assert finding(r, "hair_color").value["verdict"] == PASS


@pytest.mark.parametrize("name", list(COLORS))
def test_whole_dye_blocked(name):
    r = run(hair=COLORS[name])
    f = finding(r, "hair_color")
    assert r.positive is True and f.label == "dyed" and f.value["verdict"] == "block", f.note
    assert f.confidence <= hair_policy()["dyed"]["max_confidence"]


def test_dyed_hair_segmented_as_others_still_counted():
    """亮色染发常被分割成"其他"类：头部范围内的"其他"也算头发。"""
    r = run(hair=COLORS["blonde"], others_hair=True)
    assert finding(r, "hair_color").label == "dyed"


def test_highlights_blocked():
    r = run(streaks=COLORS["red"])
    f = finding(r, "hair_color")
    assert f.label == "highlights" and f.value["verdict"] == "block", f.note
    assert 0.08 < f.value["blocked_fraction"] < 0.35


def test_borderline_goes_to_review():
    pol = copy.deepcopy(hair_policy())
    img, cat, body = scene(streaks=COLORS["red"])
    frac = finding(hair_result(img, FakeSeg(cat), body, {}, pol), "hair_color").value["blocked_fraction"]
    pol["dyed"]["highlights_min"] = frac * 1.1      # 占比略低于阈值
    f = finding(hair_result(img, FakeSeg(cat), body, {}, pol), "hair_color")
    assert f.value["verdict"] == REVIEW


def test_white_balance_removes_camera_cast():
    """偏紫的摄像头会把黑发拍成紫色：按面部肤色校正后仍是自然发色。"""
    cast = (1.0, 0.78, 1.2)
    img, cat, body = scene(hair=(70, 70, 70), cast=cast)
    pol = copy.deepcopy(hair_policy())
    pol["white_balance"]["enabled"] = False
    raw = finding(hair_result(img, FakeSeg(cat), body, {}, pol), "hair_color")
    fixed = finding(hair_result(img, FakeSeg(cat), body, {}), "hair_color")
    assert raw.value["verdict"] != PASS
    assert fixed.value["verdict"] == PASS and fixed.value["white_balance"]["applied"]


def test_wb_gains_limited():
    g, info = wb_gains(np.tile([[200, 120, 220]], (500, 1)), hair_policy()["white_balance"])
    assert info["applied"] and g.max() <= hair_policy()["white_balance"]["max_gain"] + 1e-6
    assert wb_gains(np.zeros((10, 3)), hair_policy()["white_balance"])[0] is None


def test_other_person_hair_ignored():
    img, cat, body = scene()
    img[20:80, 560:630], cat[20:80, 560:630] = COLORS["red"], HAIR      # 背景里另一个人的红头发（不相连）
    cat[80:200, 570:620] = CLOTHES
    r = hair_result(img, FakeSeg(cat), body, {})
    assert finding(r, "hair_color").value["verdict"] == PASS


def test_too_little_hair_unknown():
    img, cat, body = scene()
    cat[cat == HAIR] = BACKGROUND
    r = hair_result(img, FakeSeg(cat), body, {})
    assert r.positive is None and not r.quality_ok


def test_length_male_rules():
    short = run("male")
    assert finding(short, "hair_length").value["level"] == "short" and short.positive is False
    long_ = run("male", hair_bottom=330)
    f = finding(long_, "hair_length")
    assert f.value["level"] == "long" and f.value["verdict"] == "block" and long_.positive is True
    near = run("male", hair_bottom=240)       # 约及肩：接近阈值 → 人工核验
    assert finding(near, "hair_length").value["verdict"] == REVIEW


def test_length_only_recorded_without_gender():
    for g in (None, "female"):
        r = run(g, hair_bottom=330)
        f = finding(r, "hair_length")
        assert r.positive is False and f.positive is None and "verdict" not in f.value


def test_length_head_top_out_of_frame_uses_eyes():
    img, cat, body = scene(hair_bottom=330)
    img, cat = img[90:], cat[90:]                     # 头顶出画
    body.joints = {k: v - [0, 90] for k, v in body.joints.items()}
    f = finding(hair_result(img, FakeSeg(cat), body, {"gender": "male"}), "hair_length")
    assert f.value["top_from"] == "eyes" and f.value["verdict"] == "block"


def test_male_without_shoulders_reviewed():
    img, cat, _ = scene()
    r = hair_result(img, FakeSeg(cat), None, {"gender": "male"})
    assert finding(r, "hair_length").value["verdict"] == REVIEW


def test_shoulders_from_silhouette_when_no_keypoints():
    """背面时关键点模型常找不到人：从人体轮廓（宽度超过头宽 2 倍处）估算肩线。"""
    def judge(hair_bottom):
        img, cat, _ = scene(hair_bottom=hair_bottom)
        yy, xx = np.mgrid[:H, :W]
        cat[(np.abs(xx - 320) <= 230) & (yy >= 240) & (cat == BACKGROUND)] = CLOTHES
        return finding(hair_result(img, FakeSeg(cat), None, {"gender": "male"}), "hair_length").value

    long_, short = judge(330), judge(170)
    assert long_["shoulder_from"] == "silhouette" and long_["verdict"] == "block"
    assert short["level"] == "short" and short["verdict"] == PASS


def test_arbitration_uses_policy():
    arb = load_thresholds()["arbitration"]
    red = run(hair=COLORS["red"])
    a = arbitrate("hair", red, None, arb, "judge", 1)
    assert a.decision == FAIL_SUGGESTED and "红色" in a.reason
    ok = arbitrate("hair", run(), None, arb, "judge", 1)
    assert ok.decision == PASS and "自然发色" in ok.reason and "处" not in ok.reason
