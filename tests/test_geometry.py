import numpy as np
import pytest

from conftest import draw_body, hand_points, make_body, make_hand
from limbcheck.geometry import (analyze_fingers, facing, fingers_result, limbs_contour_result, limbs_result, orient_sign,
                                rgb_to_lab)
from limbcheck.keypoints import mirror_body


def test_normal_hand_has_no_flags(thresholds):
    a = analyze_fingers(hand_points(), thresholds["fingers"])
    assert a["flagged"] == []
    assert a["extended"] == 5
    assert a["mid_to_palm"] > thresholds["fingers"]["middle_to_palm_min"]


@pytest.mark.parametrize("finger,ids", [("index", (6, 7, 8)), ("ring", (14, 15, 16)), ("little", (18, 19, 20))])
def test_short_finger_is_flagged(thresholds, finger, ids):
    pts = hand_points()
    mcp = {"index": 5, "ring": 13, "little": 17}[finger]
    for i in ids:   # 把整根手指压缩到 MCP 附近
        pts[i] = pts[mcp] + (pts[i] - pts[mcp]) * 0.2
    assert analyze_fingers(pts, thresholds["fingers"])["flagged"] == [finger]


def test_short_middle_finger_uses_palm_ratio(thresholds):
    pts = hand_points()
    for i in (10, 11, 12):
        pts[i] = pts[9] + (pts[i] - pts[9]) * 0.3
    assert "middle" in analyze_fingers(pts, thresholds["fingers"])["flagged"]


@pytest.mark.parametrize("side,face", [("right", "back"), ("right", "palm"), ("left", "back"), ("left", "palm")])
def test_facing_from_chirality(side, face):
    pts = hand_points(side, face)
    assert facing(pts, side) == face
    # 平面内旋转不改变手性
    th = np.radians(130)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    assert facing((pts - pts[0]) @ rot.T + pts[0], side) == face
    # 镜像画面取反
    assert facing(pts, side, mirrored=True) != face


def test_right_back_sign_is_positive():
    assert orient_sign(hand_points("right", "back")) == 1


def test_fingers_result_missing_hand_is_unknown(thresholds):
    r = fingers_result([make_hand()], thresholds["fingers"], expected=2, source="L1:test")
    assert r.positive is None and not r.quality_ok


def test_fingers_result_parts(thresholds):
    h = make_hand("left", "palm")
    h.pts[[6, 7, 8]] = h.pts[5] + (h.pts[[6, 7, 8]] - h.pts[5]) * 0.2
    r = fingers_result([make_hand("right", "palm"), h], thresholds["fingers"], expected=2, source="L1:test")
    assert r.positive is True and r.parts == ["left_hand.index"]


def test_limbs_normal_and_asymmetric(thresholds):
    ok = limbs_result(make_body(), thresholds["pose"], "L1:test", lower=True, image_size=(640, 640))
    assert ok.positive is False and ok.confidence >= 0.8
    bad = limbs_result(make_body(arm_ratio=0.4), thresholds["pose"], "L1:test", lower=True, image_size=(640, 640))
    assert bad.positive is True and "left_forearm" in bad.parts


def test_limbs_out_of_frame_is_quality_issue(thresholds):
    r = limbs_result(make_body(), thresholds["pose"], "L1:test", lower=True, image_size=(640, 400))
    assert r.positive is None and not r.quality_ok and "未完整入框" in r.note


def test_limbs_low_visibility_inside_frame_is_suspect(thresholds):
    b = make_body()
    b.vis["right_wrist"] = 0.1
    r = limbs_result(b, thresholds["pose"], "L1:test", lower=True, image_size=(640, 640))
    assert r.positive is True and "right_arm" in r.parts


def test_limbs_missing_foot_is_suspect_and_foot_out_of_frame_is_note(thresholds):
    b = make_body()
    b.vis["left_foot_index"] = 0.1
    r = limbs_result(b, thresholds["pose"], "L1:test", lower=True, image_size=(640, 640))
    assert r.positive is True and r.parts == ["left_foot"]
    r = limbs_result(make_body(), thresholds["pose"], "L1:test", lower=True, image_size=(640, 590))
    assert r.positive is False and "左脚" in r.note and "右脚" in r.note
    r = limbs_result(b, thresholds["pose"], "L1:test", upper=True, lower=False, image_size=(640, 640))
    assert r.positive is False


def test_limbs_folded_joint_in_straight_pose(thresholds):
    b = make_body()
    b.joints["right_wrist"] = b.joints["right_elbow"] + (b.joints["right_shoulder"] - b.joints["right_elbow"]) * 0.6
    r = limbs_result(b, thresholds["pose"], "L1:test", lower=True, image_size=(640, 640), straight=True)
    assert "right_arm" in r.parts and any(f.label == "folded" for f in r.findings)
    assert limbs_result(make_body(), thresholds["pose"], "L1:test", lower=True, image_size=(640, 640), straight=True).positive is False
    assert "right_arm" not in limbs_result(b, thresholds["pose"], "L1:test", lower=True, image_size=(640, 640)).parts


def test_limbs_contour_flags_hallucinated_forearm(thresholds):
    b = make_body()
    ok = limbs_contour_result(draw_body(b), b, thresholds["pose"], "L1:test")
    assert ok.positive is False and ok.confidence >= 0.9
    bad = limbs_contour_result(draw_body(b, skip=("right_arm",)), b, thresholds["pose"], "L1:test")
    assert bad.positive is True and bad.parts == ["right_arm"]
    assert bad.findings[[f.part for f in bad.findings].index("right_arm")].label == "limb_off_mask"
    legs_only = limbs_contour_result(draw_body(b, skip=("right_arm",)), b, thresholds["pose"], "L1:test", upper=False)
    assert legs_only.positive is False


def test_mirror_body_swaps_anatomical_sides():
    b = make_body()
    b.vis["right_knee"] = 0.3
    m = mirror_body(b)
    assert np.allclose(m.joints["right_wrist"], b.joints["left_wrist"]) and m.vis["left_knee"] == 0.3
    assert np.allclose(m.joints["nose"], b.joints["nose"])


def test_lab_reference_values():
    assert np.allclose(rgb_to_lab([255, 255, 255]), [100, 0, 0], atol=0.5)
    assert np.allclose(rgb_to_lab([0, 0, 0]), [0, 0, 0], atol=0.5)
    assert np.allclose(rgb_to_lab([255, 0, 0]), [53.2, 80.1, 67.2], atol=0.6)


def test_mm_per_px_from_hand_and_body():
    from conftest import make_body, make_hand
    from limbcheck.geometry import mm_per_px_body, mm_per_px_hands

    anatomy = {"palm_length": 100, "palm_breadth": 70, "forearm": 250, "upper_arm": 300}
    h = make_hand(scale=200)                       # 掌长 200 像素
    k = mm_per_px_hands([h], anatomy)
    assert 0.3 < k <= 0.5                          # 取掌长、掌宽里缩短最少的一个，≤ 100/200
    b = make_body(scale=100)                       # 前臂 ≈ 90 像素、上臂 ≈ 100 像素
    kb = mm_per_px_body(b, anatomy)
    assert kb == pytest.approx(min(250 / 90.3, 300 / 100.5), rel=0.02)
    assert mm_per_px_hands([], anatomy) is None and mm_per_px_body(None, anatomy) is None


def test_nails_only_measured_on_extended_fingers(thresholds):
    import cv2

    from conftest import draw_hand
    from limbcheck.geometry import nails_delta_e, nails_result

    h = make_hand(spread=True)
    img = draw_hand(h.pts)
    assert nails_result(img, [h], thresholds["nails"], "L1:test").positive is False
    for tip, dip in ((8, 7), (12, 11), (16, 15)):          # 食指、中指、无名指涂红色甲油
        c = h.pts[tip] + 0.3 * (h.pts[dip] - h.pts[tip])
        cv2.circle(img, tuple(int(v) for v in c), 6, (200, 20, 40), -1)
    r = nails_result(img, [h], thresholds["nails"], "L1:test")
    assert r.positive is True and "食指" in r.findings[0].note
    bent = make_hand(spread=True)
    for i in (7, 8, 11, 12, 15, 16, 19, 20):             # 四指蜷起：关键点折回掌面
        bent.pts[i] = bent.pts[i - 2] + (bent.pts[0] - bent.pts[i - 2]) * 0.3
    assert sum(v is None for v in nails_delta_e(img, bent.pts).values()) >= 4
    assert nails_result(img, [bent], thresholds["nails"], "L1:test").positive is None
