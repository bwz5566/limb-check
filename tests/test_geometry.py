import numpy as np
import pytest

from conftest import hand_points, make_body, make_hand
from limbcheck.geometry import analyze_fingers, facing, fingers_result, limbs_result, orient_sign, rgb_to_lab


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


def test_lab_reference_values():
    assert np.allclose(rgb_to_lab([255, 255, 255]), [100, 0, 0], atol=0.5)
    assert np.allclose(rgb_to_lab([0, 0, 0]), [0, 0, 0], atol=0.5)
    assert np.allclose(rgb_to_lab([255, 0, 0]), [53.2, 80.1, 67.2], atol=0.6)
