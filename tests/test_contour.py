"""轮廓复核：关键点把缺失的手指"脑补"出来时，肤色掩码应能发现。"""
from conftest import draw_hand, make_hand
from limbcheck.adapters.contour import SkinMask, contour_result


def test_all_fingers_present(thresholds):
    h = make_hand()
    img = draw_hand(h.pts)
    r, cov = contour_result(img, [h], SkinMask(), thresholds["contour"], expected=1)
    assert r.positive is False
    assert min(cov["right"].values()) == 1.0


def test_hallucinated_finger_is_caught(thresholds):
    h = make_hand()
    img = draw_hand(h.pts, skip=("ring",))     # 图上没有无名指，关键点却有
    r, cov = contour_result(img, [h], SkinMask(), thresholds["contour"], expected=1)
    assert r.positive is True
    assert r.parts == ["right_hand.ring"]
    assert cov["right"]["ring"] < thresholds["contour"]["tip_on_mask_min"]


def test_missing_palm_part_is_caught(thresholds):
    import cv2
    import numpy as np

    h = make_hand()
    img = draw_hand(h.pts)
    p = h.pts
    c = p[[0, 1, 5, 9, 13, 17]].mean(axis=0)
    gap = [p[0], c, p[9], p[13], p[17] + (p[17] - p[13]) * 0.5, p[0] + (p[17] - p[13]) * 0.8]   # 抹掉小指侧约 1/3 掌面
    cv2.fillPoly(img, [np.array(gap).astype(np.int32)], (250, 250, 250))
    r, cov = contour_result(img, [h], SkinMask(), thresholds["contour"], expected=1)
    assert r.positive is True and "right_hand.palm" in r.parts
    assert cov["right"]["palm"] < thresholds["contour"]["palm_on_mask_min"]
