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
