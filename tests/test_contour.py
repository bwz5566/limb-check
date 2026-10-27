"""轮廓复核：关键点把缺失的手指"脑补"出来时，肤色掩码应能发现。"""
from conftest import draw_hand, make_hand
from limbcheck.adapters.contour import SkinMask, contour_result
from limbcheck.keypoints import HAND_FINGER_IDS


def test_all_fingers_present(thresholds):
    h = make_hand()
    img = draw_hand(h.pts)
    r, cov = contour_result(img, [h], SkinMask(), thresholds["contour"], expected=1)
    assert r.positive is False
    assert min(v for k, v in cov["right"].items() if k != "gaps") == 1.0
    h = spread_hand()
    r, cov = contour_result(draw_hand(h.pts), [h], SkinMask(), thresholds["contour"], expected=1, open_hand=True)
    assert r.positive is False and cov["right"]["gaps"] == {"index-middle": True, "middle-ring": True, "ring-little": True}


def spread_hand():
    return make_hand(spread=True)


def test_closed_fingers_cannot_be_verified_in_open_hand_view(thresholds):
    h = make_hand()                                 # 四指伸直但并拢：无名指、小指向中指靠拢
    h.pts[[14, 15, 16], 0] -= 0.12 * 120
    h.pts[[18, 19, 20], 0] -= 0.26 * 120
    r, _ = contour_result(draw_hand(h.pts), [h], SkinMask(), thresholds["contour"], expected=1, open_hand=True)
    assert r.positive is None and r.quality_ok and "并拢" in r.note      # 这一路无法判断，不算画质不足
    r, _ = contour_result(draw_hand(h.pts), [h], SkinMask(), thresholds["contour"], expected=1)     # 单手近景等视角不要求张开
    assert r.positive is False


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


def test_occluded_finger_without_gap_is_caught(thresholds):
    """手指被肤色物体（另一只手）挡住、关键点仍画出伸直的手指：两指之间看不到缝隙。"""
    import cv2
    import numpy as np

    h = spread_hand()
    img = draw_hand(h.pts)
    p = h.pts
    cv2.fillPoly(img, [np.array([p[9], p[12], p[16], p[13]]).astype(np.int32)], (205, 160, 135))   # 中指、无名指之间填成皮肤
    r, cov = contour_result(img, [h], SkinMask(), thresholds["contour"], expected=1, open_hand=True)
    assert r.positive is True and {"right_hand.middle", "right_hand.ring"} <= set(r.parts)
    assert any(f.label == "no_gap" for f in r.findings)


def test_curled_finger_drawn_extended_is_caught(thresholds):
    """手指蜷在掌心，关键点却画成伸直：中节、末节落在背景上。"""
    h = make_hand()
    img = draw_hand(h.pts, skip=("index",))
    r, _ = contour_result(img, [h], SkinMask(), thresholds["contour"], expected=1)
    assert "right_hand.index" in r.parts


def test_gap_seen_by_contrast_when_background_is_skin_like(thresholds):
    """手挡在脸前、浅色墙前：背景颜色接近肤色，掩码分不出缝隙，但缝隙处的明暗与两侧手指不同（真人实测）。"""
    import cv2
    import numpy as np

    h = spread_hand()
    img = draw_hand(h.pts, bg=(135, 98, 82))           # 背景（脸）与手指同色系、暗一些：掩码当成皮肤，但明暗不同
    p = h.pts
    th = thresholds["contour"]
    r, cov = contour_result(img, [h], SkinMask(), th, expected=1, open_hand=True)
    assert all(v is True for v in cov["right"]["gaps"].values()), cov["right"]["gaps"]
    assert r.positive is False
    r, cov = contour_result(img, [h], SkinMask(), {**th, "gap_delta_e": None}, expected=1, open_hand=True)
    assert any(v is False for v in cov["right"]["gaps"].values())     # 只看掩码时会误报"看不到缝隙"

    # 同色物体整块挡住两指之间：整条线颜色不变，对比度也救不回来
    img2 = draw_hand(h.pts)
    cv2.fillPoly(img2, [np.array([p[9], p[12], p[16], p[13]]).astype(np.int32)], (205, 160, 135))
    r, cov = contour_result(img2, [h], SkinMask(), th, expected=1, open_hand=True)
    assert cov["right"]["gaps"]["middle-ring"] is False and r.positive is True
