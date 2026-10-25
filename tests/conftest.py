"""测试辅助：用几何方法合成手部关键点和对应的"手"图像，不依赖任何模型和外部图片。"""
from __future__ import annotations

import numpy as np
import pytest

from limbcheck.keypoints import Body, Hand

# 右手手背朝向镜头、五指向上（拇指在画面左侧）的标准手，掌长（腕→中指 MCP）= 1
_RIGHT_BACK = np.array([
    [0.0, 0.0],                                                       # 0 腕
    [-0.30, -0.25], [-0.55, -0.45], [-0.75, -0.60], [-0.90, -0.75],   # 拇指 CMC MCP IP TIP
    [-0.35, -0.95], [-0.35, -1.40], [-0.35, -1.67], [-0.35, -1.87],   # 食指
    [0.00, -1.00], [0.00, -1.50], [0.00, -1.80], [0.00, -2.02],       # 中指
    [0.32, -0.95], [0.32, -1.43], [0.32, -1.71], [0.32, -1.91],       # 无名指
    [0.60, -0.85], [0.60, -1.21], [0.60, -1.41], [0.60, -1.58],       # 小指
])


def hand_points(side: str = "right", facing: str = "back", center=(300.0, 420.0), scale: float = 120.0) -> np.ndarray:
    """返回 (21, 2) 像素坐标。左右手 / 掌心手背通过水平镜像得到（手性随之改变）。"""
    pts = _RIGHT_BACK.copy()
    if (side == "left") != (facing == "palm"):
        pts[:, 0] = -pts[:, 0]
    return pts * scale + np.asarray(center)


def make_hand(side="right", facing="back", center=(300.0, 420.0), scale=120.0, score=0.95) -> Hand:
    return Hand(pts=hand_points(side, facing, center, scale), score=score, side=side, side_score=0.95)


def draw_hand(pts: np.ndarray, shape=(560, 640), skip: tuple[str, ...] = (), skin=(205, 160, 135), bg=(250, 250, 250)) -> np.ndarray:
    """按关键点画一只"手"（RGB）。skip 中的手指不画，用来模拟缺指而关键点仍然"脑补"出来的情况。"""
    import cv2

    from limbcheck.keypoints import HAND_FINGER_IDS

    img = np.full((*shape, 3), bg, np.uint8)
    palm = np.linalg.norm(pts[9] - pts[0])
    poly = pts[[0, 1, 2, 5, 9, 13, 17]].astype(np.int32)
    cv2.fillPoly(img, [poly], skin)
    for f, ids in HAND_FINGER_IDS.items():
        if f in skip:
            continue
        chain = pts[list(ids)].astype(np.int32)
        cv2.polylines(img, [chain], False, skin, thickness=int(palm * 0.18))
        cv2.circle(img, tuple(chain[-1]), int(palm * 0.09), skin, -1)
    return img


def make_body(arm_ratio: float = 1.0, scale: float = 100.0, offset=(300.0, 80.0), vis: float = 0.95) -> Body:
    """正面站立的人体（受检者左侧在画面右侧）。arm_ratio 缩放左前臂长度，用来制造左右不对称。"""
    o = np.asarray(offset)
    j = {
        "nose": (0, 0), "left_eye": (0.1, -0.05), "right_eye": (-0.1, -0.05), "left_ear": (0.2, 0), "right_ear": (-0.2, 0),
        "left_shoulder": (0.6, 0.8), "right_shoulder": (-0.6, 0.8),
        "left_elbow": (0.7, 1.8), "right_elbow": (-0.7, 1.8),
        "left_wrist": (0.75, 1.8 + 0.9 * arm_ratio), "right_wrist": (-0.75, 2.7),
        "left_hip": (0.35, 2.6), "right_hip": (-0.35, 2.6),
        "left_knee": (0.38, 3.8), "right_knee": (-0.38, 3.8),
        "left_ankle": (0.4, 4.9), "right_ankle": (-0.4, 4.9),
    }
    return Body(joints={k: np.asarray(v, float) * scale + o for k, v in j.items()}, vis={k: vis for k in j}, score=vis)


class FakeKeypoints:
    """测试用关键点模型：直接返回给定的手和人体。"""

    name = "fake"

    def __init__(self, hands=None, bodies=None):
        self._hands = hands or []
        self._bodies = bodies or []

    def hands(self, img):
        return [Hand(pts=h.pts.copy(), score=h.score, side=h.side, side_score=h.side_score, source="fake") for h in self._hands]

    def bodies(self, img):
        return list(self._bodies)


@pytest.fixture
def thresholds():
    from limbcheck.config import load_thresholds

    return load_thresholds()
