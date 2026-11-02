"""统一不同模型的关键点格式。

- 手：21 点，顺序与 MediaPipe Hands、COCO-WholeBody 手部一致（0 腕，1–4 拇指，5–8 食指，9–12 中指，13–16 无名指，17–20 小指）。
- 身体：按关节名存放（受检者本人的左右），来源可以是 MediaPipe Pose 33 点、COCO 17 点或 COCO-WholeBody 133 点。
坐标一律为像素。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HAND_FINGER_IDS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "little": (17, 18, 19, 20),
}

BODY_JOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
)
# COCO 17 点（也是 COCO-WholeBody 133 点的前 17 个）
COCO17 = dict(zip(BODY_JOINTS, range(17)))
# MediaPipe Pose 33 点
MP33 = {
    "nose": 0, "left_eye": 2, "right_eye": 5, "left_ear": 7, "right_ear": 8,
    "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16, "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26, "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30, "left_foot_index": 31, "right_foot_index": 32,
}
# COCO-WholeBody 133 点：身体 17 点 + 双脚（17 左大脚趾、19 左脚跟、20 右大脚趾、22 右脚跟）
WB133_BODY = {**COCO17, "left_foot_index": 17, "left_heel": 19, "right_foot_index": 20, "right_heel": 22}
# COCO-WholeBody 133 点中的双手（受检者本人的左右）
WB133_LEFT_HAND = slice(91, 112)
WB133_RIGHT_HAND = slice(112, 133)

LIMBS = {
    "left_arm": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_arm": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_leg": ("left_hip", "left_knee", "left_ankle"),
    "right_leg": ("right_hip", "right_knee", "right_ankle"),
}
LIMB_CN = {"left_arm": "左臂", "right_arm": "右臂", "left_leg": "左腿", "right_leg": "右腿"}
FEET = {"left_foot": ("left_heel", "left_foot_index"), "right_foot": ("right_heel", "right_foot_index")}
FOOT_CN = {"left_foot": "左脚", "right_foot": "右脚"}


@dataclass
class Hand:
    pts: np.ndarray                 # (21, 2) 像素坐标
    score: float = 1.0              # 手的置信度
    side: str | None = None         # right / left（受检者本人），None 表示未确定
    side_score: float = 0.0
    point_scores: np.ndarray | None = None   # (21,) 每点置信度（RTMPose 有，MediaPipe 无）
    source: str = ""

    @property
    def wrist_x(self) -> float:
        return float(self.pts[0, 0])


@dataclass
class Body:
    joints: dict[str, np.ndarray] = field(default_factory=dict)   # name -> (x, y)
    vis: dict[str, float] = field(default_factory=dict)            # name -> 可见度 / 置信度
    score: float = 1.0
    source: str = ""

    def has(self, name: str, min_vis: float = 0.0) -> bool:
        return name in self.joints and self.vis.get(name, 0.0) >= min_vis


def mirror_name(name: str) -> str:
    """left_xxx ↔ right_xxx。"""
    if name.startswith("left_"):
        return "right_" + name[5:]
    if name.startswith("right_"):
        return "left_" + name[6:]
    return name


def mirror_body(body: Body) -> Body:
    """镜像画面上，模型按外观给出的关节左右与受检者本人相反，交换回本人的左右（坐标不变）。"""
    return Body(joints={mirror_name(k): v for k, v in body.joints.items()}, vis={mirror_name(k): v for k, v in body.vis.items()},
                score=body.score, source=body.source)


def body_from_array(arr: np.ndarray, scores: np.ndarray, mapping: dict[str, int], source: str) -> Body:
    joints = {k: np.asarray(arr[i][:2], dtype=float) for k, i in mapping.items() if i < len(arr)}
    vis = {k: float(scores[i]) for k, i in mapping.items() if i < len(scores)}
    return Body(joints=joints, vis=vis, score=float(np.mean(list(vis.values()))) if vis else 0.0, source=source)


def assign_sides(hands: list[Hand], expected: int, mirrored: bool = False, min_side_score: float = 0.8) -> list[Hand]:
    """确定左右手（受检者本人的左右）。

    1. 模型给出了左右（MediaPipe handedness、RTMW 全身 133 点）且两只手不同、置信度都够 → 采用；
    2. 否则双手视角检测到 2 只手时按画面 x 坐标：未镜像画面里受检者的右手在画面左侧（双手交叉时会认错）。
    """
    if expected == 2 and len(hands) == 2:
        a, b = hands
        if a.side and b.side and a.side != b.side and min(a.side_score, b.side_score) >= min_side_score:
            return sorted(hands, key=lambda h: h.side != "right")
        ordered = sorted(hands, key=lambda h: h.wrist_x, reverse=mirrored)
        ordered[0].side, ordered[1].side = "right", "left"
        return ordered
    return hands
