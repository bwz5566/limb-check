"""RTMW（COCO-WholeBody 133 点）与 RTMPose-hand，通过 rtmlib + ONNX Runtime 运行。

安装：pip install -e ".[rtmw]"（CPU）或 pip install rtmlib onnxruntime-gpu（RTX 6000）。
首次运行会从 download.openmmlab.com 下载 ONNX 模型到 ~/.cache/rtmlib/hub；
无法联网时把 zip 解压后用 det= / pose= / hand_det= / hand_pose= 传本地 .onnx 路径。
"""
from __future__ import annotations

import numpy as np

from ..keypoints import COCO17, WB133_LEFT_HAND, WB133_RIGHT_HAND, Body, Hand, body_from_array, mirror_body

HAND_MIN_SCORE = 0.3   # 手部 21 点平均置信度低于此值视为没看到这只手
OTHER_SIDE = {"left": "right", "right": "left"}


class RTMWKeypoints:
    def __init__(self, device: str = "cpu", mode: str = "balanced", hand_model: bool = False,
                 det: str | None = None, pose: str | None = None, hand_det: str | None = None,
                 hand_pose: str | None = None, backend: str = "onnxruntime", mirrored: bool = False):
        from rtmlib import Hand as RtmHand
        from rtmlib import Wholebody

        self.name = "rtmw+hand" if hand_model else "rtmw"
        self.mirrored = mirrored   # 输入为自拍镜像画面时，把模型给出的左右换回受检者本人的左右
        kw = {"backend": backend, "device": device, "mode": mode}
        if det:
            kw["det"] = det
        if pose:
            kw["pose"] = pose
        self._wb = Wholebody(**kw)
        self._hand = None
        if hand_model:
            hk = {"backend": backend, "device": device}
            if hand_det:
                hk["det"] = hand_det
            if hand_pose:
                hk["pose"] = hand_pose
            self._hand = RtmHand(**hk)

    def _wholebody(self, img: np.ndarray):
        # rtmlib 期望 BGR
        kpts, scores = self._wb(np.ascontiguousarray(img[..., ::-1]))
        return np.asarray(kpts), np.asarray(scores)

    def bodies(self, img: np.ndarray) -> list[Body]:
        kpts, scores = self._wholebody(img)
        out = [body_from_array(k, s, COCO17, self.name) for k, s in zip(kpts, scores)]
        if self.mirrored:
            out = [mirror_body(b) for b in out]
        out.sort(key=lambda b: b.score, reverse=True)
        return out

    def hands(self, img: np.ndarray) -> list[Hand]:
        if self._hand is not None:
            kpts, scores = self._hand(np.ascontiguousarray(img[..., ::-1]))
            out = []
            for k, s in zip(np.asarray(kpts), np.asarray(scores)):
                if float(np.mean(s)) >= HAND_MIN_SCORE:
                    # 手部近景模型不区分左右，交给 assign_sides 按画面位置确定
                    out.append(Hand(pts=k[:, :2].astype(float), score=float(np.clip(np.mean(s), 0, 1)),
                                    point_scores=s, source=self.name))
            return out
        kpts, scores = self._wholebody(img)
        if len(kpts) == 0:
            return []
        i = int(np.argmax(scores[:, :17].mean(axis=1)))   # 取主要人物
        out = []
        for side, sl in (("left", WB133_LEFT_HAND), ("right", WB133_RIGHT_HAND)):
            s = scores[i, sl]
            if float(np.mean(s)) >= HAND_MIN_SCORE:
                out.append(Hand(pts=kpts[i, sl, :2].astype(float), score=float(np.clip(np.mean(s), 0, 1)),
                                side=OTHER_SIDE[side] if self.mirrored else side, side_score=1.0,
                                point_scores=s, source=self.name))
        return out
