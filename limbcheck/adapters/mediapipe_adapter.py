"""MediaPipe Tasks（Hand Landmarker + Pose Landmarker），与网页 demo 用同一组 .task 模型文件。"""
from __future__ import annotations

import numpy as np

from ..config import model_path
from ..keypoints import MP33, Body, Hand, mirror_body


class MediaPipeKeypoints:
    name = "mediapipe"

    def __init__(self, hand_model: str | None = None, pose_model: str | None = None, mirrored: bool = False,
                 num_hands: int = 2, num_poses: int = 2, min_hand_conf: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self.mirrored = mirrored
        self._hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(hand_model or model_path("hand_landmarker.task"))),
            running_mode=vision.RunningMode.IMAGE, num_hands=num_hands,
            min_hand_detection_confidence=min_hand_conf))
        self._pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(pose_model or model_path("pose_landmarker_full.task"))),
            running_mode=vision.RunningMode.IMAGE, num_poses=num_poses))

    def close(self) -> None:
        for attr in ("_hand", "_pose"):
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.close()
                setattr(self, attr, None)

    def _image(self, img: np.ndarray):
        return self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(img))

    def hands(self, img: np.ndarray) -> list[Hand]:
        h, w = img.shape[:2]
        r = self._hand.detect(self._image(img))
        out = []
        for i, lms in enumerate(r.hand_landmarks):
            pts = np.array([[p.x * w, p.y * h] for p in lms], dtype=float)
            cat = r.handedness[i][0] if r.handedness and r.handedness[i] else None
            label = (cat.category_name or "").lower() if cat else ""
            # 实测（tests/test_mediapipe_smoke.py）：Tasks 版 HandLandmarker 对未镜像照片给出的就是本人左右；
            # 输入为自拍镜像画面时才需要取反
            if label in ("left", "right") and self.mirrored:
                label = "right" if label == "left" else "left"
            out.append(Hand(pts=pts, score=float(cat.score) if cat else 1.0, side=label or None,
                            side_score=float(cat.score) if cat else 0.0, source=self.name))
        return out

    def bodies(self, img: np.ndarray) -> list[Body]:
        h, w = img.shape[:2]
        r = self._pose.detect(self._image(img))
        out = []
        for lms in r.pose_landmarks:
            arr = np.array([[p.x * w, p.y * h] for p in lms], dtype=float)
            vis = np.array([p.visibility if p.visibility is not None else 1.0 for p in lms], dtype=float)
            joints = {k: arr[i] for k, i in MP33.items()}
            v = {k: float(vis[i]) for k, i in MP33.items()}
            b = Body(joints=joints, vis=v, score=float(np.mean(list(v.values()))), source=self.name)
            out.append(mirror_body(b) if self.mirrored else b)
        out.sort(key=lambda b: b.score, reverse=True)
        return out
