"""MediaPipe 多类别人像分割（selfie_multiclass_256x256）：背景、头发、身体皮肤、面部皮肤、衣物、其他。

用途：发色基线（头发掩码 + Lab 颜色统计）、手指轮廓复核的 mpseg 后端（身体皮肤掩码）。
该模型为自拍场景训练，全身远景下头发掩码可能偏小，结果只作基线。
"""
from __future__ import annotations

import numpy as np

from ..config import model_path

BACKGROUND, HAIR, BODY_SKIN, FACE_SKIN, CLOTHES, OTHERS = range(6)


class SelfieMulticlass:
    name = "mpseg"

    def __init__(self, model: str | None = None):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self._seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(model or model_path("selfie_multiclass_256x256.tflite"))),
            running_mode=vision.RunningMode.IMAGE, output_category_mask=True, output_confidence_masks=False))
        self._cache: tuple[np.ndarray, np.ndarray] | None = None

    def close(self) -> None:
        if self._seg is not None:
            self._seg.close()
            self._seg = None

    def categories(self, img: np.ndarray) -> np.ndarray:
        """返回 (H, W) uint8 类别图。同一张图重复调用会复用结果。"""
        if self._cache and self._cache[0] is img:     # 缓存持有图片引用，不会因 id 复用而取错
            return self._cache[1]
        res = self._seg.segment(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)))
        cat = res.category_mask.numpy_view().copy()
        if cat.ndim == 3:
            cat = cat[..., 0]
        self._cache = (img, cat)
        return cat

    def mask(self, img: np.ndarray, classes: tuple[int, ...]) -> np.ndarray:
        return np.isin(self.categories(img), classes).astype(np.uint8)
