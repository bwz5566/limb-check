from __future__ import annotations

from typing import Protocol

import numpy as np

from ..keypoints import Body, Hand


class KeypointModel(Protocol):
    """关键点模型接口。输入 RGB uint8 图像（H, W, 3），输出像素坐标。"""

    name: str

    def hands(self, img: np.ndarray) -> list[Hand]:
        ...

    def bodies(self, img: np.ndarray) -> list[Body]:
        ...
