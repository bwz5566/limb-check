"""模型适配器（内置节点注册在这里）。重依赖（mediapipe、rtmlib、torch/sam2）都在工厂函数内部按需导入，
没装的依赖只在真正用到对应节点时才报错。新增模型的方法见 limbcheck/registry.py 和 docs/EXTENDING.md。"""
from __future__ import annotations

from ..registry import register
from .base import KeypointModel  # noqa: F401


@register("keypoints", "mediapipe")
def _mediapipe(**kw):
    from .mediapipe_adapter import MediaPipeKeypoints
    return MediaPipeKeypoints(**kw)


@register("keypoints", "rtmw")
def _rtmw(**kw):
    from .rtmw_adapter import RTMWKeypoints
    return RTMWKeypoints(**kw)


@register("keypoints", "rtmw+hand")
def _rtmw_hand(**kw):
    from .rtmw_adapter import RTMWKeypoints
    return RTMWKeypoints(hand_model=True, **kw)


@register("contour", "skin")
def _skin(**kw):
    from .contour import SkinMask
    return SkinMask(**kw)


@register("contour", "mpseg")
def _mpseg_contour(**kw):
    from .contour import SegmenterMask
    return SegmenterMask(**kw)


@register("contour", "sam2")
def _sam2(**kw):
    from .contour import Sam2Mask
    return Sam2Mask(**kw)


@register("segmenter", "mpseg")
def _mpseg(**kw):
    from .segmenter import SelfieMulticlass
    return SelfieMulticlass(**kw)
