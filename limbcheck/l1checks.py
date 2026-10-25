"""内置 L1 判定。每个判定函数按检测项注册，流水线对每个启用的检测项依次调用并合并结果
（任一路报问题即报问题，见 core.merge_any）。

新增 / 替换判定：在自己的模块里写

    from limbcheck.registry import register

    @register("l1", "tattoo.seg", check="tattoo")
    def tattoo_seg(ctx):            # ctx: L1Context
        ...
        return CheckResult("tattoo", "L1:tattoo-seg", positive, confidence, parts)

再在流水线配置的 `l1:` 列表里加上 "tattoo.seg"，`plugins:` 里加上模块名。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from .appearance import hair_result
from .core import CheckResult, EXPECTED_HANDS
from .geometry import fingers_result, limbs_result, nails_result
from .keypoints import assign_sides
from .registry import register


@dataclass
class L1Context:
    """一次判定的上下文：图片、视角、阈值、岗位画像、已加载的模型。关键点按需计算并缓存。"""

    img: np.ndarray
    view: str
    th: dict
    profile: dict | None = None
    meta: dict = field(default_factory=dict)
    keypoints: object = None
    contour: object = None
    segmenter: object = None
    mirrored: bool = False
    latency: dict = field(default_factory=dict)
    detail: dict = field(default_factory=dict)

    def _timed(self, key: str, fn):
        t0 = time.perf_counter()
        out = fn()
        self.latency[key] = self.latency.get(key, 0.0) + (time.perf_counter() - t0) * 1000
        return out

    @property
    def expected_hands(self) -> int:
        return EXPECTED_HANDS.get(self.view, 2)

    @cached_property
    def hands(self) -> list:
        if self.keypoints is None:
            return []
        hands = assign_sides(self._timed("keypoints", lambda: self.keypoints.hands(self.img)), self.expected_hands, self.mirrored)
        if self.view == "hand" and self.meta.get("side") in ("right", "left") and len(hands) == 1:
            hands[0].side = self.meta["side"]
        self.detail["hands"] = [{"side": h.side, "score": round(h.score, 3)} for h in hands]
        return hands

    @cached_property
    def bodies(self) -> list:
        if self.keypoints is None:
            return []
        bodies = self._timed("keypoints", lambda: self.keypoints.bodies(self.img))
        self.detail["persons"] = len(bodies)
        return bodies

    def check_params(self, check: str) -> dict:
        if not self.profile:
            return {}
        return self.profile["checks"].get(check, {}).get("params", {}) or {}


@register("l1", "fingers.keypoints", check="fingers", needs="keypoints")
def fingers_keypoints(ctx: L1Context) -> CheckResult | None:
    """手指链长比法（与网页 demo 相同）。"""
    if ctx.keypoints is None:
        return None
    return fingers_result(ctx.hands, ctx.th["fingers"], ctx.expected_hands, f"L1:{ctx.keypoints.name}")


@register("l1", "fingers.contour", check="fingers", needs="contour")
def fingers_contour(ctx: L1Context) -> CheckResult | None:
    """指尖段是否落在手部掩码内，用来发现关键点"脑补"出的手指。"""
    if ctx.contour is None or not ctx.hands:
        return None
    from .adapters.contour import contour_result

    res, cov = ctx._timed("contour", lambda: contour_result(ctx.img, ctx.hands, ctx.contour, ctx.th["contour"], ctx.expected_hands))
    ctx.detail["tip_coverage"] = cov
    return res


@register("l1", "nails.delta_e", check="nails", needs="keypoints")
def nails_delta_e(ctx: L1Context) -> CheckResult | None:
    """指甲与手背肤色的 Lab 色差，只对手背朝向镜头的手判断。"""
    if ctx.keypoints is None:
        return None
    return nails_result(ctx.img, ctx.hands, ctx.th["nails"], "L1:deltaE", ctx.mirrored)


@register("l1", "limbs.pose", check="limbs", needs="keypoints")
def limbs_pose(ctx: L1Context) -> CheckResult | None:
    """四肢关节可见度 + 左右肢段长度比。下肢只在正面站立视角评估。"""
    if ctx.keypoints is None:
        return None
    p = ctx.check_params("limbs")
    bodies = ctx.bodies
    return limbs_result(bodies[0] if bodies else None, ctx.th["pose"], f"L1:{ctx.keypoints.name}",
                        upper=p.get("upper", True), lower=p.get("lower", True) and ctx.view == "stand",
                        image_size=(ctx.img.shape[1], ctx.img.shape[0]))


@register("l1", "hair.color", check="hair", needs="segmenter")
def hair_color(ctx: L1Context) -> CheckResult | None:
    """头发分割 + Lab 彩度（粗略基线，只作记录和对比）。"""
    if ctx.segmenter is None:
        return None
    return ctx._timed("segment", lambda: hair_result(ctx.img, ctx.segmenter, ctx.th["hair"]))
