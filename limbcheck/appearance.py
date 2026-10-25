"""外观项的 L1 基线。

M0 阶段只有发色有一个可运行的启发式基线（头发分割 + Lab 彩度）；纹身、疤痕先只用 L2 大模型，
等采集标注到足够数据后再训练分割模型（docs/HANDOFF.md 5.4）。这里的结果只用于记录和对比基线，
不应单独作为任何结论的依据。
"""
from __future__ import annotations

import math

import numpy as np

from .geometry import rgb_to_lab
from .core import CheckResult, Finding


def hair_stats(img: np.ndarray, hair_mask: np.ndarray) -> dict | None:
    px = img[hair_mask > 0]
    if len(px) == 0:
        return None
    lab = rgb_to_lab(px)
    L, a, b = np.median(lab, axis=0)
    return {
        "pixels": int(len(px)),
        "L": round(float(L), 1),
        "a": round(float(a), 1),
        "b": round(float(b), 1),
        "chroma": round(float(math.hypot(a, b)), 1),
        "hue": round(float(math.degrees(math.atan2(b, a)) % 360), 1),
    }


def hair_result(img: np.ndarray, segmenter, th_hair: dict, source: str = "L1:mpseg-hair") -> CheckResult:
    from .adapters.segmenter import HAIR

    stats = hair_stats(img, segmenter.mask(img, (HAIR,)))
    if stats is None or stats["pixels"] < th_hair.get("min_pixels", 400):
        return CheckResult("hair", source, None, 0.0, [], False, [], "头发像素过少，无法判断")
    thr = th_hair["dyed_chroma_min"]
    c = stats["chroma"]
    positive = c >= thr
    conf = min(0.9, 0.5 + abs(c - thr) / (2 * thr))
    f = Finding("hair", "head", "dyed" if positive else "natural", positive, round(conf, 3), source, stats,
                f"头发中位彩度 {c}（阈值 {thr}），亮度 {stats['L']}")
    return CheckResult("hair", source, positive, round(conf, 3), ["head"] if positive else [], True, [f])
