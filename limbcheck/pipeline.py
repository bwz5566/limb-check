"""评测流水线：对每张图跑 L1（按检测项注册的判定函数）和可选的 L2 大模型，再仲裁。

流水线由配置文件组装（configs/pipelines/*.yaml），每个节点都可替换：
    keypoints / contour / segmenter：模型节点（registry 中的名称、或 包.模块:类）
    l1：要启用的 L1 判定列表（registry kind=l1）
    vlm：L2 大模型（env 表示读环境变量 VLM_BASE_URL / VLM_MODEL；none 表示不用）
    plugins：额外导入的模块（自定义节点在模块里用 @register 注册）

每条预测记录（predictions.jsonl 一行）：
    {"id", "view", "session_id", "pipeline",
     "results": {check: {"l1": CheckResult, "l2": CheckResult, "decision", "shadow", "reason", "agree", "mode",
                         "shadow_first_attempt"}},
     "latency_ms": {"keypoints": .., "contour": .., "segment": .., "l2": ..}, "quality", "detail", "errors"}
"""
from __future__ import annotations

import base64
import os
import re
from pathlib import Path

import numpy as np
import yaml

from . import registry
from .arbiter import arbitrate
from .config import check_mode
from .core import HAND_VIEWS, VIEW_CHECKS, merge_any
from .geometry import mm_per_px_body, mm_per_px_hands, quality
from .l1checks import L1Context
from .prompts import l2_checks_for
from .vlm import VLMClient

DEFAULT_L1 = ["fingers.keypoints", "fingers.contour", "nails.delta_e", "limbs.pose", "hair.color"]


def encode_jpeg(img: np.ndarray, max_side: int = 1280, quality_: int = 88) -> str:
    import cv2

    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality_])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _expand_env(obj):
    """配置里的 ${VAR} / ${VAR:-默认值} 替换为环境变量。"""
    if isinstance(obj, str):
        return re.sub(r"\$\{(\w+)(?::-([^}]*))?\}", lambda m: os.environ.get(m.group(1), m.group(2) or ""), obj)
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    return obj


def load_pipeline_spec(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return _expand_env(yaml.safe_load(f) or {})


def make_vlm(spec) -> VLMClient | None:
    if spec in (None, "", "none", False):
        return None
    if spec == "env":
        return VLMClient.from_env()
    spec = dict(spec)
    if not spec.get("base_url") or not spec.get("model"):
        return None
    return VLMClient(spec.pop("base_url"), spec.pop("model"), **spec)


def _subject_patch(img: np.ndarray, ctx: L1Context) -> np.ndarray:
    """取受检部位（手部关键点 / 人体关节）周围的小块像素，用来判断受检部位本身是否过暗过亮。"""
    pts = []
    if "hands" in ctx.__dict__:          # 只有计算过关键点才用，不额外触发推理
        pts += [p for h in ctx.hands for p in h.pts]
    if "bodies" in ctx.__dict__ and ctx.bodies:
        pts += list(ctx.bodies[0].joints.values())
    h, w = img.shape[:2]
    r = max(2, int(min(h, w) * 0.01))
    patches = [img[max(0, int(y) - r): int(y) + r + 1, max(0, int(x) - r): int(x) + r + 1].reshape(-1, 3)
               for x, y in pts if 0 <= x < w and 0 <= y < h]
    patches = [p for p in patches if len(p)]
    return np.concatenate(patches)[None] if patches else img


class Pipeline:
    def __init__(self, thresholds: dict, keypoints=None, contour=None, segmenter=None, vlm=None,
                 profile: dict | None = None, mirrored: bool = False, l1: list[str] | None = None,
                 eval_attempt: int | None = None, name: str | None = None):
        self.th = thresholds
        self.kp = keypoints
        self.contour = contour
        self.seg = segmenter
        self.vlm = vlm
        self.profile = profile
        self.mirrored = mirrored
        self.l1_names = list(l1 if l1 is not None else DEFAULT_L1)
        self.l1_funcs = [(n, registry.get("l1", n), registry.meta("l1", n).get("check")) for n in self.l1_names]
        arb = thresholds.get("arbitration", {})
        # 离线评测默认按"最后一次尝试"仲裁（不能真的让受检者重做），另外统计首次尝试会触发重做的比例
        self.attempt = arb.get("max_redo", 1) if eval_attempt is None else eval_attempt
        self._name = name

    @classmethod
    def from_spec(cls, spec: dict, thresholds: dict, profile: dict | None = None, **overrides) -> "Pipeline":
        spec = {**spec, **{k: v for k, v in overrides.items() if v is not None}}
        registry.load_plugins(spec.get("plugins"))
        seg = registry.create("segmenter", spec.get("segmenter"))
        contour_spec = spec.get("contour")
        ctype = contour_spec.get("type") if isinstance(contour_spec, dict) else contour_spec
        contour = registry.create("contour", contour_spec, **({"segmenter": seg} if ctype == "mpseg" and seg else {}))
        kp_spec = spec.get("keypoints")
        if spec.get("mirrored") and kp_spec:
            kp_spec = {"type": kp_spec, "mirrored": True} if isinstance(kp_spec, str) else {**kp_spec, "mirrored": True}
        return cls(thresholds, keypoints=registry.create("keypoints", kp_spec), contour=contour, segmenter=seg,
                   vlm=make_vlm(spec.get("vlm")), profile=profile, mirrored=bool(spec.get("mirrored")),
                   l1=spec.get("l1"), name=spec.get("name"))

    @property
    def name(self) -> str:
        if self._name:
            return self._name + (f"+vlm-{self.vlm.model}" if self.vlm else "")
        parts = [self.kp.name if self.kp else "no-kp"]
        if self.contour:
            parts.append(f"contour-{self.contour.name}")
        if self.seg:
            parts.append(self.seg.name)
        if self.vlm:
            parts.append(f"vlm-{self.vlm.model}")
        return "+".join(parts)

    def describe(self) -> dict:
        return {"name": self.name, "keypoints": getattr(self.kp, "name", None),
                "contour": getattr(self.contour, "name", None), "segmenter": getattr(self.seg, "name", None),
                "l1": self.l1_names,
                "vlm": {"base_url": self.vlm.base_url, "model": self.vlm.model, "schema": self.vlm.use_schema} if self.vlm else None,
                "profile": self.profile["id"] if self.profile else None, "mirrored": self.mirrored, "eval_attempt": self.attempt}

    def enabled_checks(self, view: str) -> list[str]:
        return [c for c in VIEW_CHECKS.get(view, ()) if check_mode(self.profile, c) != "off"]

    def l2_meta(self, ctx: L1Context) -> dict:
        """给 L2 结果换算纹身 / 疤痕尺寸用的图像比例（手部视角按手掌，其余按上臂、前臂），以及岗位参数。"""
        from .marks import policy

        meta: dict = {"img_w": int(ctx.img.shape[1]), "img_h": int(ctx.img.shape[0]),
                      "params": {c: ctx.check_params(c) for c in ("tattoo", "scar")}}
        if ctx.keypoints is None or not set(VIEW_CHECKS.get(ctx.view, ())) & {"tattoo", "scar"}:
            return meta
        anatomy = policy()["measure"]["anatomy_mm"]
        try:
            if ctx.view in HAND_VIEWS:
                k, src = mm_per_px_hands(ctx.hands, anatomy), "手掌长宽"
            else:
                k, src = mm_per_px_body(ctx.bodies[0] if ctx.bodies else None, anatomy), "上臂 / 前臂长度"
        except Exception:   # 比例估不出来只影响尺寸换算（退回大模型目测），不影响其他判定
            k = None
        if k:
            meta.update(mm_per_px=k, scale_source=src)
        return meta

    def run(self, img: np.ndarray, view: str, meta: dict | None = None) -> dict:
        checks = self.enabled_checks(view)
        ctx = L1Context(img=img, view=view, th=self.th, profile=self.profile, meta=meta or {}, keypoints=self.kp,
                        contour=self.contour, segmenter=self.seg, mirrored=self.mirrored)
        errors: list[str] = []

        l1: dict = {}
        for c in checks:
            parts = []
            for name, fn, check in self.l1_funcs:
                if check != c:
                    continue
                try:
                    r = fn(ctx)
                except Exception as e:   # 单个节点失败不拖垮整条评测，记录后继续
                    errors.append(f"{name}: {type(e).__name__}: {e}")
                    continue
                if r is not None:
                    parts.append(r)
            if parts:
                l1[c] = parts[0] if len(parts) == 1 else merge_any(parts, "L1:" + "+".join(p.source.removeprefix("L1:") for p in parts))

        l2: dict = {}
        l2_checks = l2_checks_for(view, checks)
        if self.vlm and l2_checks:
            l2_meta = self.l2_meta(ctx)
            rv = self.vlm.review(view, l2_checks, [encode_jpeg(img)], l2_meta)
            l2 = rv.results
            ctx.latency["l2"] = rv.latency_ms
            errors += [f"L2: {e}" for e in rv.errors]
            ctx.detail["l2_raw"] = rv.text[:2000]
            ctx.detail["l2_scale"] = {k: v for k, v in l2_meta.items() if k != "params"}

        # 质量门控只看受检部位的亮度，避免白墙背景被误判为过亮；模糊阈值尚未实测校准，先记录不使用
        q = quality(img, self.th["quality"], _subject_patch(img, ctx))
        if any(i in q["issues"] for i in ("过暗", "过亮")):
            for r in l1.values():
                r.quality_ok = False

        results = {}
        arb_cfg = self.th.get("arbitration", {})
        for c in checks:
            a, b = l1.get(c), l2.get(c)
            if a is None and b is None:
                continue
            mode = check_mode(self.profile, c)
            final = arbitrate(c, a, b, arb_cfg, mode, self.attempt)
            first = arbitrate(c, a, b, arb_cfg, mode, 0)
            results[c] = {"l1": a.to_dict() if a else None, "l2": b.to_dict() if b else None,
                          **final.to_dict(), "shadow_first_attempt": first.shadow}
        return {"view": view, "pipeline": self.name, "results": results,
                "latency_ms": {k: round(v, 1) for k, v in ctx.latency.items()}, "quality": q,
                "detail": ctx.detail, "errors": errors}
