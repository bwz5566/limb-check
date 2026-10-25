"""合成缺指样本：在健全手的图片上用图像修复（inpainting）抹掉一根手指或其远端，生成带真值的正样本。

用途：在凑齐真实缺指样本之前，先比较各模型对"手指不见了"的反应——尤其是关键点模型会不会把手指"脑补"回来、
轮廓复核和大模型能不能发现。合成图有修复痕迹，和真实残肢外观不同，结论只能作参考，
最终指标必须用真实样本（合作机构采集）验证。见 docs/VALIDATION.md 第 8 节。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .core import FINGERS, HAND_VIEWS
from .dataset import Sample, load_image, load_samples, save_image, write_jsonl
from .geometry import dist, palm_len
from .keypoints import HAND_FINGER_IDS, assign_sides
from .core import EXPECTED_HANDS

CUTS = ("mcp", "pip")   # mcp：整指缺失（从掌指关节起）；pip：远端两节缺失（从近端指间关节起，拇指为指间关节）


def finger_mask(shape, pts: np.ndarray, finger: str, cut: str) -> tuple[np.ndarray, np.ndarray]:
    """返回 (要抹掉的区域掩码, 切口位置)。"""
    import cv2

    ids = HAND_FINGER_IDS[finger]
    palm = palm_len(pts)
    width = palm * (0.26 if finger == "thumb" else 0.21)
    if finger == "thumb":
        start = pts[ids[1]] if cut == "mcp" else pts[ids[2]]
        chain = [start] + [pts[i] for i in ids[2 if cut == "mcp" else 3:]]
    else:
        mcp, pip = pts[ids[0]], pts[ids[1]]
        start = mcp + 0.18 * (pip - mcp) if cut == "mcp" else pip
        chain = [start] + [pts[i] for i in ids[1 if cut == "mcp" else 2:] if not np.allclose(pts[i], start)]
    tip, dip = pts[ids[3]], pts[ids[2]]
    chain.append(tip + 0.45 * (tip - dip))      # 指尖再往外延一点，盖住指甲和指腹边缘
    mask = np.zeros(shape[:2], np.uint8)
    poly = np.array(chain, dtype=np.int32)
    cv2.polylines(mask, [poly], False, 1, thickness=max(3, int(width)))
    cv2.circle(mask, tuple(int(v) for v in chain[-1]), max(2, int(width * 0.7)), 1, -1)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    return mask, np.asarray(start, dtype=float)


def remove_finger(img: np.ndarray, pts: np.ndarray, finger: str, cut: str = "mcp", th_contour: dict | None = None) -> np.ndarray:
    import cv2

    from .adapters.contour import SkinMask
    from .keypoints import Hand

    mask, _ = finger_mask(img.shape, pts, finger, cut)
    palm = palm_len(pts)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # 修复时只让背景参与：把掩码周围属于手（相邻手指）的像素先替换成背景色，避免把肤色"抹"进缺失区域
    skin = SkinMask(delta_e=35, dilate_px=2).hand_mask(img, Hand(pts=pts), th_contour or {})   # 放宽阈值，宁可多替换
    ring = cv2.dilate(mask, np.ones((int(palm * 0.25) | 1,) * 2, np.uint8)) & (1 - mask)
    bg_px = bgr[(ring > 0) & (skin == 0)]
    tmp = bgr.copy()
    if len(bg_px) > 20:
        bg = np.median(bg_px, axis=0)
        tmp[(ring > 0) & (skin > 0)] = bg.astype(np.uint8)
    filled = cv2.inpaint(tmp, mask * 255, max(3, int(palm * 0.06)), cv2.INPAINT_TELEA)
    if len(bg_px) > 20:   # 补一点亮度噪声，避免修复区域过于平滑
        sigma = min(4.0, float(np.std(bg_px.mean(axis=1))) * 0.5)
        noise = np.random.default_rng(0).normal(0, sigma, filled.shape[:2])[..., None]
        filled = np.clip(filled + noise * mask[..., None], 0, 255).astype(np.uint8)
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.5)[..., None]
    out = (bgr * (1 - soft) + filled * soft).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def synthesize(data_dir: str | Path, out_dir: str | Path, keypoints, fingers=FINGERS, cuts=CUTS,
               max_per_image: int = 4, keep_originals: bool = True, seed: int = 0) -> list[Sample]:
    """对数据集中真值为"手指完整"的手部图片生成缺指变体。"""
    rng = np.random.default_rng(seed)
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out: list[Sample] = []
    for s in load_samples(data_dir):
        if s.view not in HAND_VIEWS:
            continue
        truth_f = s.truth.get("fingers") or {}
        if any(truth_f.get(k) for k in ("right", "left")):
            continue   # 只在健全手上合成
        img = load_image(data_dir / s.image)
        hands = assign_sides(keypoints.hands(img), EXPECTED_HANDS[s.view])
        if s.view == "hand" and len(hands) == 1 and s.meta.get("side") in ("right", "left"):
            hands[0].side = s.meta["side"]
        hands = [h for h in hands if h.side in ("right", "left")]
        if not hands:
            continue
        base_truth = {**s.truth, "fingers": {"right": [], "left": []}}
        if keep_originals:
            rel = Path("images") / f"{s.id}__orig.jpg"
            save_image(out_dir / rel, img)
            out.append(Sample(id=f"{s.id}__orig", image=rel.as_posix(), view=s.view, session_id=f"{s.id}__orig",
                              subject_id=s.subject_id, truth=base_truth,
                              meta={**s.meta, "source": "synthetic", "synth": "original", "source_image": s.id}))
        combos = [(h, f, c) for h in hands for f in fingers for c in cuts]
        rng.shuffle(combos)
        for h, f, c in combos[:max_per_image]:
            if dist(h.pts[HAND_FINGER_IDS[f][3]], h.pts[0]) < 5:
                continue
            new = remove_finger(img, h.pts, f, c)
            sid = f"{s.id}__{h.side}_{f}_{c}"
            rel = Path("images") / f"{sid}.jpg"
            save_image(out_dir / rel, new)
            truth = {**s.truth, "fingers": {"right": [], "left": [], h.side: [f]}}
            out.append(Sample(id=sid, image=rel.as_posix(), view=s.view, session_id=sid, subject_id=s.subject_id,
                              truth=truth, meta={**s.meta, "source": "synthetic", "synth": f"{f}_{c}", "source_image": s.id}))
    write_jsonl(out_dir / "manifest.jsonl", [x.to_dict() for x in out])
    return out
