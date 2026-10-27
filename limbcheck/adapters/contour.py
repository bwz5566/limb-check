"""手指轮廓复核：检查每根手指的指尖段关键点是否真的落在手部皮肤上。

关键点模型在手指缺失时常把指尖"脑补"到背景上；用一个独立的手部掩码交叉验证，
指尖段大部分落在掩码外即判该指"疑似缺失（关键点脑补）"。

后端：
- skin：以掌心颜色为参照的 Lab 色差掩码，纯 OpenCV、CPU 即可，背景接近肤色（木纹、米色墙）时会失效；
- mpseg：MediaPipe 多类别人像分割的"身体皮肤"类；
- sam2：SAM 2.1，以手腕和各 MCP 为正提示点，需要 torch 和权重（RTX 6000 上运行）。
"""
from __future__ import annotations

import numpy as np

from ..geometry import palm_len, rgb_to_lab
from ..keypoints import HAND_FINGER_IDS, Hand
from ..core import FINGER_CN, FINGERS, CheckResult, Finding

PALM_IDS = (0, 1, 5, 9, 13, 17)
SAMPLES_T = (0.2, 0.45, 0.7)   # DIP→指尖之间的采样位置（避开指尖边缘，缺指时这一段整体落在背景上）


def _roi(pts: np.ndarray, shape, pad: float) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    x0, y0 = pts.min(axis=0) - pad
    x1, y1 = pts.max(axis=0) + pad
    return max(0, int(x0)), max(0, int(y0)), min(w, int(x1) + 1), min(h, int(y1) + 1)


def _keep_palm_component(mask: np.ndarray, pts: np.ndarray) -> np.ndarray:
    import cv2

    n, labels = cv2.connectedComponents(mask.astype(np.uint8))
    if n <= 1:
        return mask
    keep = set()
    for i in PALM_IDS:
        x, y = int(round(pts[i][0])), int(round(pts[i][1]))
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and labels[y, x] > 0:
            keep.add(int(labels[y, x]))
    if not keep:
        return mask
    return np.isin(labels, list(keep)).astype(np.uint8)


def _ref_skin(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """参照肤色：掌面 7 个点 + 各指近节、中节中点，每个点取小块均值后再取中位数（每点一票）。
    只用掌心一块取色时，手掌缺一块就会把背景当成肤色；多点投票对局部缺失更稳。"""
    center = pts[list(PALM_IDS)].mean(axis=0)
    samples = [center] + [center + 0.5 * (pts[i] - center) for i in PALM_IDS]
    for ids in HAND_FINGER_IDS.values():
        samples += [(pts[ids[0]] + pts[ids[1]]) / 2, (pts[ids[1]] + pts[ids[2]]) / 2]
    h, w = img.shape[:2]
    r = max(2, int(0.03 * palm_len(pts)))
    cols = []
    for x, y in samples:
        x, y = int(round(x)), int(round(y))
        if 0 <= x < w and 0 <= y < h:
            cols.append(img[max(0, y - r): y + r + 1, max(0, x - r): x + r + 1].reshape(-1, 3).mean(axis=0))
    if not cols:
        return rgb_to_lab(np.array([[128.0, 128.0, 128.0]]))[0]
    return np.median(rgb_to_lab(np.array(cols)), axis=0)


class SkinMask:
    name = "skin"

    def __init__(self, delta_e: float | None = None, dilate_px: int | None = None):
        self.delta_e = delta_e
        self.dilate_px = dilate_px

    def hand_mask(self, img: np.ndarray, hand: Hand, th: dict) -> np.ndarray:
        import cv2

        pts = hand.pts
        palm = palm_len(pts)
        thr = self.delta_e or th.get("skin_delta_e", 20)
        ref = _ref_skin(img, pts)
        h, w = img.shape[:2]
        x0, y0, x1, y1 = _roi(pts, img.shape, 0.5 * palm)
        lab = rgb_to_lab(img[y0:y1, x0:x1])
        d = np.sqrt((0.5 * (lab[..., 0] - ref[0])) ** 2 + (lab[..., 1] - ref[1]) ** 2 + (lab[..., 2] - ref[2]) ** 2)
        mask = np.zeros((h, w), np.uint8)
        mask[y0:y1, x0:x1] = (d < thr).astype(np.uint8)
        k = max(3, int(palm * 0.04) | 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = _keep_palm_component(mask, pts)
        dil = self.dilate_px if self.dilate_px is not None else th.get("mask_dilate_px", 3)
        if dil:
            mask = cv2.dilate(mask, np.ones((2 * dil + 1, 2 * dil + 1), np.uint8))
        return mask


class SegmenterMask:
    name = "mpseg"

    def __init__(self, segmenter=None):
        if segmenter is None:
            from .segmenter import SelfieMulticlass
            segmenter = SelfieMulticlass()
        self.seg = segmenter

    def hand_mask(self, img: np.ndarray, hand: Hand, th: dict) -> np.ndarray:
        import cv2

        from .segmenter import BODY_SKIN

        mask = self.seg.mask(img, (BODY_SKIN,))
        mask = _keep_palm_component(mask, hand.pts)
        dil = th.get("mask_dilate_px", 3)
        return cv2.dilate(mask, np.ones((2 * dil + 1, 2 * dil + 1), np.uint8)) if dil else mask


class Sam2Mask:
    """SAM 2.1。checkpoint/config 不传时用 Hugging Face 上的 facebook/sam2.1-hiera-tiny（需能访问 HF）。"""

    name = "sam2"

    def __init__(self, device: str = "cuda", checkpoint: str | None = None, model_cfg: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
                 hf_id: str = "facebook/sam2.1-hiera-tiny"):
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if checkpoint:
            from sam2.build_sam import build_sam2
            self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))
        else:
            self.predictor = SAM2ImagePredictor.from_pretrained(hf_id, device=device)
        self._img = None

    def hand_mask(self, img: np.ndarray, hand: Hand, th: dict) -> np.ndarray:
        if self._img is not img:      # 持有图片引用：同一张图的多只手只编码一次，也不会因 id 复用取错
            self.predictor.set_image(img)
            self._img = img
        pts = hand.pts
        prompts = np.vstack([pts[[0, 5, 9, 13, 17]], pts[list(PALM_IDS)].mean(axis=0, keepdims=True)])
        masks, scores, _ = self.predictor.predict(point_coords=prompts.astype(np.float32),
                                                  point_labels=np.ones(len(prompts), dtype=np.int32),
                                                  multimask_output=True)
        return masks[int(np.argmax(scores))].astype(np.uint8)


def tip_coverage(mask: np.ndarray, pts: np.ndarray, finger: str, radius: int | None = None) -> float:
    """手指远端一节（DIP→指尖，拇指 IP→指尖）上采样点落在掩码内的比例。"""
    ids = HAND_FINGER_IDS[finger]
    if radius is None:
        radius = max(2, int(0.03 * palm_len(pts)))
    a, b = pts[ids[2]], pts[ids[3]]
    h, w = mask.shape[:2]
    hits = 0
    for t in SAMPLES_T:
        x, y = (a + t * (b - a)).round().astype(int)
        if 0 <= x < w and 0 <= y < h:
            hits += bool(mask[max(0, y - radius): y + radius + 1, max(0, x - radius): x + radius + 1].any())
    return hits / len(SAMPLES_T)


def palm_coverage(mask: np.ndarray, pts: np.ndarray, radius: int | None = None) -> float:
    """掌面内部采样点（掌心中心，以及中心到腕、各指根连线上 1/3、2/3 处，共 13 点）落在掩码内的比例。
    手掌或手背有大块残缺时关键点仍会给出完整的手形，这些点会落在背景上。"""
    if radius is None:
        radius = max(2, int(0.03 * palm_len(pts)))
    center = pts[list(PALM_IDS)].mean(axis=0)
    samples = [center] + [center + t * (pts[i] - center) for i in PALM_IDS for t in (0.33, 0.66)]
    h, w = mask.shape[:2]
    hits = 0
    for p in samples:
        x, y = np.round(p).astype(int)
        if 0 <= x < w and 0 <= y < h:
            hits += bool(mask[max(0, y - radius): y + radius + 1, max(0, x - radius): x + radius + 1].any())
    return hits / len(samples)


def contour_result(img: np.ndarray, hands: list[Hand], backend, th: dict, expected: int) -> tuple[CheckResult, dict]:
    """返回 (CheckResult, 每只手每根手指及掌面的覆盖率)。"""
    source = f"L1:contour-{backend.name}"
    thr = th.get("tip_on_mask_min", 0.5)
    palm_thr = th.get("palm_on_mask_min", 0.7)
    findings, parts, covs, detail = [], [], [], {}
    for hand in hands:
        side = hand.side or "unknown"
        mask = backend.hand_mask(img, hand, th)
        cov = {f: tip_coverage(mask, hand.pts, f) for f in FINGERS}
        pc = palm_coverage(mask, hand.pts)
        detail[side] = {**cov, "palm": pc}
        if pc < palm_thr:
            parts.append(f"{side}_hand.palm")
            findings.append(Finding("fingers", f"{side}_hand.palm", "palm_off_mask", True, round(0.5 + 0.5 * (1 - pc), 3), source, pc,
                                    f"掌面只有 {pc:.0%} 落在手部轮廓内，疑似手掌 / 手背残缺或被遮挡"))
        for f, c in cov.items():
            covs.append(c)
            if c < thr:
                parts.append(f"{side}_hand.{f}")
                findings.append(Finding("fingers", f"{side}_hand.{f}", "tip_off_mask", True, round(0.5 + 0.5 * (1 - c), 3), source, c,
                                        f"{FINGER_CN[f]}指尖段只有 {c:.0%} 落在手部轮廓内，疑似关键点脑补"))
    if parts:
        return CheckResult("fingers", source, True, max(f.confidence for f in findings), sorted(parts), True, findings), detail
    if len(hands) < expected or not covs:
        return CheckResult("fingers", source, None, 0.0, [], False, findings, f"只检测到 {len(hands)} 只手"), detail
    return CheckResult("fingers", source, False, round(0.5 + 0.5 * min(covs), 3), [], True, findings), detail
