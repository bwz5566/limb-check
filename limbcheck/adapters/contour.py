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

from ..geometry import angle, palm_len, rgb_to_lab
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


GAP_PAIRS = (("index", "middle"), ("middle", "ring"), ("ring", "little"))


def tip_coverage(mask: np.ndarray, pts: np.ndarray, finger: str, radius: int | None = None, segment: int = 2) -> float:
    """手指一节上采样点落在掩码内的比例。segment=2：远端一节（DIP→指尖，拇指 IP→指尖）；segment=1：中节（PIP→DIP）。"""
    ids = HAND_FINGER_IDS[finger]
    if radius is None:
        radius = max(2, int(0.03 * palm_len(pts)))
    a, b = pts[ids[segment]], pts[ids[segment + 1]]
    h, w = mask.shape[:2]
    hits = 0
    for t in SAMPLES_T:
        x, y = (a + t * (b - a)).round().astype(int)
        if 0 <= x < w and 0 <= y < h:
            hits += bool(mask[max(0, y - radius): y + radius + 1, max(0, x - radius): x + radius + 1].any())
    return hits / len(SAMPLES_T)


def finger_straight(pts: np.ndarray) -> dict[str, bool]:
    return {f: angle(*(pts[i] for i in (HAND_FINGER_IDS[f][0], HAND_FINGER_IDS[f][1], HAND_FINGER_IDS[f][3]))) > 145 for f in FINGERS}


def gap_contrast(lab: np.ndarray, pa: np.ndarray, pb: np.ndarray, n: int = 24) -> float | None:
    """两指末节中点连线上，中间一半与两端（手指本身）的最大色差（亮度权重 0.5）。
    背景颜色接近肤色（脸、米色墙、浅色衣服）时掩码分不出缝隙，但缝隙处的明暗、颜色仍与两侧手指不同；
    被同色物体挡住或关键点画在一整块皮肤上时，整条线颜色几乎不变。"""
    h, w = lab.shape[:2]
    pts = [np.round(pa + t * (pb - pa)).astype(int) for t in np.linspace(0, 1, n)]
    if not all(0 <= x < w and 0 <= y < h for x, y in pts):
        return None
    line = np.array([lab[y, x] for x, y in pts], dtype=float)
    ends = np.vstack([line[:3], line[-3:]]).mean(axis=0)
    mid = line[n // 4: 3 * n // 4] - ends
    return float(np.sqrt((0.5 * mid[:, 0]) ** 2 + mid[:, 1] ** 2 + mid[:, 2] ** 2).max())


def finger_gaps(mask: np.ndarray, pts: np.ndarray, min_spread: float = 0.33, lab: np.ndarray | None = None,
                contrast_min: float | None = None) -> dict[str, bool | str | None]:
    """相邻两指之间有没有缝隙：两指都伸直、末节中点距离 ≥ min_spread 掌长（已分开）时，两指之间应有点落在掩码外，
    或者（传了 lab 时）两指之间的颜色明显不同于两侧手指（色差 ≥ contrast_min，背景接近肤色时靠这一条）。
    True：有缝隙；False：被别的手、肤色物体挡住，或关键点画在了别处的皮肤上（之间也是"皮肤"）；
    "closed"：两指伸直但并拢，无法确认（被遮挡的手指可能"藏"在相邻手指上）；None：有手指蜷起，不判断。"""
    palm = palm_len(pts)
    h, w = mask.shape[:2]
    out = {}
    straight = finger_straight(pts)
    for a, b in GAP_PAIRS:
        if not (straight[a] and straight[b]):        # 只在两指都伸直时判断（蜷起的手指压在掌面上，中间本来就是皮肤）
            out[f"{a}-{b}"] = None
            continue
        pa = (pts[HAND_FINGER_IDS[a][2]] + pts[HAND_FINGER_IDS[a][3]]) / 2
        pb = (pts[HAND_FINGER_IDS[b][2]] + pts[HAND_FINGER_IDS[b][3]]) / 2
        if np.linalg.norm(pa - pb) < min_spread * palm:
            out[f"{a}-{b}"] = "closed"
            continue
        vals = []
        for t in (0.35, 0.5, 0.65):           # 两指之间取 3 点，任一点不是皮肤即有缝隙（手指稍有歪斜也能找到）
            x, y = np.round(pa + t * (pb - pa)).astype(int)
            if 0 <= x < w and 0 <= y < h:
                vals.append(bool(mask[y, x]))
        seen = None if not vals else not all(vals)
        if not seen and lab is not None and contrast_min is not None:
            c = gap_contrast(lab, pa, pb)
            if c is not None and c >= contrast_min:
                seen = True
        out[f"{a}-{b}"] = seen
    return out


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


def contour_result(img: np.ndarray, hands: list[Hand], backend, th: dict, expected: int,
                   open_hand: bool = False) -> tuple[CheckResult, dict]:
    """返回 (CheckResult, 每只手每根手指及掌面的覆盖率)。
    open_hand：张开五指的视角（掌心、手背）。此时有手指蜷起或并拢就无法逐指确认（被遮挡、缺失的手指关键点会"藏"在掌面或相邻手指上），
    没有发现问题时也不给"正常"，结果为"无法判断"（画质仍算合格）：其余来源够确定时按其余来源处理，否则交人工。"""
    source = f"L1:contour-{backend.name}"
    thr = th.get("tip_on_mask_min", 0.5)
    palm_thr = th.get("palm_on_mask_min", 0.7)
    findings, parts, covs, detail, unverified = [], [], [], {}, []
    lab = rgb_to_lab(img.astype(float)) if hands and th.get("gap_delta_e") else None
    for hand in hands:
        side = hand.side or "unknown"
        mask = backend.hand_mask(img, hand, th)
        # 末节、中节都要落在掩码内：手指蜷起、被遮挡时关键点仍画出伸直的手指，其中一节会落到背景上
        cov = {f: min(tip_coverage(mask, hand.pts, f), tip_coverage(mask, hand.pts, f, segment=1)) for f in FINGERS}
        pc = palm_coverage(mask, hand.pts)
        gaps = finger_gaps(mask, hand.pts, th.get("gap_min_spread", 0.33), lab, th.get("gap_delta_e"))
        detail[side] = {**cov, "palm": pc, "gaps": gaps}
        if pc < palm_thr:
            parts.append(f"{side}_hand.palm")
            findings.append(Finding("fingers", f"{side}_hand.palm", "palm_off_mask", True, round(0.5 + 0.5 * (1 - pc), 3), source, pc,
                                    f"掌面只有 {pc:.0%} 落在手部轮廓内，疑似手掌 / 手背残缺或被遮挡"))
        for f, c in cov.items():
            covs.append(c)
            if c < thr:
                parts.append(f"{side}_hand.{f}")
                findings.append(Finding("fingers", f"{side}_hand.{f}", "tip_off_mask", True, round(0.5 + 0.5 * (1 - c), 3), source, c,
                                        f"{FINGER_CN[f]}只有 {c:.0%} 落在手部轮廓内，疑似被遮挡、蜷起或缺失（关键点脑补）"))
        for pair, ok in gaps.items():
            if ok is False:
                a, b = pair.split("-")
                parts += [p for p in (f"{side}_hand.{a}", f"{side}_hand.{b}") if p not in parts]
                findings.append(Finding("fingers", f"{side}_hand.{a}", "no_gap", True, 0.6, source, pair,
                                        f"{FINGER_CN[a]}和{FINGER_CN[b]}之间看不到缝隙，疑似被遮挡（关键点脑补）"))
        if open_hand:
            bent = [f for f, ok in finger_straight(hand.pts).items() if not ok]
            closed = [p for p, ok in gaps.items() if ok == "closed"]
            if bent or closed:
                unverified.append(side)
                findings.append(Finding("fingers", f"{side}_hand", "uncertain", False, 0.0, source, {"bent": bent, "closed": closed},
                                        "有手指蜷起或并拢，无法逐指确认每根手指都在"))
    if parts:
        return CheckResult("fingers", source, True, max(f.confidence for f in findings), sorted(set(parts)), True, findings), detail
    if unverified:
        # 只是这一路无法判断，不是画质问题：其余来源（大模型）够确定时按其余来源处理，不再一律转人工
        return CheckResult("fingers", source, None, 0.0, [], True, findings, "有手指蜷起或并拢，无法逐指确认每根手指都在"), detail
    if len(hands) < expected or not covs:
        return CheckResult("fingers", source, None, 0.0, [], False, findings, f"只检测到 {len(hands)} 只手"), detail
    return CheckResult("fingers", source, False, round(0.5 + 0.5 * min(covs), 3), [], True, findings), detail
