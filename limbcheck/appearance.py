"""外观项的 L1：发色、发长（configs/policies/hair.yaml）。

发色：MediaPipe 多类别分割取头发掩码（只保留受检者本人的头发），用面部（背面用脖子）肤色做白平衡，
每个头发像素按 Lab 的亮度 / 彩度 / 色相归类（黑、棕、金、红、紫、绿、蓝……），
再按"拦截颜色"的占比判整体染色 / 挑染 / 自然发色；占比落在阈值附近 → 需人工核验。
发长：发尾在"头顶 → 肩线"之间的位置；男生长发规则只在招聘方填了性别时生效（不从图像推断性别）。

每条 Finding.value 里有 verdict（pass / review / block）和命中的条款，仲裁用 policy 策略读取（与纹身、疤痕相同）。
纹身、疤痕先只用 L2 大模型，等采集标注到足够数据后再训练分割模型（docs/HANDOFF.md 5.4）。
"""
from __future__ import annotations

import json
import math
from functools import lru_cache

import numpy as np

from .core import CheckResult, Finding, config_dir
from .geometry import rgb_to_lab

PASS, REVIEW, BLOCK = "pass", "review", "block"

# ---------------- 标准（配置） ----------------

_OVERRIDE: dict | None = None


@lru_cache(maxsize=1)
def _load_default() -> dict:
    path = config_dir() / "policies" / "hair.yaml"
    try:
        import yaml
    except ImportError:   # 网页 demo 只用标准库：读 export-web-config 生成的 JSON
        web = config_dir().parent / "apps" / "web-demo" / "config.generated.json"
        return json.loads(web.read_text(encoding="utf-8"))["policies"]["hair"]
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def hair_policy() -> dict:
    return _OVERRIDE if _OVERRIDE is not None else _load_default()


def set_hair_policy(p: dict | None) -> None:
    """替换当前标准（测试用）；传 None 恢复为 configs/policies/hair.yaml。"""
    global _OVERRIDE
    _OVERRIDE = p


# ---------------- 颜色 ----------------

def lch(lab: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b = lab[..., 1], lab[..., 2]
    return lab[..., 0], np.hypot(a, b), np.degrees(np.arctan2(b, a)) % 360


def _in(v: np.ndarray, r) -> np.ndarray:
    lo, hi = r
    return (v >= lo) & (v <= hi)


def _in_hue(h: np.ndarray, r) -> np.ndarray:
    lo, hi = r
    return _in(h, r) if lo <= hi else (h >= lo) | (h <= hi)     # 跨 0 度，如 [340, 55]


def classify(lab: np.ndarray, colors: dict) -> np.ndarray:
    """每个像素归入 colors 里第一个满足条件的颜色，返回颜色序号（按 colors 的顺序）；都不满足的归最后一个。"""
    L, C, h = lch(lab)
    out = np.full(len(lab), len(colors) - 1, dtype=np.int16)
    todo = np.ones(len(lab), dtype=bool)
    for i, spec in enumerate(colors.values()):
        m = todo.copy()
        if "L" in spec:
            m &= _in(L, spec["L"])
        if "C" in spec:
            m &= _in(C, spec["C"])
        if "h" in spec:
            m &= _in_hue(h, spec["h"])
        out[m] = i
        todo &= ~m
    return out


def _lab_to_rgb(lab) -> np.ndarray:
    """单个 Lab → sRGB(0–255)，rgb_to_lab 的逆运算。"""
    L, a, b = lab
    fy = (L + 16) / 116
    fx, fz = fy + a / 500, fy - b / 200
    inv = lambda t: t ** 3 if t ** 3 > 0.008856 else (t - 16 / 116) / 7.787
    x, y, z = inv(fx) * 0.95047, inv(fy), inv(fz) * 1.08883
    rgb = np.array([3.2406 * x - 1.5372 * y - 0.4986 * z, -0.9689 * x + 1.8758 * y + 0.0415 * z,
                    0.0557 * x - 0.2040 * y + 1.0570 * z])
    rgb = np.clip(rgb, 0, 1)
    return 255 * np.where(rgb > 0.0031308, 1.055 * rgb ** (1 / 2.4) - 0.055, 12.92 * rgb)


def wb_gains(skin_rgb: np.ndarray, wb: dict) -> tuple[np.ndarray | None, dict]:
    """用参考皮肤估计白平衡：把皮肤中位色的色相转到 skin_hue（亮度、彩度不变），得到每个 RGB 通道的倍数。"""
    if not wb.get("enabled", True) or len(skin_rgb) < wb.get("min_skin_pixels", 300):
        return None, {"applied": False, "skin_pixels": int(len(skin_rgb))}
    med = np.median(skin_rgb.astype(float), axis=0)
    L, a, b = rgb_to_lab(med)
    c, hue = math.hypot(a, b), math.degrees(math.atan2(b, a)) % 360
    t = math.radians(wb.get("skin_hue", 48))
    target = _lab_to_rgb((L, max(c, 8.0) * math.cos(t), max(c, 8.0) * math.sin(t)))
    g = (target / np.maximum(med, 1.0)) ** wb.get("strength", 1.0)
    g = np.clip(g, 1 / wb.get("max_gain", 1.35), wb.get("max_gain", 1.35))
    g = g / g.mean()     # 只改色调，不改整体亮度
    return g, {"applied": True, "skin_pixels": int(len(skin_rgb)), "skin_hue": round(hue, 1), "skin_C": round(c, 1),
               "gains": [round(float(x), 3) for x in g]}


# ---------------- 掩码 ----------------

def _components(mask: np.ndarray):
    import cv2

    return cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)


def main_person(cat: np.ndarray, background: int) -> np.ndarray:
    """受检者本人：非背景区域里面积最大的连通块（背景里其他人的头发不算）。"""
    n, lab, stats, _ = _components(cat != background)
    if n <= 1:
        return np.zeros(cat.shape, dtype=bool)
    return lab == 1 + int(np.argmax(stats[1:, 4]))


def _erode(mask: np.ndarray, frac: float = 0.006) -> np.ndarray:
    """去掉边缘一圈（头发和皮肤、背景交界处的混合色），掩码太小就不腐蚀。"""
    import cv2

    k = max(3, int(round(frac * max(mask.shape))) | 1)
    er = cv2.erode(mask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    return er if er.sum() >= 0.4 * mask.sum() else mask


def head_box(face: np.ndarray, body, shape, min_face: int = 300, min_vis: float = 0.3) -> tuple[int, int, int, int] | None:
    """头部范围 (x0, y0, x1, y1)：有脸按面部皮肤外扩，看不到脸（背面）按双肩估算。"""
    h, w = shape[:2]
    if face.sum() >= min_face:
        ys, xs = np.nonzero(face)
        fw = xs.max() - xs.min() + 1
        x0, x1, y1 = xs.min() - 0.6 * fw, xs.max() + 0.6 * fw, ys.max()
    elif body is not None and all(body.has(f"{s}_shoulder", min_vis) for s in ("left", "right")):
        l, r = body.joints["left_shoulder"], body.joints["right_shoulder"]
        sw, cx = abs(float(l[0] - r[0])), float(l[0] + r[0]) / 2
        x0, x1, y1 = cx - 0.6 * sw, cx + 0.6 * sw, (float(l[1] + r[1])) / 2 - 0.1 * sw
    else:
        return None
    return max(0, int(x0)), 0, min(w, int(x1)), min(h, int(y1))


def hair_masks(img: np.ndarray, segmenter, body=None) -> dict:
    """受检者本人的头发掩码。

    染成亮色（金、粉、红）的头发，分割模型常归到"其他"类而不是"头发"类，所以头部范围内的"其他"也算头发；
    长发会垂到头部范围以外，取与头部范围相交的整个连通块。帽子、发箍等也会被算进来（引导时提示摘下帽子）。"""
    from .adapters.segmenter import BACKGROUND, BODY_SKIN, FACE_SKIN, HAIR, OTHERS

    cat = segmenter.categories(img)
    person = main_person(cat, BACKGROUND)
    face = (cat == FACE_SKIN) & person
    box = head_box(face, body, img.shape)
    if box is None:
        hair = (cat == HAIR) & person
    else:
        x0, y0, x1, y1 = box
        cand = np.isin(cat, (HAIR, OTHERS)) & person
        n, lab, _, _ = _components(cand)
        keep = np.unique(lab[y0:y1, x0:x1])
        hair = np.isin(lab, keep[keep > 0]) & ((cat == HAIR) | _in_box(cat.shape, box))
        # "其他"类只取头部范围内的（垂下的长发分割成"头发"类时整块保留）
    skin = face if face.sum() >= 300 else (cat == BODY_SKIN) & person    # 背面看不到脸：用脖子等身体皮肤
    if skin is not face and hair.any():                                   # 只取头发附近的身体皮肤（脖子），不要手臂
        ys = np.nonzero(hair)[0]
        lim = int(ys.max() + 0.6 * (ys.max() - ys.min() + 1))
        skin = skin.copy()
        skin[lim:] = False
    return {"hair": hair, "face": face, "skin": _erode(skin), "skin_from": "face" if skin is face else "body", "head_box": box,
            "person": person}


def _in_box(shape, box) -> np.ndarray:
    x0, y0, x1, y1 = box
    m = np.zeros(shape[:2], dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


# ---------------- 发色判定 ----------------

def _band(frac: float, thr: float, b: float) -> bool:
    return thr * (1 - b) <= frac < thr * (1 + b)


def color_eval(img: np.ndarray, masks: dict, pol: dict) -> dict | None:
    colors, dyed = pol["colors"], pol["dyed"]
    hair = _erode(masks["hair"])
    if hair.sum() < dyed.get("min_pixels", 400):
        return None
    px = img[hair].astype(float)
    gains, wb_info = wb_gains(img[masks["skin"]], pol.get("white_balance") or {})
    if gains is not None:
        px = np.clip(px * gains, 0, 255)
    cls = classify(rgb_to_lab(px), colors)
    keys = list(colors)
    frac = {k: float(np.mean(cls == i)) for i, k in enumerate(keys)}
    blocked = {k: f for k, f in frac.items() if colors[k].get("action") == BLOCK and f > 0}
    review_cls = {k: f for k, f in frac.items() if colors[k].get("action") == REVIEW and f > 0}
    fb, fr = sum(blocked.values()), sum(review_cls.values())
    top = max(blocked, key=blocked.get) if blocked else None
    whole, hl, b = dyed["whole_min"], dyed["highlights_min"], dyed.get("borderline", 0.4)
    cname = lambda k: colors[k].get("name", k)
    main = sorted(((cname(k), f) for k, f in frac.items() if f >= 0.05), key=lambda x: -x[1])
    desc = "、".join(f"{n} {f:.0%}" for n, f in main[:3])

    if fb >= whole:
        label, verdict, text = "dyed", colors[top].get("action", BLOCK), f"整体染色（{cname(top)}，占头发 {fb:.0%}）"
        margin = (fb - whole) / whole
    elif fb >= hl:
        label, verdict, text = "highlights", dyed.get("highlights_action", BLOCK), f"挑染 / 局部染色（{cname(top)}，占头发 {fb:.0%}）"
        margin = min(fb - hl, whole - fb) / hl
    else:
        label, verdict, text = "natural", PASS, f"自然发色（{desc}）"
        margin = (hl - fb) / hl
    # 阈值两侧结论不同时，落在阈值附近 → 人工核验（整体染色和挑染都拦截时，两者之间的阈值不用核验）
    whole_act = colors[top].get("action", BLOCK) if top else BLOCK
    near_whole = _band(fb, whole, b) and whole_act != dyed.get("highlights_action", BLOCK)
    if verdict != PASS and (near_whole or _band(fb, hl, b)):
        verdict, text = REVIEW, f"疑似{text}，占比接近阈值，需人工核验"
    elif verdict == PASS and _band(fb, hl, b):
        verdict, text = REVIEW, f"有少量{cname(top)}（{fb:.0%}），接近挑染阈值，需人工核验"
    if verdict == PASS and fr >= dyed.get("unknown_review", 0.35):
        verdict, text = REVIEW, f"部分头发颜色无法归类（{fr:.0%}），需人工核验"
    conf = min(dyed.get("max_confidence", 0.8) if verdict != PASS else 0.95, 0.55 + 0.4 * min(1.0, margin))
    return {"label": label, "verdict": verdict, "text": text, "confidence": round(conf, 3), "pixels": int(hair.sum()),
            "colors": {k: round(f, 3) for k, f in frac.items() if f > 0}, "blocked_fraction": round(fb, 3),
            "main": desc, "white_balance": wb_info}


# ---------------- 发长 ----------------

def shoulder_y_from_mask(person: np.ndarray, hair: np.ndarray) -> float | None:
    """看不到肩关节（背面时关键点模型常找不到人）时，从人体轮廓估算肩线：
    头部以下第一处人体宽度超过头宽 2 倍的位置（肩膀轮廓），再往下 0.15 个头宽（约为肩关节高度）。"""
    if person is None or not hair.any():
        return None
    ys, xs = np.nonzero(hair)
    top = ys.min()
    head = hair[top: top + max(1, int(0.6 * (ys.max() - top + 1)))]
    hw = max((np.ptp(np.nonzero(r)[0]) + 1 for r in head if r.any()), default=0)
    if hw < 10:
        return None
    rows = person.any(axis=1)
    for y in range(int(np.median(ys)), person.shape[0]):
        if not rows[y]:
            continue
        xs_ = np.nonzero(person[y])[0]
        if np.ptp(xs_) + 1 >= 2.0 * hw:
            return float(y + 0.15 * hw)
    return None


def length_eval(masks: dict, body, pol: dict, min_vis: float = 0.3) -> dict | None:
    """发尾在"头顶 → 肩线"之间的位置。肩线优先用肩关节关键点，没有时从人体轮廓估算；都没有返回 None。"""
    hair = masks["hair"]
    if hair.sum() < 50:
        return None
    if body is not None and all(body.has(f"{s}_shoulder", min_vis) for s in ("left", "right")):
        sh, sh_from = float((body.joints["left_shoulder"][1] + body.joints["right_shoulder"][1]) / 2), "keypoints"
    else:
        sh, sh_from = shoulder_y_from_mask(masks.get("person"), hair), "silhouette"
        if sh is None:
            return None
    ys = np.nonzero(hair)[0]
    top, bottom = float(ys.min()), float(np.percentile(ys, 99.5))
    top_from = "hair"
    if top <= 1:      # 头顶出了画面：按眼睛位置估算（眼睛到头顶约为眼睛到肩线的 0.55 倍）
        eyes = [body.joints[e][1] for e in ("left_eye", "right_eye") if body is not None and body.has(e, min_vis)]
        if not eyes:
            return {"ratio": None, "note": "头顶不在画面内，无法测量发长"}
        ey = float(np.mean(eyes))
        top, top_from = ey - 0.55 * (sh - ey), "eyes"
    if sh - top < 10:
        return None
    ratio = (bottom - top) / (sh - top)
    lv = pol["levels"]
    level = "short" if ratio < lv["short"] else "collar" if ratio < lv["collar"] else "shoulder" if ratio < lv["shoulder"] else "long"
    return {"ratio": round(ratio, 3), "level": level, "level_name": pol["names"].get(level, level), "top_from": top_from,
            "shoulder_from": sh_from}


def length_verdict(info: dict | None, pol: dict, gender: str | None) -> tuple[str, str, float] | None:
    rule = (pol.get("rules") or {}).get(gender or "")
    if not rule:
        return None
    if not info or info.get("ratio") is None:
        return REVIEW, f"{(info or {}).get('note') or '看不到双肩或头发，无法测量发长'}，需人工核验", 0.5
    r, mx, b = info["ratio"], rule["max_ratio"], rule.get("borderline", 0.1)
    if r >= mx * (1 + b):
        return rule.get("action", BLOCK), f"{rule.get('text', '发长不符合')}：{info['level_name']}（{r:.2f}）", min(0.8, 0.55 + (r / mx - 1))
    if r >= mx * (1 - b):
        return REVIEW, f"发长接近标准（{info['level_name']}，{r:.2f}），需人工核验", 0.5
    return PASS, f"发长符合（{info['level_name']}）", min(0.95, 0.6 + (1 - r / mx))


# ---------------- 汇总 ----------------

def hair_result(img: np.ndarray, segmenter, body=None, meta: dict | None = None, pol: dict | None = None,
                source: str = "L1:mpseg-hair") -> CheckResult:
    pol = pol or hair_policy()
    meta = meta or {}
    gender = meta.get("gender") if meta.get("gender") in ("male", "female") else None
    masks = hair_masks(img, segmenter, body)
    findings: list[Finding] = []

    col = color_eval(img, masks, pol)
    if col is not None:
        findings.append(Finding("hair", "head", col["label"], col["verdict"] != PASS, col["confidence"], source,
                                {"kind": "hair_color", "verdict": col["verdict"], **{k: v for k, v in col.items() if k not in ("label", "verdict", "confidence")}},
                                col["text"]))
    lpol = pol.get("length") or {}
    info = length_eval(masks, body, lpol) if lpol else None
    lv = length_verdict(info, lpol, gender) if lpol else None
    if lv is not None:
        verdict, text, conf = lv
        findings.append(Finding("hair", "head", "long" if verdict != PASS else "length_ok", verdict != PASS, round(conf, 3), source,
                                {"kind": "hair_length", "verdict": verdict, "gender": gender, **(info or {})}, text))
    elif info and info.get("ratio") is not None:   # 没填性别：只记录发长，不判定
        findings.append(Finding("hair", "head", "length", None, 0.0, source, {"kind": "hair_length", **info},
                                f"发长：{info['level_name']}（{info['ratio']:.2f}，{'未填写性别' if not gender else '无发长要求'}，只记录）"))

    if col is None:
        return CheckResult("hair", source, None, 0.0, [], False, findings, "头发像素过少或被遮挡，无法判断发色")
    judged = [f for f in findings if isinstance(f.value, dict) and f.value.get("verdict")]
    bad = [f for f in judged if f.value["verdict"] != PASS]
    note = "；".join(f.note for f in findings if f.note)
    if bad:
        return CheckResult("hair", source, True, round(max(f.confidence for f in bad), 3), ["head"], True, findings, note)
    return CheckResult("hair", source, False, round(min(f.confidence for f in judged), 3), [], True, findings, note)
