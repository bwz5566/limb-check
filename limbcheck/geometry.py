"""L1 几何与颜色判定，逻辑与网页 demo（apps/web-demo/app.js）一致，便于离线复算和调阈值。"""
from __future__ import annotations

import math

import numpy as np

from .keypoints import HAND_FINGER_IDS, LIMB_CN, LIMBS, Body
from .core import FINGER_CN, FINGERS, CheckResult, Finding


def dist(a, b) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def angle(a, b, c) -> float:
    """∠abc，单位度。"""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (math.hypot(*v1) * math.hypot(*v2) + 1e-6)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def path_len(pts, ids) -> float:
    return sum(dist(pts[ids[i]], pts[ids[i + 1]]) for i in range(len(ids) - 1))


# ---------------- 手 ----------------

def palm_len(pts) -> float:
    return dist(pts[0], pts[9])


def finger_extended(pts, finger: str, th: dict) -> bool:
    a, b, c, d = HAND_FINGER_IDS[finger]
    if finger == "thumb":
        return angle(pts[a], pts[b], pts[d]) > th.get("thumb_extended_angle", 140) and dist(pts[d], pts[5]) > 0.45 * palm_len(pts)
    return angle(pts[a], pts[b], pts[d]) > th.get("extended_angle", 145) and dist(pts[0], pts[d]) > dist(pts[0], pts[b]) * 1.1


def extended_count(pts, th: dict) -> int:
    return sum(finger_extended(pts, f, th) for f in FINGERS)


def orient_sign(pts) -> int:
    """叉积 (lm5−lm0)×(lm17−lm0) 的符号；掌心 / 手背翻转时符号翻转。"""
    v1 = pts[5] - pts[0]
    v2 = pts[17] - pts[0]
    return int(np.sign(v1[0] * v2[1] - v1[1] * v2[0]))


def facing(pts, side: str | None, mirrored: bool = False) -> str | None:
    """判断掌心还是手背朝向镜头。

    由手的手性决定（与手在画面内怎么转无关）：图像坐标 y 向下、未镜像画面中，
    右手手背朝前时 orient_sign = +1、掌心朝前时 = −1；左手相反；镜像画面再取反。
    （已用 MediaPipe 官方测试图 right_hands.jpg / left_hands.jpg 等核对，见 tests/test_mediapipe_smoke.py）
    """
    if side not in ("right", "left"):
        return None
    s = orient_sign(pts) * (1 if side == "right" else -1) * (-1 if mirrored else 1)
    return "back" if s > 0 else "palm" if s < 0 else None


def analyze_fingers(pts, th: dict) -> dict:
    """手指链长比法：各指链长 / 中指链长 低于 参考 × short_factor 记疑似缺失或短缩。"""
    ref, factor = th["ref"], th["short_factor"]
    mid = path_len(pts, HAND_FINGER_IDS["middle"])
    palm = palm_len(pts)
    out, margins = {}, []
    for f in FINGERS:
        ratio = path_len(pts, HAND_FINGER_IDS[f]) / (mid + 1e-6)
        limit = ref[f] * factor
        short = ratio < limit
        margins.append(ratio / limit - 1)
        out[f] = {"ratio": round(ratio, 3), "extended": finger_extended(pts, f, th), "short": short}
    mid_to_palm = mid / (palm + 1e-6)
    if mid_to_palm < th["middle_to_palm_min"]:
        out["middle"]["short"] = True
        margins.append(mid_to_palm / th["middle_to_palm_min"] - 1)
    return {
        "fingers": out,
        "mid_to_palm": round(mid_to_palm, 3),
        "extended": sum(v["extended"] for v in out.values()),
        "flagged": [f for f in FINGERS if out[f]["short"]],
        "margin": round(min(margins), 3),
    }


def fingers_result(hands: list, th: dict, expected: int, source: str) -> CheckResult:
    """关键点长度比 → 手指检测项的 L1 结果。hands 为已确定左右的 Hand 列表。"""
    findings, parts, confs = [], [], []
    for h in hands:
        a = analyze_fingers(h.pts, th)
        side = h.side or "unknown"
        for f in a["flagged"]:
            parts.append(f"{side}_hand.{f}")
            findings.append(Finding("fingers", f"{side}_hand.{f}", "missing_or_short", True,
                                    round(min(1.0, 0.5 - a["fingers"][f]["ratio"] / (th["ref"][f] * th["short_factor"]) + 1), 3),
                                    source, a["fingers"][f]["ratio"], f"{FINGER_CN[f]}长度比 {a['fingers'][f]['ratio']}"))
        if not a["flagged"]:
            conf = min(1.0, 0.5 + a["margin"]) * h.score
            confs.append(conf)
            findings.append(Finding("fingers", f"{side}_hand", "normal", False, round(conf, 3), source,
                                    {f: a["fingers"][f]["ratio"] for f in FINGERS}))
    if parts:
        return CheckResult("fingers", source, True, max(f.confidence for f in findings if f.positive), sorted(parts),
                           True, findings)
    if len(hands) < expected:
        return CheckResult("fingers", source, None, 0.0, [], False, findings, f"只检测到 {len(hands)} 只手（应为 {expected}）")
    return CheckResult("fingers", source, False, round(min(confs), 3) if confs else 0.0, [], True, findings)


# ---------------- 颜色 ----------------

def rgb_to_lab(rgb) -> np.ndarray:
    """sRGB(0–255) → CIE Lab（D65），与 app.js 的 toLab 相同公式。支持 (...,3) 数组。"""
    c = np.asarray(rgb, dtype=float) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883
    t = lambda v: np.where(v > 0.008856, np.cbrt(v), 7.787 * v + 16 / 116)
    fx, fy, fz = t(x), t(y), t(z)
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1)


def mean_color(img, cx: float, cy: float, r: float) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0 = max(0, int(round(cx - r))), max(0, int(round(cy - r)))
    x1, y1 = min(w, max(x0 + 1, int(round(cx + r)))), min(h, max(y0 + 1, int(round(cy + r))))
    return img[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)


def nails_delta_e(img, pts) -> dict[str, float]:
    """手背中心取肤色，指尖向 DIP 方向 30% 处取指甲颜色，返回每个手指的 Lab 色差 ΔE。"""
    palm = palm_len(pts)
    skin_c = (pts[0] + pts[5] + pts[17]) / 3
    skin = rgb_to_lab(mean_color(img, skin_c[0], skin_c[1], max(4, 0.08 * palm)))
    out = {}
    for f, ids in HAND_FINGER_IDS.items():
        tip, dip = pts[ids[3]], pts[ids[2]]
        seg = dist(tip, dip)
        c = tip + 0.3 * (dip - tip)
        lab = rgb_to_lab(mean_color(img, c[0], c[1], max(3, 0.18 * seg)))
        out[f] = round(float(np.linalg.norm(lab - skin)), 1)
    return out


def nails_result(img, hands: list, th_nails: dict, source: str, mirrored: bool = False) -> CheckResult | None:
    """只对手背朝向镜头的手做色差法美甲判断；没有手背朝前的手时返回 None。"""
    thr = th_nails["delta_e"]
    findings, parts, confs = [], [], []
    for h in hands:
        if facing(h.pts, h.side, mirrored) != "back":
            continue
        de = nails_delta_e(img, h.pts)
        worst = max(de.values())
        side = h.side or "unknown"
        if worst > thr:
            flagged = [f for f, v in de.items() if v > thr]
            parts.append(f"{side}_hand")
            conf = min(1.0, 0.55 + (worst - thr) / thr)
            findings.append(Finding("nails", f"{side}_hand", "polish", True, round(conf, 3), source, de,
                                    "疑似美甲：" + "、".join(FINGER_CN[f] for f in flagged)))
        else:
            conf = min(1.0, 0.55 + (thr - worst) / thr)
            findings.append(Finding("nails", f"{side}_hand", "none", False, round(conf, 3), source, de))
        confs.append(conf)
    if not findings:
        return None
    if parts:
        return CheckResult("nails", source, True, max(f.confidence for f in findings if f.positive), parts, True, findings)
    return CheckResult("nails", source, False, round(min(confs), 3), [], True, findings)


# ---------------- 身体 ----------------

def torso_len(b: Body) -> float:
    j = b.joints
    return (dist(j["left_shoulder"], j["left_hip"]) + dist(j["right_shoulder"], j["right_hip"])) / 2


SEGMENTS = {
    "upper_arm": (("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow")),
    "forearm": (("left_elbow", "left_wrist"), ("right_elbow", "right_wrist")),
    "thigh": (("left_hip", "left_knee"), ("right_hip", "right_knee")),
    "shank": (("left_knee", "left_ankle"), ("right_knee", "right_ankle")),
}
SEGMENT_CN = {"upper_arm": "上臂", "forearm": "前臂", "thigh": "大腿", "shank": "小腿"}


def limbs_result(body: Body | None, th_pose: dict, source: str, upper: bool = True, lower: bool = True,
                 image_size: tuple[int, int] | None = None) -> CheckResult:
    """四肢：关节可见度 + 左右肢段长度比。

    肢体缺失时关键点模型常在画面内给出低可见度的"脑补"点 → 记疑似；
    关节落在画面外 → 视为没拍全（画质 / 姿势问题，应重拍），不算疑似。
    """
    if body is None:
        return CheckResult("limbs", source, None, 0.0, [], False, [], "未检测到人体")
    min_vis = th_pose["min_visibility"]
    lo, hi = th_pose["limb_symmetry"]
    core = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if not all(body.has(k, min_vis) for k in core):
        return CheckResult("limbs", source, None, 0.0, [], False, [], "躯干关键点不完整（未完整入框或背对镜头）")

    def inside(j: str) -> bool:
        if image_size is None or j not in body.joints:
            return True
        w, h = image_size
        x, y = body.joints[j]
        return 0.01 * w <= x <= 0.99 * w and 0.01 * h <= y <= 0.99 * h

    limbs = [k for k in LIMBS if (upper and k.endswith("arm")) or (lower and k.endswith("leg"))]
    findings, parts, vis_all, out_of_frame = [], [], [], []
    for k in limbs:
        if not all(inside(j) for j in LIMBS[k]):
            out_of_frame.append(k)
            continue
        v = min(body.vis.get(j, 0.0) for j in LIMBS[k])
        vis_all.append(v)
        if v < min_vis:
            parts.append(k)
            findings.append(Finding("limbs", k, "low_visibility", True, round(0.6 * (1 - v), 3), source, round(v, 3),
                                    f"{LIMB_CN[k]}关键点可见度低（{v:.2f}），疑似缺失或被遮挡"))
    for seg, (l, r) in SEGMENTS.items():
        if (seg in ("upper_arm", "forearm") and not upper) or (seg in ("thigh", "shank") and not lower):
            continue
        if not all(body.has(j, min_vis) and inside(j) for j in (*l, *r)):
            continue
        ratio = dist(body.joints[l[0]], body.joints[l[1]]) / (dist(body.joints[r[0]], body.joints[r[1]]) + 1e-6)
        if not lo <= ratio <= hi:
            side = "left" if ratio < 1 else "right"
            parts.append(f"{side}_{seg}")
            findings.append(Finding("limbs", f"{side}_{seg}", "asymmetric", True, 0.6, source, round(ratio, 3),
                                    f"{SEGMENT_CN[seg]}左右长度比 {ratio:.2f} 超出 [{lo}, {hi}]"))
        else:
            findings.append(Finding("limbs", seg, "normal", False, 1.0, source, round(ratio, 3)))
    if parts:
        return CheckResult("limbs", source, True, max(f.confidence for f in findings if f.positive), sorted(set(parts)), True, findings)
    if out_of_frame:
        return CheckResult("limbs", source, None, 0.0, [], False, findings,
                           "、".join(LIMB_CN[k] for k in out_of_frame) + "未完整入框")
    conf = min(1.0, min(vis_all) / max(min_vis, 1e-6) * 0.8) if vis_all else 0.0
    return CheckResult("limbs", source, False, round(conf, 3), [], True, findings)


# ---------------- 画质 ----------------

def brightness(img) -> float:
    g = img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114
    return float(g.mean())


def blur_score(img) -> float:
    import cv2

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def quality(img, th_quality: dict, subject=None) -> dict:
    """subject：受检部位的像素（形状 (1, N, 3)）；给出时亮度按受检部位算，模糊度仍按整图算。"""
    b, s = brightness(img if subject is None else subject), blur_score(img)
    lo, hi = th_quality["brightness"]
    issues = []
    if b < lo:
        issues.append("过暗")
    if b > hi:
        issues.append("过亮")
    if s < th_quality["blur_laplacian_min"]:
        issues.append("模糊")
    return {"brightness": round(b, 1), "blur": round(s, 1), "ok": not issues, "issues": issues}
