"""调试可视化：把关键点、手部掩码、判定结果画到图上，用于逐张排查漏检和误报。"""
from __future__ import annotations

import numpy as np

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
BODY_EDGES = [("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
              ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
              ("right_shoulder", "right_hip"), ("left_hip", "right_hip"), ("left_hip", "left_knee"),
              ("left_knee", "left_ankle"), ("right_hip", "right_knee"), ("right_knee", "right_ankle")]


def draw(img: np.ndarray, pipeline, view: str, record: dict | None = None) -> np.ndarray:
    """返回叠加了关键点、掩码轮廓和结论文字的 RGB 图。record 为 pipeline.run 的输出（可选）。"""
    import cv2

    from .core import EXPECTED_HANDS, HAND_VIEWS
    from .keypoints import assign_sides

    out = img.copy()
    scale = max(1, int(max(img.shape[:2]) / 600))
    if pipeline.kp is not None and view in HAND_VIEWS:
        hands = assign_sides(pipeline.kp.hands(img), EXPECTED_HANDS.get(view, 2), pipeline.mirrored)
        for h in hands:
            if pipeline.contour is not None:
                mask = pipeline.contour.hand_mask(img, h, pipeline.th["contour"])
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(out, cnts, -1, (255, 0, 255), scale)
            p = h.pts.astype(int)
            for a, b in HAND_EDGES:
                cv2.line(out, tuple(p[a]), tuple(p[b]), (0, 220, 255), scale)
            for q in p:
                cv2.circle(out, tuple(q), 2 * scale, (255, 140, 0), -1)
            cv2.putText(out, f"{h.side} {h.score:.2f}", tuple(p[0] + [5, 15]), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (255, 255, 0), scale)
    elif pipeline.kp is not None:
        bodies = pipeline.kp.bodies(img)
        if bodies:
            b = bodies[0]
            for a, c in BODY_EDGES:
                if a in b.joints and c in b.joints:
                    cv2.line(out, tuple(b.joints[a].astype(int)), tuple(b.joints[c].astype(int)), (0, 220, 255), 2 * scale)
    if record:
        y = 20 * scale
        for c, r in record.get("results", {}).items():
            l1, l2 = r.get("l1") or {}, r.get("l2") or {}
            txt = f"{c}: {r['shadow']}  L1={l1.get('positive')} {l1.get('parts', '')}  L2={l2.get('positive')} {l2.get('parts', '')}"
            cv2.putText(out, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, (0, 0, 0), 3 * scale)
            cv2.putText(out, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, (255, 255, 255), scale)
            y += 18 * scale
    return out
