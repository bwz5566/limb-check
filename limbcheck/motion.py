"""肢体协调性：分析动作步骤中记录的关键点序列（只依赖标准库，网页 demo 服务端和离线命令共用）。

网页 demo 在每个身体动作步骤里按约 15 帧 / 秒记录人体关键点，结束后随仲裁请求一起发给服务端：

    {"aspect": 画面宽 / 高, "steps": [{"id": "arms", "status": "pass", "ms": 8200,
      "seq": [{"t": 毫秒, "ph": 阶段, "k": {"left_shoulder": [x, y, 可见度], ...}}, ...]}, ...]}

x、y 为 0–1 的画面相对坐标（未镜像，受检者的左侧在画面右侧）；横坐标乘 aspect 后与纵坐标同一单位。

每个动作给出指标和结论（正常 / 活动范围受限 / 左右不对称 / 不稳 / 节律不齐 / 未完成），
阈值在 configs/thresholds.yaml 的 motion 段。只有关键点一路证据，所以发现问题时只建议人工确认，不自动判不符合。
"""
from __future__ import annotations

import math
import statistics as st

from .core import CheckResult, Finding

SOURCE = "L1:motion"
LABEL_CN = {"normal": "正常", "limited_rom": "活动范围受限", "asymmetric": "左右不对称", "unstable": "平衡不稳",
            "irregular": "节律不齐", "incomplete": "未完成"}
PART_CN = {"left_arm": "左臂", "right_arm": "右臂", "arms": "双臂", "left_leg": "左腿", "right_leg": "右腿", "legs": "双腿",
           "whole_body": "全身", "hands": "双手"}
STEP_CN = {"arms": "双臂侧平举 → 上举", "legs": "单腿抬起", "squat": "深蹲后起立", "march": "原地高抬腿",
           "cross_touch": "交叉摸肩", "flex": "握拳 → 张开", "pinch": "拇指依次对捏"}
SIDES = ("left", "right")

DEFAULT_TH = {
    "min_frames": 8,
    "min_vis": 0.5,
    "arms": {"t_min_abduction": 70, "up_min_abduction": 140, "max_asym_deg": 25, "min_elbow": 140},
    "legs": {"min_lift": 0.5, "max_asym": 0.35, "max_sway": 0.15},
    "squat": {"min_depth": 0.25, "max_knee_asym_deg": 25, "max_shift": 0.15},
    "march": {"lift": 0.35, "min_count": 6, "min_alternation": 0.7, "max_cv": 0.45, "max_count_diff": 2},
    "cross_touch": {"touch_ratio": 0.45, "min_count": 4, "min_alternation": 0.7, "max_cv": 0.8},
    "confidence": {"normal": 0.85, "normal_single": 0.7, "abnormal": 0.6},
}


def thresholds(th: dict | None) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_TH.items()}
    for k, v in (th or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


# ---------------- 几何 ----------------

class Frame:
    """一帧：关节 → (x, y)（横坐标已乘画面宽高比），可见度不足的关节视为缺失。"""

    def __init__(self, raw: dict, aspect: float, min_vis: float):
        self.t = float(raw.get("t", 0))
        self.ph = int(raw.get("ph", 0))
        self.p = {k: (v[0] * aspect, v[1]) for k, v in (raw.get("k") or {}).items()
                  if isinstance(v, (list, tuple)) and len(v) >= 2 and (len(v) < 3 or v[2] is None or v[2] >= min_vis)}

    def has(self, *names: str) -> bool:
        return all(n in self.p for n in names)

    def mid(self, a: str, b: str) -> tuple[float, float]:
        return ((self.p[a][0] + self.p[b][0]) / 2, (self.p[a][1] + self.p[b][1]) / 2)


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle(a, b, c) -> float:
    v1, v2 = (a[0] - b[0], a[1] - b[1]), (c[0] - b[0], c[1] - b[1])
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (math.hypot(*v1) * math.hypot(*v2) + 1e-9)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _pct(vals: list[float], p: float) -> float | None:
    a = sorted(v for v in vals if v is not None)
    if not a:
        return None
    i = (len(a) - 1) * p / 100
    lo, hi = math.floor(i), math.ceil(i)
    return a[lo] + (a[hi] - a[lo]) * (i - lo)


def _torso(frames: list[Frame]) -> float | None:
    v = [_dist(f.mid("left_shoulder", "right_shoulder"), f.mid("left_hip", "right_hip"))
         for f in frames if f.has("left_shoulder", "right_shoulder", "left_hip", "right_hip")]
    return st.median(v) if v else None


def _r(x, n=2):
    return None if x is None else round(x, n)


def _events(signal: list[tuple[float, float, str]], enter: float, leave: float, above: bool) -> list[tuple[str, float]]:
    """按迟滞阈值把每侧的连续信号切成"动作事件"，返回 [(侧, 峰值时刻)]，按时间排序。
    signal: [(t, 值, 侧)]；above=True 表示值超过 enter 进入事件（抬腿），False 表示低于 enter 进入（摸肩距离）。"""
    out = []
    for side in SIDES:
        on, best = False, None
        for t, v, s in signal:
            if s != side:
                continue
            hit = v > enter if above else v < enter
            gone = v < leave if above else v > leave
            if not on and hit:
                on, best = True, (t, v)
            elif on:
                if (v > best[1]) if above else (v < best[1]):
                    best = (t, v)
                if gone:
                    out.append((side, best[0]))
                    on = False
        if on:
            out.append((side, best[0]))
    return sorted(out, key=lambda e: e[1])


def _rhythm(events: list[tuple[str, float]]) -> dict:
    n = len(events)
    alt = sum(events[i][0] != events[i - 1][0] for i in range(1, n)) / (n - 1) if n > 1 else None
    iv = [events[i][1] - events[i - 1][1] for i in range(1, n)]
    cv = st.pstdev(iv) / st.mean(iv) if len(iv) >= 2 and st.mean(iv) > 0 else None
    cnt = {s: sum(e[0] == s for e in events) for s in SIDES}
    return {"count": n, "left": cnt["left"], "right": cnt["right"], "alternation": _r(alt), "interval_cv": _r(cv),
            "interval_ms": round(st.mean(iv)) if iv else None}


# ---------------- 各动作 ----------------

def _arms(frames, th, out):
    t = th["arms"]
    res = {}
    for ph, name, need in ((0, "T", t["t_min_abduction"]), (1, "up", t["up_min_abduction"])):
        fs = [f for f in frames if f.ph == ph]
        if len(fs) < 3:
            continue
        abd, elb, asym = {s: [] for s in SIDES}, {s: [] for s in SIDES}, []
        for f in fs:
            cur = {}
            for s in SIDES:
                sh, el, wr, hip = (f"{s}_shoulder", f"{s}_elbow", f"{s}_wrist", f"{s}_hip")
                if f.has(sh, el, wr, hip):
                    cur[s] = _angle(f.p[hip], f.p[sh], f.p[wr])
                    abd[s].append(cur[s])
                    elb[s].append(_angle(f.p[sh], f.p[el], f.p[wr]))
            if len(cur) == 2:
                asym.append(abs(cur["left"] - cur["right"]))
        hold = asym[-max(3, len(asym) // 3):] if asym else []
        r = {s: {"abduction": _r(_pct(abd[s], 90), 0), "elbow": _r(_pct(elb[s], 90), 0)} for s in SIDES}
        r["asym_deg"] = _r(st.median(hold), 0) if hold else None
        res[name] = r
        label = "侧平举" if name == "T" else "上举"
        for s in SIDES:
            a, e = r[s]["abduction"], r[s]["elbow"]
            if a is not None and a < need:
                out.append(("limited_rom", f"{s}_arm", f"{label}时{PART_CN[s + '_arm']}只抬到 {a:.0f}°（要求 ≥ {need}°）"))
            if e is not None and e < t["min_elbow"]:
                out.append(("limited_rom", f"{s}_arm", f"{label}时{PART_CN[s + '_arm']}肘部伸不直（{e:.0f}°）"))
        if r["asym_deg"] is not None and r["asym_deg"] > t["max_asym_deg"]:
            out.append(("asymmetric", "arms", f"{label}时左右臂高度相差 {r['asym_deg']:.0f}°（上限 {t['max_asym_deg']}°）"))
    return res


def _lift_series(frames):
    """每帧每侧的抬腿比例：1 − (膝 y − 髋 y) / 大腿长；站直 ≈ 0，大腿抬平 ≈ 1。
    大腿长取本步骤里大腿、小腿投影长度 95 分位的较大者（抬腿时大腿投影变短，小腿基本不变）。"""
    def p95(a, b, s):
        return _pct([_dist(f.p[f"{s}_{a}"], f.p[f"{s}_{b}"]) for f in frames if f.has(f"{s}_{a}", f"{s}_{b}")], 95) or 0.0
    thigh = {s: max(p95("hip", "knee", s), p95("knee", "ankle", s)) for s in SIDES}
    sig = []
    for f in frames:
        for s in SIDES:
            if thigh[s] and f.has(f"{s}_hip", f"{s}_knee"):
                sig.append((f.t, 1 - (f.p[f"{s}_knee"][1] - f.p[f"{s}_hip"][1]) / thigh[s], s, f))
    return sig


def _legs(frames, th, out):
    t, torso = th["legs"], _torso(frames)
    sig = _lift_series(frames)
    res = {}
    for ph, s in ((0, "left"), (1, "right")):
        vals = [(v, f) for _, v, side, f in sig if side == s and f.ph == ph]
        if len(vals) < 3:
            continue
        lift = _pct([v for v, _ in vals], 90)
        up = [f for v, f in vals if v > 0.3 and f.has("left_hip", "right_hip")]
        sway = st.pstdev([f.mid("left_hip", "right_hip")[0] for f in up]) / torso if torso and len(up) >= 3 else None
        res[s] = {"lift": _r(lift), "sway": _r(sway)}
        if lift < t["min_lift"]:
            out.append(("limited_rom", f"{s}_leg", f"{PART_CN[s + '_leg']}抬腿高度 {lift:.2f}（要求 ≥ {t['min_lift']}）"))
        if sway is not None and sway > t["max_sway"]:
            out.append(("unstable", f"{s}_leg", f"抬{PART_CN[s + '_leg']}时身体左右晃动 {sway:.2f} 躯干长（上限 {t['max_sway']}）"))
    if "left" in res and "right" in res and abs(res["left"]["lift"] - res["right"]["lift"]) > t["max_asym"]:
        out.append(("asymmetric", "legs", f"左右腿抬起高度相差 {abs(res['left']['lift'] - res['right']['lift']):.2f}（上限 {t['max_asym']}）"))
    return res


def _squat(frames, th, out):
    t, torso = th["squat"], _torso(frames)
    fs = [f for f in frames if f.has("left_hip", "right_hip", "left_ankle", "right_ankle")]
    if len(fs) < 3 or not torso:
        return {}
    hip_y = [f.mid("left_hip", "right_hip")[1] for f in fs]
    leg = _pct([(_dist(f.p["left_hip"], f.p["left_ankle"]) + _dist(f.p["right_hip"], f.p["right_ankle"])) / 2 for f in fs], 90)
    top = _pct(hip_y, 10)
    deep_i = max(range(len(fs)), key=lambda i: hip_y[i])
    deep = fs[deep_i]
    depth = (hip_y[deep_i] - top) / leg
    base_x = st.median([f.mid("left_hip", "right_hip")[0] for f, y in zip(fs, hip_y) if y - top < 0.05 * leg])
    shift = abs(deep.mid("left_hip", "right_hip")[0] - base_x) / torso
    knee = {s: _angle(deep.p[f"{s}_hip"], deep.p[f"{s}_knee"], deep.p[f"{s}_ankle"]) for s in SIDES
            if deep.has(f"{s}_hip", f"{s}_knee", f"{s}_ankle")}
    asym = abs(knee["left"] - knee["right"]) if len(knee) == 2 else None
    res = {"depth": _r(depth), "shift": _r(shift), "knee_asym_deg": _r(asym, 0)}
    if depth < t["min_depth"]:
        out.append(("limited_rom", "legs", f"下蹲深度 {depth:.2f} 腿长（要求 ≥ {t['min_depth']}）"))
    if asym is not None and asym > t["max_knee_asym_deg"]:
        out.append(("asymmetric", "legs", f"蹲到最低时左右膝角度相差 {asym:.0f}°（上限 {t['max_knee_asym_deg']}°）"))
    if shift > t["max_shift"]:
        out.append(("unstable", "whole_body", f"下蹲时身体向一侧偏移 {shift:.2f} 躯干长（上限 {t['max_shift']}）"))
    return res


def _alternating(events, t, out, part, what):
    r = _rhythm(events)
    if r["count"] < t["min_count"]:
        out.append(("incomplete", part, f"{what}只完成 {r['count']} 次（要求 ≥ {t['min_count']}）"))
        return r
    if r["alternation"] is not None and r["alternation"] < t["min_alternation"]:
        out.append(("irregular", part, f"{what}左右交替比例 {r['alternation']:.0%}（要求 ≥ {t['min_alternation']:.0%}）"))
    if r["interval_cv"] is not None and r["interval_cv"] > t["max_cv"]:
        out.append(("irregular", part, f"{what}节律不齐（间隔变异系数 {r['interval_cv']:.2f}，上限 {t['max_cv']}）"))
    if "max_count_diff" in t and abs(r["left"] - r["right"]) > t["max_count_diff"]:
        out.append(("asymmetric", part, f"{what}左 {r['left']} 次、右 {r['right']} 次，相差过大"))
    return r


def _march(frames, th, out):
    t = th["march"]
    sig = [(tt, v, s) for tt, v, s, _ in _lift_series(frames)]
    return _alternating(_events(sig, t["lift"], t["lift"] * 0.5, True), t, out, "legs", "高抬腿")


def _cross_touch(frames, th, out):
    t = th["cross_touch"]
    sig = []
    for f in frames:
        if not f.has("left_shoulder", "right_shoulder"):
            continue
        sw = _dist(f.p["left_shoulder"], f.p["right_shoulder"]) + 1e-9
        for s, o in (("left", "right"), ("right", "left")):
            # 手放下时手腕常在画面外、可见度低：算"离开肩部"，否则两次摸肩之间没有离开，会被合并成一次（浏览器端照常计数）
            d = _dist(f.p[f"{s}_wrist"], f.p[f"{o}_shoulder"]) / sw if f"{s}_wrist" in f.p else math.inf
            sig.append((f.t, d, s))
    return _alternating(_events(sig, t["touch_ratio"], t["touch_ratio"] * 1.5, False), t, out, "arms", "交叉摸肩")


ANALYZERS = {"arms": _arms, "legs": _legs, "squat": _squat, "march": _march, "cross_touch": _cross_touch}
STATUS_ONLY = {"flex": "hands", "pinch": "hands"}       # 手部精细动作：看步骤是否按时完成（指标由网页端记录）


def analyze(motion: dict | None, th: dict | None = None) -> list[dict]:
    """逐个动作分析，返回 [{step, title, status, frames, metrics, findings: [{label, part, note}], verdict}]。"""
    th = thresholds(th)
    aspect = float((motion or {}).get("aspect") or 1.0)
    out = []
    for s in (motion or {}).get("steps", []):
        sid = s.get("id")
        if sid not in ANALYZERS and sid not in STATUS_ONLY:
            continue
        found: list[tuple[str, str, str]] = []
        frames = [Frame(r, aspect, th["min_vis"]) for r in s.get("seq") or []]
        metrics, analyzed = {}, False
        if sid in ANALYZERS and len(frames) >= th["min_frames"]:
            metrics = ANALYZERS[sid](frames, th, found)
            analyzed = bool(metrics)
        if s.get("status") == "timeout":
            found.append(("incomplete", STATUS_ONLY.get(sid, "whole_body"), f"{STEP_CN.get(sid, sid)}未在时限内完成"
                          + (f"（{s['note']}）" if s.get("note") else "")))
            analyzed = True
        elif sid in STATUS_ONLY:
            analyzed = True
        verdict = "abnormal" if found else "normal" if analyzed else "insufficient"
        out.append({"step": sid, "title": STEP_CN.get(sid, sid), "status": s.get("status"), "frames": len(frames),
                    "ms": s.get("ms"), "metrics": metrics, "verdict": verdict,
                    "findings": [{"label": l, "part": p, "note": n} for l, p, n in found]})
    return out


def motion_result(analysis: list[dict], th: dict | None = None) -> CheckResult | None:
    """把各动作的分析汇总成"肢体协调性"检测项的 L1 结果。"""
    if not analysis:
        return None
    conf = thresholds(th)["confidence"]
    findings, parts = [], []
    for a in analysis:
        for f in a["findings"]:
            parts.append(f["part"])
            findings.append(Finding("coordination", f["part"], f["label"], True, conf["abnormal"], SOURCE,
                                    {"step": a["step"]}, f"{a['title']}：{f['note']}"))
        if a["verdict"] == "normal":
            findings.append(Finding("coordination", a["step"], "normal", False, conf["normal"], SOURCE, a["metrics"] or None,
                                    a["title"]))
    done = [a for a in analysis if a["verdict"] != "insufficient"]
    if parts:
        return CheckResult("coordination", SOURCE, True, conf["abnormal"], sorted(set(parts)), True, findings)
    if not done:
        return CheckResult("coordination", SOURCE, None, 0.0, [], False, findings, "动作序列帧数不足，无法分析")
    body = [a for a in done if a["step"] in ANALYZERS]
    c = conf["normal"] if len(body) >= 2 else conf["normal_single"]
    return CheckResult("coordination", SOURCE, False, c, [], True, findings,
                       f"分析了 {len(done)} 个动作：" + "、".join(a["title"] for a in done))
