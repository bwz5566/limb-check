"""肢体协调性：用合成的关键点序列验证每个动作的指标和结论。"""
import math

from limbcheck.motion import analyze, motion_result

SH, HIP = 0.06, 0.035


def pose(abd=(0, 0), lift=(0, 0), touch=None, dx=0.0, drop=0.0):
    """abd：左右臂外展角（度，0 = 自然下垂，90 = 侧平举，180 = 上举）；lift：左右抬腿比例；touch：哪只手摸对侧肩；
    dx：身体左右平移；drop：髋部下移（下蹲）。受检者左侧在画面右侧。"""
    k = {}
    for i, s in enumerate(("left", "right")):
        sign = 1 if s == "left" else -1
        sh = (0.5 + sign * SH + dx, 0.3 + drop)
        a = math.radians(abd[i])
        d = (sign * math.sin(a), math.cos(a))
        el = (sh[0] + 0.1 * d[0], sh[1] + 0.1 * d[1])
        wr = (el[0] + 0.09 * d[0], el[1] + 0.09 * d[1])
        if touch == s:
            wr = (0.5 - sign * SH + dx, 0.3 + drop + 0.01)
        hip = (0.5 + sign * HIP + dx, 0.55 + drop)
        knee = (hip[0] + sign * 0.01, hip[1] + 0.12 * (1 - lift[i]) - drop * 0.6)
        ankle = (knee[0], knee[1] + 0.11)
        for n, p in (("shoulder", sh), ("elbow", el), ("wrist", wr), ("hip", hip), ("knee", knee), ("ankle", ankle)):
            k[f"{s}_{n}"] = [round(p[0], 4), round(p[1], 4), 0.95]
    return k


def seq(frames):
    return [{"t": i * 66, "ph": ph, "k": k} for i, (ph, k) in enumerate(frames)]


def run(*steps):
    a = analyze({"aspect": 1.0, "steps": list(steps)})
    return a, motion_result(a)


def arms_step(left_up=170, right_up=170, left_t=90):
    fr = [(0, pose((left_t * min(1, i / 10), 90 * min(1, i / 10)))) for i in range(25)]
    fr += [(1, pose((90 + (left_up - 90) * min(1, i / 10), 90 + (right_up - 90) * min(1, i / 10)))) for i in range(25)]
    return {"id": "arms", "status": "pass", "seq": seq(fr)}


def march_step(n=8, period=10, gaps=None, only=None):
    fr = []
    for c in range(n):
        side = only or ("left" if c % 2 == 0 else "right")
        fr += [(0, pose())] * (gaps[c] if gaps else 0)
        p = period
        for j in range(p):
            l = math.sin(math.pi * j / p) * 0.8
            fr.append((0, pose(lift=(l, 0) if side == "left" else (0, l))))
    return {"id": "march", "status": "pass", "seq": seq(fr)}


def test_normal_arms_pass():
    a, r = run(arms_step())
    m = a[0]["metrics"]
    assert a[0]["verdict"] == "normal" and m["T"]["left"]["abduction"] >= 85 and m["up"]["right"]["abduction"] >= 165
    assert r.positive is False


def test_limited_and_asymmetric_arm():
    a, r = run(arms_step(right_up=120))
    labels = {(f["label"], f["part"]) for f in a[0]["findings"]}
    assert ("limited_rom", "right_arm") in labels and ("asymmetric", "arms") in labels
    assert r.positive is True and r.confidence < 0.85          # 只有关键点一路 → 建议人工确认


def test_march_counts_and_rhythm():
    a, r = run(march_step(), arms_step())
    m = a[0]["metrics"]
    assert m["count"] == 8 and m["alternation"] == 1.0 and m["interval_cv"] < 0.1
    assert r.positive is False and r.confidence >= 0.85
    a, _ = run(march_step(n=4))
    assert a[0]["findings"][0]["label"] == "incomplete"
    a, _ = run(march_step(n=8, only="left"))
    assert {f["label"] for f in a[0]["findings"]} >= {"irregular", "asymmetric"}
    a, _ = run(march_step(n=8, gaps=[0, 20, 0, 25, 0, 2, 30, 0]))
    assert any("节律不齐" in f["note"] for f in a[0]["findings"])


def test_legs_lift_sway_and_asymmetry():
    fr = [(0, pose(lift=(0.9 * min(1, i / 8), 0))) for i in range(20)] + [(1, pose(lift=(0, 0.3 * min(1, i / 8)))) for i in range(20)]
    a, _ = run({"id": "legs", "status": "pass", "seq": seq(fr)})
    labels = {(f["label"], f["part"]) for f in a[0]["findings"]}
    assert ("limited_rom", "right_leg") in labels and ("asymmetric", "legs") in labels
    fr = [(0, pose(lift=(0.9, 0), dx=0.05 * (-1) ** i)) for i in range(20)] + [(1, pose(lift=(0, 0.9))) for i in range(20)]
    a, _ = run({"id": "legs", "status": "pass", "seq": seq(fr)})
    assert [(f["label"], f["part"]) for f in a[0]["findings"]] == [("unstable", "left_leg")]


def test_squat_depth_and_shift():
    fr = [(0, pose(drop=0.12 * math.sin(math.pi * i / 30))) for i in range(31)]
    a, _ = run({"id": "squat", "status": "pass", "seq": seq(fr)})
    assert a[0]["verdict"] == "normal" and a[0]["metrics"]["depth"] >= 0.25
    fr = [(0, pose(drop=0.04 * math.sin(math.pi * i / 30), dx=0.06 * math.sin(math.pi * i / 30))) for i in range(31)]
    a, _ = run({"id": "squat", "status": "pass", "seq": seq(fr)})
    assert {f["label"] for f in a[0]["findings"]} == {"limited_rom", "unstable"}


def test_cross_touch_alternation():
    fr = []
    for c in range(6):
        fr += [(0, pose())] * 5 + [(0, pose(touch="right" if c % 2 == 0 else "left"))] * 5
    a, r = run({"id": "cross_touch", "status": "pass", "seq": seq(fr)})
    assert a[0]["metrics"]["count"] == 6 and a[0]["metrics"]["alternation"] == 1.0 and a[0]["verdict"] == "normal"


def test_cross_touch_counts_when_lowered_wrist_leaves_frame():
    """实测：手放下时手腕在画面外、可见度很低，两次摸肩之间看不到"离开"，曾被合并成一次（浏览器 6 次、服务端只数到 2 次）。"""
    def hidden(k):
        k = {n: list(v) for n, v in k.items()}
        for s in ("left", "right"):
            k[f"{s}_wrist"][2] = 0.05
        return k
    fr = []
    for c in range(6):
        fr += [(0, hidden(pose()))] * 5 + [(0, pose(touch="right" if c % 2 == 0 else "left"))] * 5
    a, _ = run({"id": "cross_touch", "status": "pass", "seq": seq(fr)})
    assert a[0]["metrics"]["count"] == 6 and a[0]["verdict"] == "normal"


def test_timeout_and_insufficient():
    a, r = run({"id": "pinch", "status": "timeout", "note": "超时"}, {"id": "squat", "status": "pass", "seq": []})
    assert a[0]["findings"][0]["label"] == "incomplete" and a[1]["verdict"] == "insufficient"
    assert r.positive is True and r.parts == ["hands"]
    _, r = run({"id": "squat", "status": "pass", "seq": []})
    assert r.positive is None and not r.quality_ok
    assert analyze({"steps": [{"id": "palms", "status": "pass"}]}) == []
