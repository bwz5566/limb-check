"""肢体正常综合判断（limbcheck/summary.py，configs/summary.yaml）。"""
from limbcheck.summary import summarize


def e(decision, mode="judge", notes=None, reason=""):
    return {"mode": mode, "decision": decision, "shadow": decision, "notes": notes or [], "reason": reason}


OK = {"limbs": e("pass"), "fingers": e("pass"), "nails": e("pass"), "hair": e("pass"), "tattoo": e("pass"), "scar": e("pass")}
STEPS = [{"id": "arms", "title": "双臂侧平举 → 上举", "status": "pass"}, {"id": "challenge", "title": "随机手势挑战", "status": "pass"}]
MODES = {"liveness": "judge"}


def test_all_pass():
    s = summarize(OK, STEPS, modes=MODES)
    assert s["outcome"] == "pass" and s["name"] == "通过"
    assert [g["id"] for g in s["groups"]] == ["limbs", "hands", "nails", "hair", "marks", "liveness"]


def test_nails_problem_is_conditional():
    """指甲可整改：疑似美甲不淘汰，给"有条件通过"并写明整改要求。"""
    s = summarize({**OK, "nails": e("fail_suggested", notes=["右手：有美甲（甲油），入职前需卸除"])}, STEPS, modes=MODES)
    assert s["outcome"] == "conditional" and "卸除美甲" in s["conditions"][0]["text"]
    assert not s["conditions"][0]["confirmed"]
    assert next(g for g in s["groups"] if g["id"] == "nails")["decision"] == "conditional"


def test_fail_beats_everything():
    s = summarize({**OK, "fingers": e("fail", reason="右手食指疑似缺失"), "tattoo": e("review"), "nails": e("fail")}, STEPS, modes=MODES)
    assert s["outcome"] == "fail" and s["failed"][0]["group"] == "手部" and "食指" in s["failed"][0]["reason"]


def test_review_needs_human():
    s = summarize({**OK, "hair": e("fail_suggested", reason="整体染色（红色）")}, STEPS, modes=MODES)
    assert s["outcome"] == "manual" and s["manual"][0]["group"] == "头发"


def test_unfinished_step_needs_human():
    steps = STEPS + [{"id": "flex", "title": "握拳 → 张开", "status": "timeout", "note": "超时"}]
    s = summarize(OK, steps, modes=MODES)
    assert s["outcome"] == "manual" and "握拳" in s["manual"][0]["reason"]


def test_record_only_does_not_count():
    s = summarize({**OK, "nails": e("not_assessed", mode="record_only"), "hair": e("not_assessed", mode="record_only")}, STEPS,
                  modes=MODES)
    assert s["outcome"] == "pass"
    assert next(g for g in s["groups"] if g["id"] == "hair")["record_only"]


def test_scope_limits_groups_and_steps():
    steps = [{"id": "flex", "title": "握拳 → 张开", "status": "timeout"}]
    s = summarize({"hair": e("pass")}, steps, checks=["hair"], modes=MODES)
    assert s["outcome"] == "pass" and [g["id"] for g in s["groups"]] == ["hair"]


def test_challenge_failure_is_liveness_review():
    steps = [{"id": "challenge", "title": "随机手势挑战", "status": "review", "note": "1 轮未在时限内完成"}]
    s = summarize(OK, steps, modes=MODES)
    assert s["outcome"] == "manual" and s["manual"][0]["group"] == "活体 / 防作弊"
    assert summarize(OK, steps, modes={"liveness": "record_only"})["outcome"] == "pass"


def test_redo():
    assert summarize({**OK, "limbs": e("redo")}, STEPS, modes=MODES)["outcome"] == "redo"
