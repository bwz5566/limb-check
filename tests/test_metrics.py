from limbcheck.dataset import Sample
from limbcheck.metrics import compute, evaluate_acceptance, percentile, wilson


def test_wilson():
    lo, hi = wilson(0, 30)
    assert lo == 0 and 0.10 < hi < 0.12          # 0/30 的上限约 11.4%
    lo, hi = wilson(45, 50)
    assert 0.78 < lo < 0.80 and 0.95 < hi < 0.97
    assert wilson(0, 0) == (None, None)


def test_percentile():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([], 95) is None


def _res(l1, l2, shadow, first=None):
    mk = lambda p: None if p == "-" else {"positive": p, "confidence": 0.9, "parts": []}
    return {"l1": mk(l1), "l2": mk(l2), "shadow": shadow, "decision": shadow, "shadow_first_attempt": first or shadow,
            "agree": None if "-" in (l1, l2) or None in (l1, l2) else l1 == l2}


def _sample(i, missing, meta=None):
    return Sample(id=f"s{i}", image="x.jpg", view="palms", session_id=f"s{i}",
                  truth={"fingers": {"right": ["index"] if missing else [], "left": []}}, meta=meta or {})


def test_compute_system_metrics():
    samples = [_sample(i, i < 4, {"lighting": "dark" if i % 2 else "bright"}) for i in range(10)]
    preds = {}
    rows = [  # (l1, l2, shadow)
        (True, True, "fail_suggested"), (True, False, "review"), (False, False, "pass"), (None, True, "fail_suggested"),
        (False, False, "pass"), (False, False, "pass"), (True, False, "review"), (False, False, "pass"),
        (False, None, "review"), (False, False, "pass"),
    ]
    for s, (a, b, d) in zip(samples, rows):
        preds[s.id] = {"id": s.id, "results": {"fingers": _res(a, b, d)}, "latency_ms": {"l2": 1000 + 100 * int(s.id[1:])}}
    m = compute(samples, preds)
    sess = m["checks"]["fingers"]["session"]
    assert sess["n_pos"] == 4 and sess["n_neg"] == 6
    assert sess["system"]["recall"]["k"] == 3 and sess["system"]["miss_rate"]["k"] == 1
    assert sess["system"]["false_alarm_rate"]["k"] == 2
    assert sess["l1"]["recall"]["k"] == 2            # 无结果的正样本按漏检计
    assert sess["l1"]["coverage"]["k"] == 9
    assert sess["agreement"]["rate"]["n"] == 8
    assert m["latency"]["l2"]["n"] == 10
    assert "lighting" in sess["fairness"]
    assert sess["misses"] == ["s2"]


def test_acceptance():
    m = {"checks": {"fingers": {"session": {"system": {"recall": {"value": 0.92, "n": 40, "ci_low": 0.8, "ci_high": 0.97}}}}},
         "sessions": {"duration_median_s": 150}, "latency": {}, "liveness": {}}
    acc = {"criteria": [
        {"id": "a", "metric": "fingers.session.system.recall", "op": ">=", "target": 0.9, "min_n": 30},
        {"id": "b", "metric": "fingers.session.system.recall", "op": ">=", "target": 0.9, "use": "ci_low"},
        {"id": "c", "metric": "fingers.session.system.recall", "op": ">=", "target": 0.9, "min_n": 50},
        {"id": "d", "metric": "sessions.duration_median_s", "op": "<=", "target": 180},
        {"id": "e", "metric": "liveness.genuine_false_reject", "op": "<=", "target": 0.05},
        {"id": "f", "metric": "fingers.session.system.recall", "op": "record"},
    ]}
    st = {r["id"]: r["status"] for r in evaluate_acceptance(m, acc)}
    assert st == {"a": "pass", "b": "fail", "c": "insufficient", "d": "pass", "e": "missing", "f": "record"}


def test_session_metrics():
    from limbcheck.metrics import session_stats

    sessions = [
        {"steps": [{"status": "pass"}], "duration_ms": 120000, "challenge": {"status": "pass"}, "truth": {"attack": "none"}},
        {"steps": [{"status": "timeout"}], "duration_ms": 200000, "challenge": {"status": "review"}, "truth": {"attack": "replay"}},
        {"steps": [{"status": "pass"}], "duration_ms": 150000, "challenge": {"status": "timeout"}, "truth": {"attack": "replay"}},
    ]
    exp, live = session_stats(sessions)
    assert exp["completion_rate"]["k"] == 2 and exp["duration_median_s"] == 150
    assert live["attack"]["replay"]["interception"]["value"] == 1.0
    assert live["genuine_false_reject"]["value"] == 0.0
