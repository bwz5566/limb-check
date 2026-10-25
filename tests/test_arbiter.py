import pytest

from limbcheck.arbiter import arbitrate, judge, worst
from limbcheck.core import FAIL_SUGGESTED, NOT_ASSESSED, PASS, REDO, REVIEW, CheckResult
from limbcheck.registry import register

CFG = {"high_conf": 0.8, "low_conf": 0.5, "max_redo": 1}


def r(pos, conf=0.9, q=True, parts=(), src="X"):
    return CheckResult("fingers", src, pos, conf, list(parts), q)


@pytest.mark.parametrize("l1,l2,attempt,expected", [
    (r(False), r(False), 1, PASS),
    (r(True, parts=["right_hand.index"]), r(True, parts=["right_hand"]), 1, FAIL_SUGGESTED),
    (r(False), r(True), 1, REVIEW),
    (r(False, 0.3), r(True, 0.9), 0, REDO),            # 不一致且一方置信度低 → 先重做
    (r(False, 0.6), r(False, 0.9), 0, REDO),           # 一致正常但置信度低 → 先重做
    (r(False, 0.6), r(False, 0.9), 1, REVIEW),         # 重做后仍低 → 人工
    (r(False, q=False), r(False), 0, REDO),            # 画质不足 → 重做
    (r(False, q=False), r(False), 1, REVIEW),          # 最后一次仍画质不足 → 不能自动通过
    (r(None), r(False), 1, PASS),                      # L1 无法判断，L2 正常且置信度高
    (r(None), r(None), 1, REVIEW),
    (None, r(False), 1, PASS),                         # 流水线没有 L1（如纹身）
    (None, r(True), 1, FAIL_SUGGESTED),
    (r(False, 0.7), None, 1, REVIEW),                  # 单路正常但置信度不足
    (None, None, 1, REVIEW),
])
def test_judge_table(l1, l2, attempt, expected):
    assert judge(l1, l2, CFG, attempt)[0] == expected


def test_never_auto_fail():
    for l1 in (r(True), r(False), r(None), None):
        for l2 in (r(True), r(False), r(None), None):
            assert judge(l1, l2, CFG, 1)[0] != "fail"


def test_record_only_keeps_shadow():
    a = arbitrate("tattoo", None, r(True), CFG, mode="record_only")
    assert a.decision == NOT_ASSESSED and a.shadow == FAIL_SUGGESTED


def test_off_mode():
    assert arbitrate("hair", r(True), r(True), CFG, mode="off").shadow == NOT_ASSESSED


def test_agreement_flag():
    assert arbitrate("fingers", r(False), r(False), CFG).agree is True
    assert arbitrate("fingers", r(False), r(True), CFG).agree is False
    assert arbitrate("fingers", r(True, parts=["left_hand.index"]), r(True, parts=["right_hand"]), CFG).agree is False


def test_per_check_override():
    cfg = {**CFG, "per_check": {"scar": {"high_conf": 0.95}}}
    assert arbitrate("scar", None, r(False, 0.9), cfg, attempt=1).shadow == REVIEW
    assert arbitrate("tattoo", None, r(False, 0.9), cfg, attempt=1).shadow == PASS


def test_custom_strategy():
    @register("arbiter", "always_review_test")
    def always(l1, l2, cfg, attempt):
        return REVIEW, "测试策略"

    a = arbitrate("fingers", r(False), r(False), {**CFG, "strategy": "always_review_test"})
    assert a.shadow == REVIEW and a.reason == "测试策略"


def test_worst():
    assert worst([PASS, REDO, PASS]) == REDO
    assert worst([PASS, FAIL_SUGGESTED, REVIEW]) == FAIL_SUGGESTED
    assert worst([]) is None
