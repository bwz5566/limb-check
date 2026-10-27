"""L1 / L2 一致性仲裁（仅标准库）。

规则：
- 能明确判断的直接判：所有可用来源都认为正常、置信度都 ≥ high_conf、画质合格 → 自动"满足"；
  认为有问题、置信度 ≥ auto_fail_conf（默认策略还要求两路一致）→ 直接"不符合"（fail）。
- 置信度不够、两路不一致 → fail_suggested / review，转人工；auto_fail_conf 设为 null 时退回"有问题一律转人工"。
- 阈值随筛选严格度（configs/strictness.yaml）变化，见 limbcheck.strictness.arbitration_cfg。
- record_only 的检测项照常计算（shadow），但对外结论为 not_assessed。
- l1 / l2 为 None 表示流水线里没有这一路（例如纹身没有 L1 模型）；
  positive 为 None 表示这一路跑了但无法判断。

可替换：thresholds.yaml 的 arbitration.strategy 指定策略名（默认 default），
自定义策略用 @register("arbiter", "名称") 注册一个 fn(l1, l2, cfg, attempt) -> (decision, reason)；
arbitration.per_check 可按检测项覆盖阈值（如疤痕要求更高置信度）。

纹身、疤痕、头发、指甲用 policy 策略：L2 结果里每处标记（头发为 L1 的发色、发长判定，指甲为每只手）已按录用标准（configs/policies/marks.yaml，limbcheck/marks.py）
判为 pass / review / block / clean，策略据此给结论：block → 疑似问题·待人工确认（附命中条款）；
clean（疑似污渍）→ 请清洁后重做；review → 人工核验；全部符合 → 按默认规则看置信度。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .core import FAIL, FAIL_SUGGESTED, NOT_ASSESSED, PASS, REDO, REVIEW, CheckResult
from .registry import get, register

DEFAULT_CFG = {"high_conf": 0.8, "low_conf": 0.5, "max_redo": 1, "auto_fail_conf": None, "auto_fail_single_source": False}
SEVERITY = {NOT_ASSESSED: 0, PASS: 1, REDO: 2, REVIEW: 3, FAIL_SUGGESTED: 4, FAIL: 5}


def _auto_fail(conf: float, cfg: dict) -> bool:
    thr = cfg.get("auto_fail_conf")
    return thr is not None and conf >= thr


@dataclass
class Arbitration:
    check: str
    decision: str          # 对外结论（考虑 mode）
    shadow: str            # 按 judge 模式算出的结论（record_only 项也有，用于评测）
    reason: str
    agree: bool | None     # 两路都有结论时是否一致；不足两路为 None
    mode: str = "judge"

    def to_dict(self) -> dict:
        return asdict(self)


def _parts_overlap(a: CheckResult, b: CheckResult) -> bool:
    if not a.parts or not b.parts:
        return True
    # 手指比较到"哪只手"即可：L1 给 right_hand.index，L2 可能只给 right_hand
    hand = lambda p: p.split(".")[0]
    return bool({hand(p) for p in a.parts} & {hand(p) for p in b.parts})


def agreement(l1: CheckResult | None, l2: CheckResult | None) -> bool | None:
    if l1 is None or l2 is None or l1.positive is None or l2.positive is None:
        return None
    if l1.positive != l2.positive:
        return False
    return _parts_overlap(l1, l2) if l1.positive else True


@register("arbiter", "default")
def judge(l1: CheckResult | None, l2: CheckResult | None, cfg: dict | None = None, attempt: int = 0) -> tuple[str, str]:
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    hi, lo, can_redo = cfg["high_conf"], cfg["low_conf"], attempt < cfg["max_redo"]
    srcs = [r for r in (l1, l2) if r is not None]
    if not srcs:
        return REVIEW, "没有任何模型结果"
    bad_quality = any(not r.quality_ok for r in srcs)
    if bad_quality and can_redo:
        return REDO, "画质或姿势不足以判断"
    valid = [r for r in srcs if r.positive is not None]
    if not valid:
        return (REDO, "模型均无法判断") if can_redo else (REVIEW, "模型均无法判断")
    if any(r.positive for r in valid):
        if len(valid) == 2 and not all(r.positive for r in valid):
            if can_redo and min(r.confidence for r in valid) < lo:
                return REDO, "两路不一致且有一方置信度低"
            return REVIEW, f"两路不一致：{' / '.join(f'{r.source}={_yn(r)}' for r in valid)}"
        who = "两路一致" if len(valid) == 2 else f"{valid[0].source}（单路）"
        conf = min(r.confidence for r in valid)
        if (len(valid) == 2 or cfg.get("auto_fail_single_source")) and not bad_quality and _auto_fail(conf, cfg):
            return FAIL, f"{who}认为有问题，置信度 {conf:.2f}，直接判定"
        return FAIL_SUGGESTED, f"{who}认为有问题，待人工确认"
    # 全部认为正常
    conf = min(r.confidence for r in valid)
    if bad_quality:
        return REVIEW, "认为正常但画质不足"
    if len(valid) < len(srcs):
        missing = [r.source for r in srcs if r.positive is None]
        if conf >= hi:
            return PASS, f"{'、'.join(missing)} 无法判断，按其余来源处理：正常"
        return REVIEW, f"{'、'.join(missing)} 无法判断，其余来源置信度不足"
    if conf >= hi:
        return PASS, "两路一致：正常" if len(valid) == 2 else f"仅 {valid[0].source}：正常且置信度高"
    if can_redo:
        return REDO, f"认为正常但置信度低（{conf:.2f}）"
    return REVIEW, f"重做后置信度仍低（{conf:.2f}）"


def _verdicts(r: CheckResult | None) -> list[tuple[str, str, float]]:
    """(verdict, 说明, 置信度)。value 有 kind 的（逐处标记、发色发长）用 note，跨处规则用命中的条款。"""
    out = []
    for f in (r.findings if r else []):
        v = f.value if isinstance(f.value, dict) else None
        if v and v.get("verdict"):
            texts = [x["text"] for x in v.get("rules") or [] if x.get("action") == v["verdict"]]
            out.append((v["verdict"], f.note if v.get("kind") else "；".join(texts), f.confidence))
    return out


def _whole_item(r: CheckResult | None) -> bool:
    """整体判定（发色发长 hair_*、指甲 nail_*），不是纹身疤痕那样的逐处标记。"""
    return any(isinstance(f.value, dict) and str(f.value.get("kind", "")).startswith(("hair_", "nail_")) for f in (r.findings if r else []))


@register("arbiter", "policy")
def policy_judge(l1: CheckResult | None, l2: CheckResult | None, cfg: dict | None = None, attempt: int = 0) -> tuple[str, str]:
    """按录用标准逐处判定后的结论（纹身、疤痕）。没有标准判定信息时退回默认策略。"""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    vs = _verdicts(l1) + _verdicts(l2)
    if not vs:
        return judge(l1, l2, cfg, attempt)
    pick = lambda a: [t for v, t, _ in vs if v == a]
    if pick("block"):
        conf = max(c for v, _, c in vs if v == "block")
        if _auto_fail(conf, cfg):
            return FAIL, f"按录用标准不符合（置信度 {conf:.2f}）：" + "；".join(pick("block"))
        return FAIL_SUGGESTED, f"按录用标准疑似不符合，置信度 {conf:.2f} 不足以直接判定，待人工确认：" + "；".join(pick("block"))
    if pick("clean"):
        msg = "；".join(pick("clean"))
        return (REDO, msg) if attempt < cfg["max_redo"] else (REVIEW, f"{msg}；已重拍仍存在时交人工核验")
    if pick("review"):
        return REVIEW, "需人工核验：" + "；".join(pick("review"))
    decision, reason = judge(l1, l2, cfg, attempt)
    passed = [t for v, t, _ in vs if v == "pass" and t]
    if _whole_item(l1) or _whole_item(l2):      # 发色发长、指甲：列出各项判定
        return decision, ("；".join(passed) + "；" if passed else "") + reason
    return decision, (f"发现 {len(passed)} 处，均符合录用标准；" if passed else "") + reason


def _yn(r: CheckResult) -> str:
    return "有问题" if r.positive else "正常"


def arbitrate(check: str, l1: CheckResult | None, l2: CheckResult | None, cfg: dict | None = None,
              mode: str = "judge", attempt: int = 0) -> Arbitration:
    if mode == "off":
        return Arbitration(check, NOT_ASSESSED, NOT_ASSESSED, "岗位未启用该项", None, mode)
    cfg = dict(cfg or {})
    cfg.update((cfg.pop("per_check", None) or {}).get(check) or {})
    strategy = get("arbiter", cfg.pop("strategy", None) or "default")
    shadow, reason = strategy(l1, l2, cfg, attempt)
    decision = NOT_ASSESSED if mode == "record_only" else shadow
    if mode == "record_only":
        reason = f"只记录（书面依据未落实）；按判定模式应为 {shadow}：{reason}"
    return Arbitration(check, decision, shadow, reason, agreement(l1, l2), mode)


def worst(decisions: list[str]) -> str | None:
    """会话级汇总：取最严重的结论（fail > fail_suggested > review > redo > pass > not_assessed）。"""
    ds = [d for d in decisions if d]
    return max(ds, key=SEVERITY.__getitem__) if ds else None
