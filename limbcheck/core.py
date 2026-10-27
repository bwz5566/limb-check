"""核心常量与数据结构（仅标准库，网页 demo 的 server.py 也会导入）。

约定：
- `positive=True` 表示"该检测项发现了问题"（缺指、有美甲、有纹身、染发、肢体异常、疑似攻击），
  `False` 表示正常，`None` 表示无法判断（没检测到手、模型调用失败、输出无法解析等）。
- 左右一律指受检者本人的左右（解剖学左右），与画面左右无关。
- 检测项与视角定义来自 configs/checks.json；新增检测项、视角只需改该文件，
  再按需注册 L1 判定（limbcheck.l1checks），见 docs/EXTENDING.md。
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


def config_dir() -> Path:
    """配置目录：环境变量 LIMBCHECK_CONFIG_DIR，否则为仓库内 configs/。"""
    return Path(os.environ.get("LIMBCHECK_CONFIG_DIR") or Path(__file__).resolve().parent.parent / "configs")


def _load_check_defs() -> dict:
    path = config_dir() / "checks.json"
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}。请在仓库内运行（pip install -e .），或设置 LIMBCHECK_CONFIG_DIR。")
    return json.loads(path.read_text(encoding="utf-8"))


CHECK_DEFS = _load_check_defs()
CHECKS: tuple[str, ...] = tuple(CHECK_DEFS["checks"])
CHECK_CN = {k: v["name"] for k, v in CHECK_DEFS["checks"].items()}
IMAGE_CHECKS = tuple(k for k, v in CHECK_DEFS["checks"].items() if v.get("views"))
APPEARANCE_CHECKS = tuple(k for k, v in CHECK_DEFS["checks"].items() if v.get("appearance"))
BASIS_REQUIRED = tuple(k for k, v in CHECK_DEFS["checks"].items() if v.get("basis_required"))

# 拍摄视角 → 该视角能评估的检测项
VIEW_CHECKS: dict[str, tuple[str, ...]] = {
    view: tuple(c for c, d in CHECK_DEFS["checks"].items() if view in d.get("views", ())) for view in CHECK_DEFS["views"]
}
VIEW_CN = {k: v["name"] for k, v in CHECK_DEFS["views"].items()}
HAND_VIEWS = tuple(k for k, v in CHECK_DEFS["views"].items() if v.get("kind") == "hand")
BODY_VIEWS = tuple(k for k, v in CHECK_DEFS["views"].items() if v.get("kind") == "body")
EXPECTED_HANDS = {k: v["expected_hands"] for k, v in CHECK_DEFS["views"].items() if "expected_hands" in v}

FINGERS = ("thumb", "index", "middle", "ring", "little")
FINGER_CN = {"thumb": "拇指", "index": "食指", "middle": "中指", "ring": "无名指", "little": "小指"}
CN_FINGER = {v: k for k, v in FINGER_CN.items()}

SIDES = ("right", "left")
SIDE_CN = {"right": "右手", "left": "左手"}

# 仲裁结论
PASS = "pass"                      # 满足（自动通过）
FAIL = "fail"                      # 明确不符合（置信度 ≥ auto_fail_conf，自动判定；保留申诉 / 线下复核渠道）
FAIL_SUGGESTED = "fail_suggested"  # 认为有问题但置信度不够 → 进入人工确认
REVIEW = "review"                  # 不一致或置信度不足 → 人工复核
REDO = "redo"                      # 画质或姿势不满足 → 重做该步骤
NOT_ASSESSED = "not_assessed"      # 未启用或 record_only 模式
DECISIONS = (PASS, FAIL, FAIL_SUGGESTED, REVIEW, REDO, NOT_ASSESSED)
DECISION_CN = {
    PASS: "满足",
    FAIL: "不符合",
    FAIL_SUGGESTED: "疑似问题·待人工确认",
    REVIEW: "需人工复核",
    REDO: "需重做",
    NOT_ASSESSED: "未评估",
}


@dataclass
class Finding:
    """一条具体判断，例如"右手食指疑似缺失或短缩"。"""

    check: str
    part: str                      # 如 right_hand.index、left_forearm、head
    label: str                     # 如 missing_or_short、normal、polish、tattoo、none、dyed
    positive: bool | None
    confidence: float = 0.0
    source: str = ""               # 如 L1:mediapipe、L1:contour-skin、L2:Qwen/Qwen3-VL-8B-Instruct
    value: Any = None              # 如长度比、ΔE、面积占比、颜色描述
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Finding":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__ if k in d})


@dataclass
class CheckResult:
    """某个来源（L1 某模型 / L2 某大模型 / 人工）对一个检测项的整体判断。"""

    check: str
    source: str
    positive: bool | None
    confidence: float = 0.0
    parts: list[str] = field(default_factory=list)   # 判为有问题的部位
    quality_ok: bool = True                          # False 表示画质/姿势不足以判断
    findings: list[Finding] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["findings"] = [f.to_dict() for f in self.findings]
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> "CheckResult | None":
        if d is None:
            return None
        kw = {k: d.get(k) for k in cls.__dataclass_fields__ if k in d}
        kw["findings"] = [Finding.from_dict(f) for f in d.get("findings") or []]
        kw["parts"] = list(d.get("parts") or [])
        return cls(**kw)


def merge_any(results: Iterable[CheckResult | None], source: str) -> CheckResult | None:
    """把同一层的多路结果按"任一路报问题即报问题"合并（如关键点长度比 + 轮廓复核）。"""
    rs = [r for r in results if r is not None]
    if not rs:
        return None
    check = rs[0].check
    pos = [r for r in rs if r.positive is True]
    neg = [r for r in rs if r.positive is False]
    if pos:
        positive, conf = True, max(r.confidence for r in pos)
    elif neg and len(neg) == len(rs):
        positive, conf = False, min(r.confidence for r in neg)
    else:
        positive, conf = None, 0.0
    parts = sorted({p for r in pos for p in r.parts})
    return CheckResult(
        check=check,
        source=source,
        positive=positive,
        confidence=round(conf, 3),
        parts=parts,
        quality_ok=all(r.quality_ok for r in rs),
        findings=[f for r in rs for f in r.findings],
        note="; ".join(r.note for r in rs if r.note),
    )
