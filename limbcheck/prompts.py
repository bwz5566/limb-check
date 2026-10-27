"""L2 视觉大模型的提示词、输出 JSON 约束和解析（仅标准库，网页 demo 的 server.py 也直接用）。

提示词文本放在 configs/prompts/<版本>.json，代码只负责拼装和校验：
- 改提示词：复制 v1.json 为 v2.json 修改，用环境变量 LIMBCHECK_PROMPT_VERSION=v2 或
  configs/checks.json 的 default_prompt_version 切换，再跑回归评测对比（docs/VALIDATION.md 第 9 节）。
- 新增检测项：在 checks.json 和提示词文件的 checks 里各加一项即可，见 docs/EXTENDING.md。
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache

from .core import CHECK_CN, CHECK_DEFS, EXPECTED_HANDS, FINGERS, VIEW_CHECKS, CheckResult, Finding, config_dir

UNKNOWN_LABELS = {"not_visible", "uncertain", "unknown"}


@lru_cache(maxsize=8)
def load_prompts(version: str | None = None) -> dict:
    version = version or os.environ.get("LIMBCHECK_PROMPT_VERSION") or CHECK_DEFS.get("default_prompt_version", "v1")
    path = config_dir() / "prompts" / f"{version}.json"
    if not path.exists():
        raise FileNotFoundError(f"找不到提示词文件 {path}")
    p = json.loads(path.read_text(encoding="utf-8"))
    p.setdefault("version", version)
    return p


def prompt_version() -> str:
    return load_prompts()["version"]


def l2_checks_for(view: str, enabled=None) -> list[str]:
    """该视角下交给 L2 的检测项：视角能看到 ∩ 岗位启用 ∩ 提示词文件里有规则。"""
    rules = load_prompts()["checks"]
    return [c for c in VIEW_CHECKS.get(view, ()) if c in rules and (enabled is None or c in enabled)]


def schema(checks: list[str] | None = None) -> dict:
    """输出 JSON Schema（vLLM 等支持结构化输出时用 response_format 强制）。"""
    p = load_prompts()
    return {
        "type": "object",
        "required": ["items", "image_quality"],
        "properties": {
            "items": {"type": "array", "items": {
                "type": "object",
                "required": ["check", "part", "label", "confidence"],
                "properties": {
                    "check": {"enum": list(checks or p["checks"])},
                    "part": {"type": "string"},
                    "label": {"type": "string"},
                    "fingers": {"type": "array", "items": {"enum": list(FINGERS)}},
                    "value": {"type": ["number", "string", "null"]},
                    "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                    "attrs": {"type": "object"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "uncertain": {"type": "boolean"},
                },
            }},
            "image_quality": {"enum": ["good", "fair", "poor"]},
            "reason": {"type": "string", "maxLength": 200},
        },
    }


def build_user_prompt(view: str, checks: list[str]) -> str:
    p = load_prompts()
    if view not in p["view_desc"]:
        raise ValueError(f"提示词文件没有视角 {view} 的描述")
    if not checks:
        raise ValueError("没有需要评估的项目")
    rules = "\n".join(f"{i + 1}. {p['checks'][c]['rule']}" for i, c in enumerate(checks))
    names = "、".join(f"{CHECK_CN.get(c, c)}（{c}）" for c in checks)
    example = {
        "items": [{"check": checks[0], "part": "…", "label": "…", "fingers": [], "value": None, "confidence": 0.9, "uncertain": False}],
        "image_quality": "good",
        "reason": "一句话说明依据",
    }
    return (
        f"图片：{p['view_desc'][view]}。\n"
        f"本次只评估：{names}。不要输出其他项目。\n"
        f"按以下规则输出 items，每条都要有 check、part、label、confidence（0 到 1），可选 fingers、value、bbox、attrs、uncertain：\n{rules}\n"
        f"image_quality 为 good、fair 或 poor；reason 不超过 50 字。\n"
        f"输出格式示例：{json.dumps(example, ensure_ascii=False)}"
    )


def build_messages(view: str, checks: list[str], images: list[str]) -> list[dict]:
    """images 为 data URL（data:image/jpeg;base64,...）或 http(s) 地址。"""
    content = [{"type": "text", "text": build_user_prompt(view, checks)}]
    content += [{"type": "image_url", "image_url": {"url": u}} for u in images]
    return [{"role": "system", "content": load_prompts()["system"]}, {"role": "user", "content": content}]


def _extract_json(text: str):
    text = re.sub(r"```(?:json)?", "", text or "").strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("输出中没有 JSON 对象")
    depth = 0
    in_str = esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start: i + 1])
    raise ValueError("JSON 对象不完整")


def _bbox(v) -> list[float] | None:
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        return None
    try:
        return [float(x) for x in v]
    except (TypeError, ValueError):
        return None


def parse_reply(text: str, checks: list[str]) -> tuple[dict | None, list[str]]:
    """解析并校验模型输出。返回 (清洗后的对象, 问题列表)。对象为 None 表示完全无法使用。"""
    rules = load_prompts()["checks"]
    errors: list[str] = []
    try:
        obj = _extract_json(text)
    except (ValueError, json.JSONDecodeError) as e:
        return None, [f"无法解析 JSON：{e}"]
    if not isinstance(obj, dict) or not isinstance(obj.get("items"), list):
        return None, ["缺少 items 数组"]
    items = []
    for raw in obj["items"]:
        if not isinstance(raw, dict):
            errors.append("items 中有非对象元素")
            continue
        c = raw.get("check")
        if c not in checks:
            errors.append(f"输出了未要求的项目 {c!r}，已丢弃")   # 护栏：只保留本次启用的项目
            continue
        label = str(raw.get("label", "")).strip()
        if label not in set(rules[c]["positive"]) | set(rules[c]["negative"]) | UNKNOWN_LABELS:
            errors.append(f"{c} 的 label {label!r} 不在约定范围内，按 uncertain 处理")
            label = "uncertain"
        try:
            conf = min(1.0, max(0.0, float(raw.get("confidence", 0))))
        except (TypeError, ValueError):
            conf = 0.0
            errors.append(f"{c} 的 confidence 不是数字")
        items.append({
            "check": c,
            "part": str(raw.get("part", "")),
            "label": label,
            "fingers": [f for f in raw.get("fingers") or [] if f in FINGERS],
            "value": raw.get("value"),
            "bbox": _bbox(raw.get("bbox")),
            "attrs": {k: v for k, v in raw["attrs"].items() if isinstance(k, str)} if isinstance(raw.get("attrs"), dict) else {},
            "confidence": conf,
            "uncertain": bool(raw.get("uncertain", False)),
        })
    quality = obj.get("image_quality")
    if quality not in ("good", "fair", "poor"):
        errors.append(f"image_quality {quality!r} 不在约定范围内")
        quality = "fair"
    return {"items": items, "image_quality": quality, "reason": str(obj.get("reason", ""))[:200]}, errors


def to_check_results(parsed: dict | None, checks: list[str], view: str, source: str, errors: list[str] | None = None,
                     meta: dict | None = None) -> dict[str, CheckResult]:
    """把解析后的 L2 输出转换为每个检测项的 CheckResult。

    - 任一条为"有问题"标签 → 有问题（uncertain 的有问题标签置信度减半，仍按有问题处理，宁可多送人工）；
    - 全部为"正常"标签且不 uncertain → 正常（逐手的项目要求每只手都返回了）；
    - 其余（未返回、无法确定）→ 无法判断。
    - 提示词里带 "policy" 的项目（纹身、疤痕）改由 limbcheck.marks 按录用标准判定；
      meta 为 {mm_per_px, img_w, img_h, scale_source, params}，用来把框换算成毫米、带上岗位参数。
    """
    rules = load_prompts()["checks"]
    out: dict[str, CheckResult] = {}
    if parsed is None:
        note = "; ".join(errors or ["无输出"])
        return {c: CheckResult(check=c, source=source, positive=None, note=note) for c in checks}
    quality_ok = parsed["image_quality"] != "poor"
    for c in checks:
        pos_labels, neg_labels = set(rules[c]["positive"]), set(rules[c]["negative"])
        items = [i for i in parsed["items"] if i["check"] == c]
        if rules[c].get("policy"):
            from .marks import policy_result

            others = [i for i in parsed["items"] if i["check"] != c and rules.get(i["check"], {}).get("policy")]
            out[c] = policy_result(c, items, view, source, quality_ok, meta, parsed.get("reason", ""), others)
            continue
        findings, pos_parts, pos_conf, neg_conf, unsure = [], [], [], [], False
        for i in items:
            label = i["label"]
            if label in pos_labels:
                is_pos = True
                pos_conf.append(i["confidence"] * (0.5 if i["uncertain"] else 1.0))
                pos_parts += [f"{i['part']}.{f}" for f in i["fingers"]] if i["fingers"] else [i["part"]]
            elif label in neg_labels and not i["uncertain"]:
                is_pos = False
                neg_conf.append(i["confidence"])
            else:
                is_pos = None
                unsure = True
            findings.append(Finding(check=c, part=i["part"], label=label, positive=is_pos, confidence=i["confidence"],
                                    source=source, value=i["fingers"] or i["value"]))
        note = ""
        if pos_conf:
            positive, conf = True, max(pos_conf)
        elif not items:
            positive, conf, note = None, 0.0, "模型未返回该项"
        elif unsure:
            positive, conf, note = None, 0.0, "模型表示无法确定"
        else:
            positive, conf = False, min(neg_conf)
            need = EXPECTED_HANDS.get(view) if rules[c].get("per_hand") else None
            got = len({i["part"] for i in items})
            if need and got < need:
                positive, conf, note = None, 0.0, f"只返回了 {got} 只手"
        out[c] = CheckResult(check=c, source=source, positive=positive, confidence=round(conf, 3),
                             parts=sorted(set(pos_parts)), quality_ok=quality_ok, findings=findings,
                             note=note or parsed.get("reason", ""))
    return out
