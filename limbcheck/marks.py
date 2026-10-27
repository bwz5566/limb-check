"""纹身 / 疤痕 / 污渍按录用标准判定（仅标准库，网页 demo 的 server.py 也直接用）。

输入是"皮肤标记"（Mark）：大模型（或将来的分割模型）给出的每一处纹身、疤痕、污渍，带部位、框、内容类别、
颜色深浅、与参照物的大小比较等属性；本模块把它们按 configs/policies/marks.yaml 的录用标准逐条判定，
给出 pass / review / block / clean 和命中的条款，供仲裁（arbiter 的 policy 策略）和报告使用。

尺寸：有框（bbox）且知道图像比例（meta.mm_per_px，由手掌、前臂等关键点估算，见 geometry.mm_per_px_*）时按
框换算成毫米；否则用大模型估计的厘米数；再否则用大模型给出的"比参照物小 / 相近 / 大"。与参照面积
（身份证 46.2 cm²、1 分硬币 2.54 cm²）相差不足 borderline 的算临界，交人工核验。

跨视角规则（手指纹身 ≥ 2 处、同一只手手背多处、疤痕与纹身同部位）在 session_review 里按整场检测合并计算。
系统只给建议：block 在仲裁里映射为"疑似问题·待人工确认"，从不直接输出"不通过"。
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from .core import FINGER_CN, FINGERS, SIDE_CN, CheckResult, Finding, config_dir

PASS, REVIEW, CLEAN, BLOCK = "pass", "review", "clean", "block"
ACTIONS = (PASS, REVIEW, CLEAN, BLOCK)
_RANK = {PASS: 0, REVIEW: 1, CLEAN: 2, BLOCK: 3}
ACTION_CN = {PASS: "符合标准", REVIEW: "需人工核验", CLEAN: "疑似污渍·请清洁后重拍", BLOCK: "不符合录用标准·待人工确认"}
REGIONS = ("neck", "upper_arm", "forearm", "finger", "hand_back", "palm", "face", "other")
REGION_CN = {"neck": "脖子", "upper_arm": "大臂", "forearm": "小臂", "finger": "手指", "hand_back": "手背", "palm": "手掌",
             "face": "面部", "other": "其他部位"}
POLICY_CHECKS = ("tattoo", "scar")


# ---------------- 标准（配置） ----------------

_OVERRIDE: dict | None = None


@lru_cache(maxsize=1)
def _load_default() -> dict:
    path = config_dir() / "policies" / "marks.yaml"
    try:
        import yaml
    except ImportError:   # 网页 demo 只用标准库：读 export-web-config 生成的 JSON
        web = config_dir().parent / "apps" / "web-demo" / "config.generated.json"
        return json.loads(web.read_text(encoding="utf-8"))["policies"]["marks"]
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def policy() -> dict:
    return _OVERRIDE if _OVERRIDE is not None else _load_default()


def set_policy(p: dict | None) -> None:
    """替换当前标准（测试、或服务端从别处加载时用）；传 None 恢复为 configs/policies/marks.yaml。"""
    global _OVERRIDE
    _OVERRIDE = p


def worst_action(actions) -> str:
    acts = [a for a in actions if a]
    return max(acts, key=_RANK.__getitem__) if acts else PASS


def ref_area_mm2(ref: str, pol: dict | None = None) -> float:
    r = (pol or policy())["reference"]
    if ref == "coin":
        return math.pi * (r["coin_diameter_mm"] / 2) ** 2
    w, h = r["id_card_mm"]
    return w * h


REF_CN = {"id_card": "身份证（85.6×54 mm）", "coin": " 1 分硬币（直径 1.8 cm）"}


# ---------------- 标记 ----------------

@dataclass
class Mark:
    kind: str                       # tattoo / scar / stain
    part: str                       # 大模型给的部位名，如 right_forearm
    region: str                     # 归一化部位，见 REGIONS
    side: str | None = None
    fingers: list[str] = field(default_factory=list)
    view: str = ""
    confidence: float = 0.0
    uncertain: bool = False
    attrs: dict = field(default_factory=dict)
    bbox: list | None = None
    size: dict | None = None        # {w_mm, h_mm, area_mm2, source}

    @property
    def where(self) -> str:
        if self.region == "finger":
            return SIDE_CN.get(self.side or "", "") + ("、".join(FINGER_CN[f] for f in self.fingers) or "手指")
        return {"right": "右", "left": "左"}.get(self.side or "", "") + REGION_CN.get(self.region, self.region)


def region_of(part: str, view: str = "", fingers=None) -> tuple[str, str | None]:
    """部位名 → (归一化部位, 左右)。左右一律是受检者本人的左右。"""
    p = (part or "").lower().replace("-", "_").replace(" ", "_")
    side = "right" if "right" in p else "left" if "left" in p else None
    if fingers or "finger" in p or any(f in p.split(".")[-1] for f in FINGERS if f != "thumb") or p.endswith("thumb"):
        return "finger", side
    if "neck" in p or "throat" in p:
        return "neck", side
    if "upper_arm" in p or "bicep" in p or "shoulder" in p:
        return "upper_arm", side
    if "forearm" in p or "lower_arm" in p or "wrist" in p:
        return "forearm", side
    if "palm" in p:
        return "palm", side
    if "hand" in p:
        return ("palm" if view == "palms" else "hand_back"), side
    if "arm" in p:
        return "forearm", side
    if "face" in p or "head" in p or "ear" in p:
        return "face", side
    return "other", side


def measure(bbox, meta: dict | None, pol: dict | None = None) -> dict | None:
    """框 → 毫米。meta: {mm_per_px, img_w, img_h, scale_source}。缺任何一项返回 None。"""
    if not bbox or not meta or not meta.get("mm_per_px"):
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    fmt = (pol or policy())["measure"].get("bbox_format", "rel1000")
    if fmt == "rel1000":
        if not meta.get("img_w") or not meta.get("img_h"):
            return None
        sx, sy = meta["img_w"] / 1000.0, meta["img_h"] / 1000.0
    else:
        sx = sy = 1.0
    k = float(meta["mm_per_px"])
    w, h = abs(x2 - x1) * sx * k, abs(y2 - y1) * sy * k
    if w <= 0 or h <= 0:
        return None
    return {"w_mm": round(w, 1), "h_mm": round(h, 1), "area_mm2": round(w * h, 1), "source": meta.get("scale_source") or "关键点估算"}


def _size_from_attrs(attrs: dict) -> dict | None:
    v = attrs.get("size_cm")
    try:
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            w, h = float(v[0]) * 10, float(v[1]) * 10
        elif isinstance(v, (int, float)):
            w = h = float(v) * 10
        else:
            return None
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return {"w_mm": round(w, 1), "h_mm": round(h, 1), "area_mm2": round(w * h, 1), "source": "大模型目测"}


def make_mark(item: dict, view: str, meta: dict | None = None, pol: dict | None = None) -> Mark:
    """把解析后的 L2 条目（prompts.parse_reply 的 items）转成 Mark。"""
    attrs = dict(item.get("attrs") or {})
    label = item.get("label")
    kind = "stain" if label == "stain" else ("scar" if item.get("check") == "scar" else "tattoo")
    region, side = region_of(item.get("part", ""), view, item.get("fingers"))
    m = Mark(kind=kind, part=item.get("part", ""), region=region, side=side, fingers=list(item.get("fingers") or []),
             view=view, confidence=float(item.get("confidence") or 0.0), uncertain=bool(item.get("uncertain")),
             attrs=attrs, bbox=item.get("bbox"))
    m.size = measure(m.bbox, meta, pol) or _size_from_attrs(attrs)
    return m


# ---------------- 单处判定 ----------------

def size_class(m: Mark, ref: str, pol: dict | None = None) -> tuple[str | None, dict]:
    """与参照物比较：small / near / large / None（无法判断）。"""
    pol = pol or policy()
    band = pol["measure"].get("borderline", 0.3)
    area = ref_area_mm2(ref, pol)
    info: dict = {"ref": ref}
    measured = None
    if m.size:
        ratio = m.size["area_mm2"] / area
        info.update(ratio=round(ratio, 2), area_cm2=round(m.size["area_mm2"] / 100, 2), source=m.size["source"])
        measured = "large" if ratio >= 1 + band else "small" if ratio <= 1 - band else "near"
    vs = {"smaller": "small", "near": "near", "similar": "near", "larger": "large"}.get(str(m.attrs.get("vs_ref") or "").lower())
    if vs:
        info["vs_ref"] = vs
    if measured and vs and {measured, vs} == {"small", "large"}:
        return "near", {**info, "conflict": True}     # 测量与模型目测相反 → 当临界处理，交人工
    return measured or vs, info


def _fmt_size(info: dict) -> str:
    ref = "身份证" if info.get("ref") == "id_card" else " 1 分硬币"
    if "area_cm2" in info:
        return f"约 {info['area_cm2']} cm²，是{ref}的 {info['ratio']} 倍（按{info['source']}估算）"
    if info.get("vs_ref"):
        return {"small": f"目测小于{ref}", "near": f"目测与{ref}相近", "large": f"目测不小于{ref}"}[info["vs_ref"]]
    return "大小无法估计"


def eval_tattoo(m: Mark, pol: dict | None = None) -> tuple[str, list[tuple[str, str]], dict]:
    """返回 (动作, [(动作, 说明)], 尺寸信息)。"""
    pol = pol or policy()
    regions = pol["tattoo"]["regions"]
    reg = regions.get(m.region) or regions["other"]
    hits: list[tuple[str, str]] = []
    content = str(m.attrs.get("content") or "unclear")
    cdef = pol["content"].get(content) or pol["content"]["unclear"]
    if cdef["action"] == BLOCK:
        hits.append((BLOCK, f"攻击性 / 违规内容：{cdef['name']}（无论面积大小均不录用）"))
    elif cdef["action"] == REVIEW:
        hits.append((REVIEW, f"{cdef['name']}，需核验内容"))
    else:
        hits.append((PASS, cdef["name"].split("（")[0]))
    ink = m.attrs.get("ink")
    info: dict = {}
    if "action" in reg:      # 按部位一刀切：脖子、面部、其他部位
        extra = "（即使面积小也不录用）" if m.region == "neck" else ""
        hits.append((reg["action"], f"{reg['name']}纹身{extra}"))
    else:
        ref = reg.get("size_ref", "id_card")
        cls, info = size_class(m, ref, pol)
        faded = ink == "faded"
        ref_cn = REF_CN[ref]
        if cls == "large":
            hits.append((reg["large_faded"], f"面积 ≥ {ref_cn.strip()}，但已洗过、色素明显变淡") if faded
                        else (reg["large"], f"面积 ≥ {ref_cn.strip()}"))
        elif cls == "near":
            why = f"按关键点估算的面积与大模型目测（与{ref_cn}比）不一致" if info.get("conflict") else f"面积接近{ref_cn}（临界）"
            hits.append((PASS if faded and reg["large_faded"] == PASS else REVIEW, f"{why}，以核验为准"))
        elif cls == "small":
            hits.append((PASS, f"面积小于{ref_cn}"))
        else:
            hits.append((pol["tattoo"].get("unknown_size", REVIEW), "无法估计面积"))
        if ink == "dark":
            hits.append((reg["dark"], "颜色过深" + ("，最终以核验为准" if reg["dark"] == REVIEW else "")))
        elif ink in ("light", "faded"):
            hits.append((PASS, "颜色浅" if ink == "light" else "洗过褪色的淡痕"))
        elif reg["dark"] == BLOCK:
            hits.append((REVIEW, "颜色深浅无法判断"))
        if reg.get("single") and not any(a == BLOCK for a, _ in hits):
            hits.append((reg["single"], f"{reg['name']}属外露高频部位，作业时持续外露"))
    if m.attrs.get("covers_scar"):
        hits.append((pol["tattoo"].get("covers_scar", BLOCK), "纹身覆盖 / 搭配疤痕"))
    if m.attrs.get("sticker"):
        hits.append((pol["tattoo"].get("sticker", REVIEW), "疑似贴纸纹身，需核验是否可去除"))
    if m.uncertain:
        hits.append((REVIEW, "模型对是否为纹身不确定（可能是疤痕、胎记或污渍）"))
    return worst_action(a for a, _ in hits), hits, info


def _length_mm(m: Mark) -> float | None:
    if m.size:
        return max(m.size["w_mm"], m.size["h_mm"])
    try:
        return float(m.attrs["length_cm"]) * 10 if m.attrs.get("length_cm") is not None else None
    except (TypeError, ValueError):
        return None


def eval_scar(m: Mark, pol: dict | None = None, fine_work: bool = True) -> tuple[str, list[tuple[str, str]], dict]:
    pol = pol or policy()
    s = pol["scar"]
    hits: list[tuple[str, str]] = []
    pattern = m.attrs.get("pattern") or "single"
    try:
        count = int(m.attrs.get("count") or 1)
    except (TypeError, ValueError):
        count = 1
    hyper = m.attrs.get("hypertrophic") or "none"
    length = _length_mm(m)
    info: dict = {"length_mm": round(length, 1) if length else None, "count": count}
    if m.size:
        info.update(area_cm2=round(m.size["area_mm2"] / 100, 2), source=m.size["source"])
    if pattern == "parallel_multiple":
        hits.append((s["parallel_multiple"], "多条整齐平行划痕（疑似人为自伤痕迹）"))
    if hyper == "severe":
        hits.append((s["hypertrophic_severe"], "严重增生，影响手部 / 手臂灵活操作"))
    elif hyper == "mild":
        hits.append((s["hypertrophic_mild"], "轻度增生"))
    if m.attrs.get("with_tattoo"):
        hits.append((s["with_tattoo"], "疤痕同时搭配纹身"))
    multiple = count >= 2 or pattern == "multiple_scattered"
    if multiple and m.region in ("hand_back", "palm"):
        hits.append((s["hand_back_multiple"], f"{REGION_CN[m.region]}多处陈旧疤痕"))
    elif multiple and pattern != "parallel_multiple":
        hits.append((REVIEW, "多处疤痕"))
    large_area = m.size and m.size["area_mm2"] >= ref_area_mm2("id_card", pol)
    if large_area or (length and length >= s["large_mm"]):
        hits.append((s["large"], "大面积长条疤痕"))
    elif length and length >= s["long_mm"]:
        if m.attrs.get("surgical"):
            hits.append((s["surgical_long_fine_work"] if fine_work else s["surgical_long_other"],
                         "长条手术刀疤" + ("（精细作业岗位）" if fine_work else "")))
        else:
            hits.append((s["medium"], "长条疤痕"))
    elif length and length >= s["short_mm"]:
        hits.append((s["medium"], f"疤痕长约 {length / 10:.1f} cm，介于短小与长条之间"))
    elif length is None:
        hits.append((s.get("unknown_size", REVIEW), "无法估计疤痕长度"))
    elif not multiple and pattern == "single" and hyper == "none":
        hits.append((s["short_single"], "单条短小、孤立、无增生的意外伤疤"))
    if m.uncertain:
        hits.append((REVIEW, "模型对是否为疤痕不确定（可能是纹身、胎记、淤青或污渍）"))
    return worst_action(a for a, _ in hits), hits, info


def eval_mark(m: Mark, pol: dict | None = None, params: dict | None = None) -> tuple[str, list[tuple[str, str]], dict]:
    pol = pol or policy()
    if m.kind == "stain":
        return pol["stain"].get("action", CLEAN), [(pol["stain"].get("action", CLEAN), pol["stain"]["message"].format(部位=m.where))], {}
    if m.kind == "scar":
        return eval_scar(m, pol, fine_work=bool((params or {}).get("scar", {}).get("fine_work", True)))
    return eval_tattoo(m, pol)


# ---------------- 多处 / 跨视角规则 ----------------

def count_rules(marks: list[Mark], pol: dict | None = None) -> list[tuple[str, str, str, list[str]]]:
    """数量与组合规则。marks 可以来自一个视角，也可以是整场检测所有视角（按部位去重）。
    返回 [(所属检测项, 动作, 说明, 部位列表)]。"""
    pol = pol or policy()
    regions = pol["tattoo"]["regions"]
    out = []
    tattoos = [m for m in marks if m.kind == "tattoo"]
    # 手指：按（哪只手, 哪根手指）去重；没说是哪根手指的每条算一处
    fingers, unnamed = set(), 0
    for m in tattoos:
        if m.region != "finger":
            continue
        if m.fingers:
            fingers |= {(m.side, f) for f in m.fingers}
        else:
            unnamed += 1
    n = len(fingers) + unnamed
    mx = regions["finger"].get("max_count")
    if mx is not None and n > mx:
        out.append(("tattoo", BLOCK, f"手指纹身 {n} 处（≥ {mx + 1} 处不录用）", sorted(f"{s or 'unknown'}_hand.{f}" for s, f in fingers)))
    # 手背 / 手掌：同一只手多处（不同视角拍到的同一处无法可靠去重，按单个视角里的最多处数计）
    for region in ("hand_back", "palm"):
        mx = regions[region].get("max_count")
        if mx is None:
            continue
        for side in ("right", "left", None):
            per_view: dict[str, int] = {}
            for m in tattoos:
                if m.region == region and m.side == side:
                    per_view[m.view] = per_view.get(m.view, 0) + int(m.attrs.get("count") or 1)
            k = max(per_view.values(), default=0)
            if k > mx:
                where = (SIDE_CN.get(side or "", "") or "") + REGION_CN[region]
                out.append(("tattoo", BLOCK, f"{where}多处纹身叠加（{k} 处），视觉观感突出", [f"{side or 'unknown'}_{region}"]))
    # 疤痕与纹身在同一部位
    scars = {(m.region, m.side) for m in marks if m.kind == "scar"}
    both = sorted({(m.region, m.side) for m in tattoos} & scars, key=str)
    if both:
        act = pol["scar"].get("with_tattoo", BLOCK)
        for region, side in both:
            out.append(("scar", act, f"{({'right': '右', 'left': '左'}.get(side or '', ''))}{REGION_CN.get(region, region)}疤痕同时搭配纹身",
                        [f"{side or 'unknown'}_{region}"]))
    return out


# ---------------- 转成 CheckResult ----------------

def _part_of(m: Mark) -> list[str]:
    side = m.side or "unknown"
    if m.region == "finger":
        return [f"{side}_hand.{f}" for f in m.fingers] or [f"{side}_hand"]
    return [f"{side}_{m.region}" if m.region not in ("neck", "face", "other") else (m.region if m.region != "other" else m.part or "other")]


def mark_finding(check: str, m: Mark, action: str, hits, info: dict, source: str) -> Finding:
    desc = []
    if m.kind == "tattoo":
        c = m.attrs.get("content")
        cname = (policy()["content"].get(c) or {}).get("name", "").split("（")[0] if c else ""
        desc = [x for x in (cname, {"faded": "已褪色", "light": "浅色", "dark": "深色"}.get(m.attrs.get("ink"), ""),
                            f"文字“{m.attrs['text']}”" if m.attrs.get("text") else "") if x]
    if info.get("ref") or info.get("area_cm2"):
        desc.append(_fmt_size(info) if info.get("ref") else f"约 {info['area_cm2']} cm²")
    if info.get("length_mm"):
        desc.append(f"长约 {info['length_mm'] / 10:.1f} cm")
    kind_cn = {"tattoo": "纹身", "scar": "疤痕", "stain": "疑似污渍"}[m.kind]
    reasons = [t for a, t in hits if a == action] if action != PASS else [t for _, t in hits]
    note = f"{m.where}{kind_cn}" + (f"（{'，'.join(desc)}）" if desc else "") + f" → {ACTION_CN[action]}：{'；'.join(reasons)}"
    value = {"verdict": action, "kind": m.kind, "region": m.region, "side": m.side, "fingers": m.fingers, "view": m.view,
             "rules": [{"action": a, "text": t} for a, t in hits], "size": m.size, "size_check": info, "attrs": m.attrs,
             "bbox": m.bbox}
    conf = m.confidence * (0.5 if m.uncertain and action != PASS else 1.0)   # 拿不准的问题项置信度减半，仍送人工
    return Finding(check, _part_of(m)[0], m.kind, action != PASS, round(conf, 3), source, value, note)


def policy_result(check: str, items: list[dict], view: str, source: str, quality_ok: bool = True,
                  meta: dict | None = None, reason: str = "", other_items: list[dict] | None = None) -> CheckResult:
    """把一个视角里某检测项（tattoo / scar）的 L2 条目按录用标准判成 CheckResult。

    other_items：同一视角另一检测项的条目（纹身 ↔ 疤痕），只用于"疤痕同时搭配纹身"这类组合规则。
    positive：True = 有不符合标准或需核验的标记；False = 没有标记或全部符合标准；None = 无法判断（含疑似污渍待重拍）。
    每条 Finding.value 里有 verdict（pass / review / block / clean）和命中的条款，仲裁的 policy 策略读这些。
    """
    pol = policy()
    params = (meta or {}).get("params") or {}
    if not items:
        return CheckResult(check, source, None, 0.0, [], quality_ok, [], "模型未返回该项")
    findings, marks, unsure = [], [], False
    for i in items:
        if i["label"] == "none":
            unsure |= i["uncertain"]
            findings.append(Finding(check, i["part"] or "all_visible", "none", None if i["uncertain"] else False,
                                    i["confidence"], source, None))
            continue
        if i["label"] not in ("tattoo", "scar", "stain"):
            unsure = True
            findings.append(Finding(check, i["part"], i["label"], None, i["confidence"], source, None, "模型表示无法确定"))
            continue
        m = make_mark(i, view, meta, pol)
        action, hits, info = eval_mark(m, pol, params)
        marks.append(m)
        findings.append(mark_finding(check, m, action, hits, info, source))
    others = [make_mark(i, view, meta, pol) for i in other_items or [] if i["label"] in ("tattoo", "scar")]
    for scope, action, text, parts in count_rules(marks + others, pol):
        if scope == check:
            findings.append(Finding(check, parts[0] if parts else "all_visible", "rule", action != PASS, 0.9, source,
                                    {"verdict": action, "rules": [{"action": action, "text": text}], "parts": parts}, text))
    verdicts = [(f, f.value["verdict"]) for f in findings if isinstance(f.value, dict) and f.value.get("verdict")]
    bad = [f for f, v in verdicts if v in (BLOCK, REVIEW)]
    stains = [f for f, v in verdicts if v == CLEAN]
    if bad:
        return CheckResult(check, source, True, round(max(f.confidence for f in bad), 3), sorted({f.part for f in bad}),
                           quality_ok, findings, "；".join(f.note for f in bad)[:400])
    if stains:
        return CheckResult(check, source, None, 0.0, sorted({f.part for f in stains}), False, findings,
                           "；".join(f.note for f in stains)[:400])
    if unsure:
        return CheckResult(check, source, None, 0.0, [], quality_ok, findings, "模型表示无法确定")
    confs = [f.confidence for f in findings]
    note = "；".join(f.note for f in findings if f.note) or reason
    return CheckResult(check, source, False, round(min(confs), 3) if confs else 0.0, [], quality_ok, findings, note)


# ---------------- 整场检测汇总（跨视角） ----------------

def marks_from_result(r: CheckResult | dict | None) -> list[Mark]:
    """从（单视角）CheckResult 的 findings 还原 Mark，用于跨视角合并。"""
    if r is None:
        return []
    if isinstance(r, dict):
        r = CheckResult.from_dict(r)
    out = []
    for f in r.findings:
        v = f.value if isinstance(f.value, dict) else None
        if not v or "kind" not in v:
            continue
        out.append(Mark(kind=v["kind"], part=f.part, region=v.get("region") or "other", side=v.get("side"),
                        fingers=list(v.get("fingers") or []), view=v.get("view") or "", confidence=f.confidence,
                        attrs=v.get("attrs") or {}, bbox=v.get("bbox"), size=v.get("size")))
    return out


@dataclass
class SessionReview:
    check: str
    action: str                      # 跨视角规则的结论（pass 表示没有额外命中）
    rules: list[dict]                # [{action, text, parts}]
    marks: list[dict]                # 各视角的标记（报告用）
    reminders: list[dict]            # 清洁提醒 [{view, part, message}]

    def to_dict(self) -> dict:
        return asdict(self)


def session_review(frames: dict[str, dict]) -> dict[str, SessionReview]:
    """整场检测汇总。frames: {视角: {检测项: 该视角的 L2 CheckResult（或其 dict）}}。

    单视角内的规则各视角已经判过，这里只补充跨视角才能发现的（如左手背面拍到一处手指纹身、右手掌面又拍到一处），
    并汇总所有标记和清洁提醒供报告使用。返回 {tattoo: SessionReview, scar: SessionReview}。
    """
    listed: dict[str, list] = {c: [] for c in POLICY_CHECKS}
    reminders: dict[str, list] = {c: [] for c in POLICY_CHECKS}
    all_marks: list[Mark] = []
    per_view: dict[str, list[Mark]] = {}
    for view, by_check in frames.items():
        for check in POLICY_CHECKS:
            r = (by_check or {}).get(check)
            if r is None:
                continue
            rd = r if isinstance(r, dict) else r.to_dict()
            for f in rd.get("findings") or []:
                v = f.get("value") if isinstance(f.get("value"), dict) else None
                if not v or "verdict" not in v:
                    continue
                if "kind" in v:
                    listed[check].append({"view": view, "part": f["part"], "kind": v["kind"], "verdict": v["verdict"],
                                          "note": f.get("note", ""), "size": v.get("size"), "size_check": v.get("size_check"),
                                          "attrs": v.get("attrs"), "bbox": v.get("bbox")})
                if v["verdict"] == CLEAN:
                    reminders[check].append({"view": view, "part": f["part"], "message": f.get("note", "")})
            ms = marks_from_result(rd)
            all_marks += ms
            per_view.setdefault(view, []).extend(ms)
    already = {(scope, text) for ms in per_view.values() for scope, _, text, _ in count_rules(ms)}
    rules: dict[str, list] = {c: [] for c in POLICY_CHECKS}
    for scope, action, text, parts in count_rules(all_marks):
        if (scope, text) not in already:
            rules[scope].append({"action": action, "text": text, "parts": parts})
    return {c: SessionReview(c, worst_action(r["action"] for r in rules[c]), rules[c], listed[c], reminders[c]) for c in POLICY_CHECKS}
