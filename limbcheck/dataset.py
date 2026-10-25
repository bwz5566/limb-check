"""评测数据集：清单格式、真值、导入（网页 demo 导出的报告 / 图片文件夹 + 标注表）。

数据集目录：
    data/<名称>/
        manifest.jsonl          每行一张图（样本），见 Sample
        sessions.jsonl          每行一次检测会话（仅 demo 导入时有：步骤用时、挑战结果、端侧性能等）
        images/                 图片
        predictions/demo_browser.jsonl   浏览器端 L1 与 L2 的原始结果（仅 demo 导入时有）

真值（truth）字段，任何字段为 null 表示"未标注"，不参与该项统计：
    fingers: {"right": ["index"], "left": []}     缺失或明显短缩的手指（thumb/index/middle/ring/little）
    nails:   {"right": false, "left": true}         是否有美甲（甲油、甲片、延长甲）
    tattoo / scar: true | false                     是否有可见纹身 / 明显疤痕
    hair:    "natural" | "dyed" | "highlights"      发色
    limbs:   true | false                           是否有肢体缺失或明显异常
    attack:  "none" | "replay" | "photo" | "virtual_camera" | "other"   （会话级）防作弊测试类型
"""
from __future__ import annotations

import base64
import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .core import CN_FINGER, FINGERS, HAND_VIEWS, IMAGE_CHECKS, SIDES, VIEW_CHECKS

TRUTH_KEYS = ("fingers", "nails", "tattoo", "scar", "hair", "limbs", "attack")   # 另加 checks.json 里新增检测项的同名键（是/否）


@dataclass
class Sample:
    id: str
    image: str                      # 相对数据集目录的路径
    view: str
    session_id: str
    subject_id: str = ""
    truth: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------- 读写 ----------------

def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_samples(data_dir: str | Path) -> list[Sample]:
    rows = read_jsonl(Path(data_dir) / "manifest.jsonl")
    if not rows:
        raise FileNotFoundError(f"{data_dir}/manifest.jsonl 不存在或为空")
    return [Sample(id=r["id"], image=r["image"], view=r["view"], session_id=r.get("session_id") or r["id"],
                   subject_id=r.get("subject_id") or "", truth=r.get("truth") or {}, meta=r.get("meta") or {}) for r in rows]


def load_sessions(data_dir: str | Path) -> list[dict]:
    return read_jsonl(Path(data_dir) / "sessions.jsonl")


def load_image(path: str | Path) -> np.ndarray:
    """读取为 RGB uint8。用 imdecode 以支持 Windows 中文路径。"""
    import cv2

    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片 {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(path: str | Path, img: np.ndarray, quality: int = 92) -> None:
    import cv2

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(p.suffix or ".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError(f"无法编码图片 {path}")
    buf.tofile(str(p))


# ---------------- 真值 ----------------

def expected_sides(sample: Sample) -> tuple[str, ...]:
    if sample.view in ("palms", "backs"):
        return SIDES
    if sample.view == "hand":
        side = sample.meta.get("side")
        return (side,) if side in SIDES else SIDES
    return ()


def truth_positive(sample_or_truth, check: str, sides: tuple[str, ...] = SIDES) -> bool | None:
    """该检测项的真值：True 有问题 / False 正常 / None 未标注。"""
    t = sample_or_truth.truth if isinstance(sample_or_truth, Sample) else sample_or_truth
    if isinstance(sample_or_truth, Sample):
        sides = expected_sides(sample_or_truth) or sides
    v = t.get(check)
    if v is None:
        return None
    if check == "fingers":
        if not isinstance(v, dict) or any(v.get(s) is None for s in sides):
            return None
        return any(bool(v.get(s)) for s in sides)
    if check == "nails":
        if not isinstance(v, dict) or any(v.get(s) is None for s in sides):
            return None
        return any(bool(v.get(s)) for s in sides)
    if check == "hair":
        return v != "natural"
    if check == "attack":
        return v not in ("none", "", None)
    return bool(v)


def truth_finger_parts(sample: Sample) -> set[str] | None:
    v = sample.truth.get("fingers")
    if not isinstance(v, dict):
        return None
    sides = expected_sides(sample) or SIDES
    if any(v.get(s) is None for s in sides):
        return None
    return {f"{s}_hand.{f}" for s in sides for f in v.get(s) or []}


# ---------------- 文件夹 + 标注表 ----------------

LABEL_COLUMNS = ["image", "view", "session_id", "subject_id", "side", "right_missing", "left_missing",
                 "right_nails", "left_nails", "tattoo", "scar", "hair", "limbs",
                 "skin_tone", "lighting", "device", "scenario", "note"]
VIEW_PATTERNS = [
    (r"palm|掌心", "palms"), (r"backs|back_of_hand|手背", "backs"), (r"arms?_?up|上举", "arms_up"),
    (r"arms|t_?pose|侧平举", "arms"), (r"stand|front|正面", "stand"), (r"(?<!hand_)back|背面", "back"), (r"hand|手", "hand"),
]
BUILTIN_TRUTH = ("fingers", "nails", "tattoo", "scar", "hair", "limbs")
EXTRA_CHECKS = tuple(c for c in IMAGE_CHECKS if c not in BUILTIN_TRUTH)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRUE_WORDS = {"1", "y", "yes", "true", "是", "有", "t"}
FALSE_WORDS = {"0", "n", "no", "false", "否", "无", "f"}
NONE_WORDS = {"none", "无", "0", "no", "-"}


def guess_view(name: str) -> str:
    low = name.lower()
    for pat, view in VIEW_PATTERNS:
        if re.search(pat, low):
            return view
    return ""


def write_labels_template(image_dir: str | Path, out_csv: str | Path) -> int:
    image_dir = Path(image_dir)
    files = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        w.writeheader()
        for p in files:
            view = guess_view(p.stem)
            hand = view in HAND_VIEWS
            w.writerow({"image": p.relative_to(image_dir).as_posix(), "view": view, "session_id": p.stem,
                        "right_missing": "none" if hand else "", "left_missing": "none" if hand else "",
                        "right_nails": "0" if view in ("backs", "hand") else "", "left_nails": "0" if view in ("backs", "hand") else ""})
    return len(files)


def _bool(v: str):
    v = (v or "").strip().lower()
    if v in TRUE_WORDS:
        return True
    if v in FALSE_WORDS:
        return False
    if v == "":
        return None
    raise ValueError(f"无法识别的是/否取值 {v!r}")


def _fingers(v: str):
    v = (v or "").strip()
    if v == "":
        return None
    if v.lower() in NONE_WORDS:
        return []
    out = []
    for tok in re.split(r"[;,，、/\s]+", v):
        if not tok:
            continue
        key = CN_FINGER.get(tok, tok.lower())
        if key not in FINGERS:
            raise ValueError(f"无法识别的手指 {tok!r}（可用 {FINGERS} 或 拇指/食指/中指/无名指/小指）")
        out.append(key)
    return out


def import_folder(image_dir: str | Path, labels_csv: str | Path, out_dir: str | Path, source: str = "folder") -> list[Sample]:
    """把图片文件夹 + 标注表转换为数据集。图片按原相对路径复制到 out_dir/images/。"""
    import shutil

    image_dir, out_dir = Path(image_dir), Path(out_dir)
    samples, errors = [], []
    with open(labels_csv, encoding="utf-8-sig", newline="") as f:
        for n, row in enumerate(csv.DictReader(f), start=2):
            try:
                img = (row.get("image") or "").strip()
                view = (row.get("view") or "").strip()
                if view not in VIEW_CHECKS:
                    raise ValueError(f"view 必须是 {list(VIEW_CHECKS)} 之一，实际为 {view!r}")
                src = image_dir / img
                if not src.exists():
                    raise ValueError(f"找不到图片 {src}")
                truth = {
                    "fingers": {"right": _fingers(row.get("right_missing")), "left": _fingers(row.get("left_missing"))},
                    "nails": {"right": _bool(row.get("right_nails")), "left": _bool(row.get("left_nails"))},
                    "tattoo": _bool(row.get("tattoo")),
                    "scar": _bool(row.get("scar")),
                    "hair": (row.get("hair") or "").strip() or None,
                    "limbs": _bool(row.get("limbs")),
                }
                for c in EXTRA_CHECKS:      # configs/checks.json 里新增的检测项：标注表加一列同名的是/否
                    if c in row:
                        truth[c] = _bool(row.get(c))
                if truth["hair"] not in (None, "natural", "dyed", "highlights"):
                    raise ValueError(f"hair 只能是 natural / dyed / highlights，实际为 {truth['hair']!r}")
                meta = {k: v for k, v in row.items() if k not in LABEL_COLUMNS[:13] and k not in EXTRA_CHECKS and v not in (None, "")}
                if (row.get("side") or "").strip():
                    meta["side"] = row["side"].strip()
                meta["source"] = source
                sid = (row.get("session_id") or "").strip() or Path(img).stem
                dst = Path("images") / img
                (out_dir / dst).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, out_dir / dst)
                samples.append(Sample(id=f"{sid}__{Path(img).stem}" if sid != Path(img).stem else sid, image=dst.as_posix(), view=view,
                                      session_id=sid, subject_id=(row.get("subject_id") or "").strip(), truth=truth, meta=meta))
            except ValueError as e:
                errors.append(f"第 {n} 行：{e}")
    if errors:
        raise ValueError("标注表有错误：\n" + "\n".join(errors))
    write_jsonl(out_dir / "manifest.jsonl", [s.to_dict() for s in samples])
    return samples


# ---------------- 网页 demo 报告 ----------------

DEMO_SCHEMA = "limbcheck.demo_report/v2"


def _decode_data_url(url: str) -> tuple[bytes, str]:
    head, b64 = url.split(",", 1)
    ext = ".png" if "png" in head else ".jpg"
    return base64.b64decode(b64), ext


def import_demo_reports(paths: list[str | Path], out_dir: str | Path) -> tuple[list[Sample], list[dict]]:
    """导入网页 demo 在验证模式下导出的报告 JSON（可多份），追加到数据集。"""
    out_dir = Path(out_dir)
    samples = [Sample(**r) for r in read_jsonl(out_dir / "manifest.jsonl")]
    sessions = read_jsonl(out_dir / "sessions.jsonl")
    preds = read_jsonl(out_dir / "predictions" / "demo_browser.jsonl")
    seen = {s["session_id"] for s in sessions}
    new_samples, new_sessions = [], []
    for path in paths:
        rep = json.loads(Path(path).read_text(encoding="utf-8"))
        if rep.get("schema") != DEMO_SCHEMA:
            raise ValueError(f"{path} 不是 {DEMO_SCHEMA} 格式（请用 apps/web-demo 新版导出）")
        sid = rep["session_id"]
        if sid in seen:
            continue
        seen.add(sid)
        truth = rep.get("truth") or {}
        meta = {**(rep.get("meta") or {}), "source": "demo", "job": (rep.get("job") or {}).get("id")}
        reviews = {rv.get("view"): rv for rv in rep.get("reviews") or []}
        arb = rep.get("arbitration_by_view") or {}
        for fr in rep.get("frames") or []:
            view = fr["view"]
            data, ext = _decode_data_url(fr["img"])
            rel = Path("images") / f"{sid}_{view}{ext}"
            (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            (out_dir / rel).write_bytes(data)
            s = Sample(id=f"{sid}_{view}", image=rel.as_posix(), view=view, session_id=sid,
                       subject_id=rep.get("subject_id") or "", truth=truth, meta=meta)
            new_samples.append(s)
            rv = reviews.get(view) or {}
            results = {}
            for check, l1 in (fr.get("l1") or {}).items():
                results[check] = {"l1": l1, "l2": (rv.get("results") or {}).get(check), **(arb.get(view, {}).get(check) or {})}
            for check, l2 in (rv.get("results") or {}).items():
                results.setdefault(check, {"l1": None, "l2": l2, **(arb.get(view, {}).get(check) or {})})
            preds.append({"id": s.id, "view": view, "session_id": sid, "results": results,
                          "latency_ms": {"l2": rv.get("latency_ms")} if rv else {}, "pipeline": "browser"})
        steps = rep.get("steps") or []
        challenge = next((st for st in steps if st.get("id") == "challenge"), None)
        new_sessions.append({
            "session_id": sid, "subject_id": rep.get("subject_id") or "", "at": rep.get("at"),
            "job": (rep.get("job") or {}).get("id"), "validation": rep.get("validation", False),
            "truth": truth, "meta": meta, "duration_ms": rep.get("duration_ms"),
            "steps": [{k: st.get(k) for k in ("id", "status", "ms", "note")} for st in steps],
            "challenge": {"status": challenge.get("status"), "metrics": challenge.get("metrics")} if challenge else None,
            "perf": rep.get("perf"), "client": rep.get("client"),
            "l2_latency_ms": [rv.get("latency_ms") for rv in rep.get("reviews") or [] if rv.get("latency_ms")],
        })
    samples += new_samples
    sessions += new_sessions
    write_jsonl(out_dir / "manifest.jsonl", [s.to_dict() for s in samples])
    write_jsonl(out_dir / "sessions.jsonl", sessions)
    write_jsonl(out_dir / "predictions" / "demo_browser.jsonl", preds)
    return new_samples, new_sessions
