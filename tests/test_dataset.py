import base64
import json

import numpy as np
import pytest

from limbcheck.dataset import (Sample, import_demo_reports, import_folder, load_samples, read_jsonl, save_image,
                               truth_finger_parts, truth_positive, write_labels_template)


def _write_imgs(d, names):
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        save_image(d / n, np.full((40, 60, 3), 200, np.uint8))


def test_labels_template_guesses_view(tmp_path):
    _write_imgs(tmp_path / "raw", ["v01_palms.jpg", "v01_手背.jpg", "v01_stand.jpg", "v01_背面.jpg", "x.jpg"])
    write_labels_template(tmp_path / "raw", tmp_path / "labels.csv")
    text = (tmp_path / "labels.csv").read_text(encoding="utf-8-sig")
    rows = {line.split(",")[0]: line.split(",")[1] for line in text.splitlines()[1:]}
    assert rows == {"v01_palms.jpg": "palms", "v01_手背.jpg": "backs", "v01_stand.jpg": "stand", "v01_背面.jpg": "back", "x.jpg": ""}


def test_import_folder(tmp_path):
    _write_imgs(tmp_path / "raw", ["a.jpg", "b.jpg", "c.jpg"])
    (tmp_path / "labels.csv").write_text(
        "image,view,session_id,subject_id,side,right_missing,left_missing,right_nails,left_nails,tattoo,scar,hair,limbs,lighting\n"
        "a.jpg,palms,s1,v1,,无名指、小指,none,,,,,,,dark\n"
        "b.jpg,backs,s1,v1,,none,none,0,1,,,,,dark\n"
        "c.jpg,stand,s1,v1,,,,,,是,否,dyed,0,bright\n", encoding="utf-8")
    samples = import_folder(tmp_path / "raw", tmp_path / "labels.csv", tmp_path / "ds")
    assert len(samples) == 3
    a, b, c = load_samples(tmp_path / "ds")
    assert a.truth["fingers"] == {"right": ["ring", "little"], "left": []}
    assert truth_positive(a, "fingers") is True and truth_finger_parts(a) == {"right_hand.ring", "right_hand.little"}
    assert truth_positive(b, "fingers") is False and truth_positive(b, "nails") is True
    assert truth_positive(c, "tattoo") is True and truth_positive(c, "scar") is False and truth_positive(c, "hair") is True
    assert truth_positive(c, "fingers") is None          # 未标注
    assert a.meta["lighting"] == "dark" and a.meta["source"] == "folder"
    assert (tmp_path / "ds" / a.image).exists()


def test_import_folder_reports_bad_rows(tmp_path):
    _write_imgs(tmp_path / "raw", ["a.jpg"])
    (tmp_path / "labels.csv").write_text("image,view,right_missing\na.jpg,palms,第六指\nmissing.jpg,foo,\n", encoding="utf-8")
    with pytest.raises(ValueError) as e:
        import_folder(tmp_path / "raw", tmp_path / "labels.csv", tmp_path / "ds")
    assert "第 2 行" in str(e.value) and "第 3 行" in str(e.value)


def _demo_report(sid="20261025-1200-ab"):
    jpg = base64.b64encode(b"\xff\xd8fakejpeg").decode()
    return {
        "schema": "limbcheck.demo_report/v2", "session_id": sid, "subject_id": "v07", "validation": True,
        "job": {"id": "precision", "name": "精密装配"}, "duration_ms": 150000,
        "truth": {"fingers": {"right": [], "left": ["index"]}, "nails": {"right": False, "left": False}, "attack": "none"},
        "meta": {"lighting": "normal", "device": "laptop"},
        "steps": [{"id": "palms", "status": "review", "ms": 8000}, {"id": "challenge", "status": "pass", "ms": 6000, "metrics": {}}],
        "frames": [{"view": "palms", "img": "data:image/jpeg;base64," + jpg,
                    "l1": {"fingers": {"check": "fingers", "source": "L1:mediapipe-web", "positive": True, "confidence": 0.8, "parts": ["left_hand.index"]}}}],
        "reviews": [{"view": "palms", "latency_ms": 2100, "results": {"fingers": {"check": "fingers", "source": "L2:x", "positive": True, "confidence": 0.7, "parts": ["left_hand"]}}}],
        "arbitration_by_view": {"palms": {"fingers": {"decision": "fail_suggested", "shadow": "fail_suggested", "reason": "两路一致"}}},
        "perf": {"hand_ms": {"p95": 30}},
    }


def test_import_demo(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps(_demo_report(), ensure_ascii=False), encoding="utf-8")
    s, sess = import_demo_reports([p, p], tmp_path / "ds")      # 重复导入同一会话应去重
    assert len(s) == 1 and len(sess) == 1
    assert s[0].id.endswith("_palms") and s[0].meta["source"] == "demo"
    preds = read_jsonl(tmp_path / "ds" / "predictions" / "demo_browser.jsonl")
    assert preds[0]["results"]["fingers"]["shadow"] == "fail_suggested"
    assert read_jsonl(tmp_path / "ds" / "sessions.jsonl")[0]["challenge"]["status"] == "pass"


def test_reject_old_demo_format(tmp_path):
    p = tmp_path / "old.json"
    p.write_text(json.dumps({"job": "x", "results": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        import_demo_reports([p], tmp_path / "ds")


def test_hand_view_uses_side():
    s = Sample(id="x", image="x", view="hand", session_id="x", truth={"fingers": {"right": [], "left": None}}, meta={"side": "right"})
    assert truth_positive(s, "fingers") is False
