"""网页 demo 的 server.py：只用标准库、复用 limbcheck 的仲裁；接口在未配置大模型时也能工作。"""
import importlib.util
import json
import threading
import urllib.error
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from limbcheck.core import CheckResult, Finding

SERVER = Path(__file__).resolve().parent.parent / "apps" / "web-demo" / "server.py"


@pytest.fixture(scope="module")
def server(monkeypatch_module):
    for k in ("VLM_API_KEY", "VLM_BASE_URL", "VLM_MODEL"):
        monkeypatch_module.delenv(k, raising=False)
    spec = importlib.util.spec_from_file_location("webdemo_server", SERVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def base_url(server):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(server.Handler, directory=str(server.ROOT)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_arbitrate_session_modes(server):
    l1 = CheckResult("fingers", "L1:mediapipe-web", False, 0.95).to_dict()
    l2 = CheckResult("fingers", "L2:x", True, 0.9, ["left_hand.index"]).to_dict()
    tattoo = CheckResult("tattoo", "L2:x", False, 0.9).to_dict()
    out = server.arbitrate_session("precision", [
        {"view": "palms", "l1": {"fingers": l1}, "l2": {"fingers": l2}},
        {"view": "stand", "l1": {}, "l2": {"tattoo": tattoo}},
    ])
    f = out["by_view"]["palms"]["fingers"]
    assert f["decision"] == "review" and f["agree"] is False          # 两路不一致，最后一次尝试 → 人工复核
    assert f["shadow_first_attempt"] == "review"                       # 双方置信度都高，首次也不要求重做
    t = out["by_view"]["stand"]["tattoo"]
    assert t["mode"] == "judge" and t["decision"] == "pass"            # 纹身按录用标准判定（有书面依据）
    nails = CheckResult("nails", "L2:x", True, 0.9, ["right_hand"]).to_dict()
    h = server.arbitrate_session("line", [{"view": "backs", "l2": {"nails": nails}}])["by_view"]["backs"]["nails"]
    assert h["mode"] == "record_only" and h["decision"] == "not_assessed" and h["shadow"] == "fail_suggested"
    assert out["by_check"]["fingers"]["decision"] == "review"
    assert out["overall"] == "review"                                  # 只汇总 judge 模式的项目


def test_endpoints_without_vlm(server, base_url):
    with urllib.request.urlopen(base_url + "/api/config", timeout=10) as r:
        cfg = json.loads(r.read())
    assert cfg["vlm"] is False and cfg["web"]["jobs"] and cfg["prompt_version"]
    code, body = _post(base_url + "/api/review", {"view": "palms", "images": []})
    assert code == 503 and "VLM" in body["error"]
    code, body = _post(base_url + "/api/arbitrate", {"job": "line", "frames": [
        {"view": "backs", "l1": {"nails": CheckResult("nails", "L1:deltaE", True, 0.9, ["right_hand"]).to_dict()}}]})
    assert code == 200 and body["by_view"]["backs"]["nails"]["decision"] == "not_assessed"   # 普通产线美甲只记录


def test_models_route(server, base_url):
    if not (server.MODELS / "hand_landmarker.task").exists():
        pytest.skip("models/ 下没有模型文件")
    with urllib.request.urlopen(base_url + "/models/hand_landmarker.task", timeout=10) as r:
        assert r.status == 200 and len(r.read()) > 1_000_000
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(base_url + "/models/..%2F..%2Fpyproject.toml", timeout=10)


def test_session_required_views_and_cross_view_marks(server):
    fingers_ok = CheckResult("fingers", "L1:mediapipe-web", False, 0.95).to_dict()
    out = server.arbitrate_session("precision", [{"view": "palms", "l1": {"fingers": fingers_ok}}])
    f = out["by_check"]["fingers"]
    assert f["decision"] == "review" and any("双手手背" in n for n in f["notes"])     # 没拍手背 → 无法确认手背正常

    def finger_tattoo(side, finger):
        v = {"verdict": "pass", "kind": "tattoo", "region": "finger", "side": side, "fingers": [finger], "attrs": {}, "rules": []}
        return CheckResult("tattoo", "L2:x", False, 0.9, [], True,
                           [Finding("tattoo", f"{side}_hand.{finger}", "tattoo", False, 0.9, "L2:x", v, "")]).to_dict()

    out = server.arbitrate_session("precision", [
        {"view": "backs", "l1": {"fingers": fingers_ok}, "l2": {"tattoo": finger_tattoo("right", "index")}},
        {"view": "palms", "l1": {"fingers": fingers_ok}, "l2": {"tattoo": finger_tattoo("left", "ring")}},
    ])
    t = out["by_check"]["tattoo"]
    assert t["decision"] == "fail" and "手指纹身 2 处" in t["notes"][0]          # 两处都很确定 → 直接判定
    assert len(t["policy"]["marks"]) == 2


def test_session_coordination_from_motion(server):
    from test_motion import arms_step, march_step

    out = server.arbitrate_session("warehouse", [], motion={"aspect": 1.0, "steps": [arms_step(), march_step()]})
    c = out["by_check"]["coordination"]
    assert c["decision"] == "pass" and [a["verdict"] for a in c["motion"]] == ["normal", "normal"]
    out = server.arbitrate_session("warehouse", [], motion={"aspect": 1.0, "steps": [arms_step(right_up=110), march_step(n=3)]})
    c = out["by_check"]["coordination"]
    assert c["decision"] == "fail_suggested" and out["overall"] == "fail_suggested"     # 只有关键点一路，不自动判不符合
    assert "coordination" not in server.arbitrate_session("warehouse", [])["by_check"]


def test_env_file_does_not_override_environment(server, tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("# 注释\nVLM_API_KEY=sk-file\nVLM_MODEL=\nLIMBCHECK_TEST_X='quoted'\n", encoding="utf-8")
    monkeypatch.delenv("VLM_API_KEY", raising=False)
    monkeypatch.setenv("LIMBCHECK_TEST_X", "env")
    monkeypatch.delenv("VLM_MODEL", raising=False)
    server.load_env_file(f)
    import os
    assert os.environ["VLM_API_KEY"] == "sk-file" and os.environ["LIMBCHECK_TEST_X"] == "env" and "VLM_MODEL" not in os.environ
    monkeypatch.delenv("VLM_API_KEY")


def test_session_prefers_server_l1(server):
    """服务端对高清截图的判定优先；某项服务端没给结果时才用浏览器端。"""
    browser_ok = CheckResult("fingers", "L1:mediapipe-web", False, 0.95).to_dict()
    server_bad = CheckResult("fingers", "L1:rtmw+contour-skin", None, 0.0, quality_ok=False,
                             note="有手指蜷起或并拢").to_dict()
    nails_ok = CheckResult("nails", "L1:deltaE-web", False, 0.9).to_dict()
    fr = {"view": "backs", "l1": {"fingers": browser_ok, "nails": nails_ok}, "l1_server": {"fingers": server_bad}}
    out = server.arbitrate_session("precision", [fr, {"view": "palms", "l1": {"fingers": browser_ok}}])
    v = out["by_view"]["backs"]
    assert v["fingers"]["l1_from"] == "server" and v["fingers"]["decision"] != "pass"
    assert v["nails"]["l1_from"] == "browser"
    assert out["by_view"]["palms"]["fingers"]["l1_from"] == "browser"


def test_analyze_endpoint(server, base_url, monkeypatch):
    code, d = _post(base_url + "/api/analyze", {"view": "nope", "image": "x"})
    assert code == 400
    monkeypatch.setattr(server, "JUDGE", server.ServerJudge("off"))
    code, d = _post(base_url + "/api/analyze", {"view": "backs", "image": "data:image/jpeg;base64,AA=="})
    assert code == 503
    with urllib.request.urlopen(base_url + "/api/config", timeout=10) as r:
        assert json.loads(r.read())["server_judge"]["enabled"] is False


def test_analyze_runs_python_pipeline(server, base_url, monkeypatch):
    pytest.importorskip("mediapipe")
    import base64

    import cv2

    judge = server.ServerJudge("mediapipe-skin")
    judge.load()
    if judge.base is None:
        pytest.skip(f"服务端流水线加载失败：{judge.error}")
    monkeypatch.setattr(server, "JUDGE", judge)
    img = np.full((360, 640, 3), 200, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    uri = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    code, d = _post(base_url + "/api/analyze", {"view": "backs", "image": uri, "job": "precision"})
    assert code == 200 and d["pipeline"] == "mediapipe-skin" and d["img_size"] == [640, 360]
    # 画面里没有手 → 不能判"正常"
    assert d["l1"]["fingers"]["positive"] is None and d["l1"]["fingers"]["quality_ok"] is False
    code, d = _post(base_url + "/api/analyze", {"view": "backs", "image": "data:image/jpeg;base64,AAAA"})
    assert code == 400


def test_records_append_and_summary(server, tmp_path):
    def session(sid, steps, overall="pass", job="line"):
        return {"type": "session", "session_id": sid, "job": {"id": job, "name": job}, "duration_ms": 90000, "overall": overall,
                "steps": steps, "checks": {"fingers": {"decision": overall, "mode": "judge"}}}

    ok = {"id": "palms", "title": "双手掌心朝前", "status": "pass", "ms": 8000, "first_ok_ms": 5000, "hints": []}
    stuck = {"id": "pinch", "title": "拇指依次对捏", "status": "timeout", "ms": 30000, "note": "超时",
             "hints": [{"text": "右手：拇指碰食指", "ms": 20000}, {"text": "请把两只手分开", "ms": 3000}]}
    server.append_record(session("S1", [ok, stuck], "review"), tmp_path)
    server.append_record(session("S2", [ok, {**stuck, "status": "pass", "hints": [{"text": "右手：拇指碰食指", "ms": 4000}]}]), tmp_path)
    server.append_record(session("S2", [ok, {**stuck, "status": "pass", "hints": []}]), tmp_path)   # 重拍后的新记录覆盖旧的
    server.append_record({"type": "feedback", "session_id": "S1", "rating": "confusing", "steps": ["pinch"], "note": "不知道用哪只手"}, tmp_path)
    with pytest.raises(ValueError):
        server.append_record({"type": "nope", "session_id": "S3"}, tmp_path)
    with pytest.raises(ValueError):
        server.append_record({"type": "session"}, tmp_path)

    d = server.summarize_records(server.load_records(tmp_path))
    assert d["sessions"] == 2 and d["feedback_n"] == 1 and d["overall"] == {"review": 1, "pass": 1}
    pinch = d["steps"][0]                      # 完成率最低的排在最前
    assert pinch["id"] == "pinch" and pinch["pass_rate"] == 0.5 and pinch["confusing"] == 1
    assert pinch["top_hints"][0] == {"text": "右手：拇指碰食指", "total_s": 20.0, "sessions": 1}
    assert d["feedback"]["recent"][0]["note"] == "不知道用哪只手"
    assert server.summarize_records(server.load_records(tmp_path), job="warehouse")["sessions"] == 0


def test_records_endpoints(server, base_url, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "RECORDS_DIR", tmp_path)
    code, d = _post(base_url + "/api/records", {"type": "session", "session_id": "S9", "steps": [], "job": {"id": "line"}})
    assert code == 200 and d["ok"]
    code, d = _post(base_url + "/api/records", {"type": "session"})
    assert code == 400
    with urllib.request.urlopen(base_url + "/api/records/summary?days=7", timeout=10) as r:
        d = json.loads(r.read())
    assert d["sessions"] == 1 and d["dir"] == str(tmp_path)


def test_session_scope_skips_out_of_scope_required_views(server):
    limbs_ok = CheckResult("limbs", "L1:mediapipe-web", False, 0.95).to_dict()
    frames = [{"view": "stand", "l1": {"limbs": limbs_ok}}]
    full = server.arbitrate_session("line", frames)
    assert full["by_check"]["fingers"]["decision"] == "review"           # 全流程：没拍手 → 手部需复核
    body = server.arbitrate_session("line", frames, checks=["limbs", "tattoo", "scar", "hair", "coordination"])
    assert "fingers" not in body["by_check"] and body["overall"] == "pass"


def test_subject_meta_only_accepts_entered_gender(server):
    assert server.subject_meta({"gender": "male"}) == {"gender": "male"}
    assert server.subject_meta({"gender": "unknown", "x": 1}) == {} and server.subject_meta(None) == {}


def test_nails_without_l2_goes_to_review(server):
    ok = [{"check": "nails", "part": s, "label": "none", "positive": False, "confidence": 0.9, "source": "L1:deltaE",
           "value": {"kind": "nail_color", "verdict": "pass"}} for s in ("right_hand", "left_hand")]
    l1 = {"nails": CheckResult.from_dict({"check": "nails", "source": "L1:deltaE", "positive": False, "confidence": 0.9,
                                          "findings": ok}).to_dict()}
    out = server.arbitrate_session("precision", [{"view": "nails_close", "l1": l1}], checks=["nails"])
    e = out["by_check"]["nails"]
    assert e["decision"] == "review" and "大模型" in e["notes"][0]
    l2 = {"nails": CheckResult("nails", "L2:x", False, 0.9).to_dict()}
    assert server.arbitrate_session("precision", [{"view": "nails_close", "l1": l1, "l2": l2}], checks=["nails"])["by_check"]["nails"]["decision"] == "pass"


def test_arbitrate_session_summary(server):
    ok = CheckResult("fingers", "L2:x", False, 0.95).to_dict()
    steps = [{"id": "palms", "title": "双手掌心", "status": "pass"}, {"id": "flex", "title": "握拳 → 张开", "status": "timeout", "note": "超时"}]
    out = server.arbitrate_session("precision", [{"view": "palms", "l2": {"fingers": ok}}], checks=["fingers"], steps=steps)
    s = out["summary"]
    assert s["outcome"] == "manual" and any("握拳" in m["reason"] for m in s["manual"]) and [g["id"] for g in s["groups"]] == ["hands"]
    assert server.WEB_CONFIG["summary"]["fixable"]["nails"]


def test_report_s20261027_cases(server):
    """实测报告 S20261027-181043：两路都认为正常却交了人工复核的三种情况，现在都自动通过。"""
    fingers_kp = CheckResult("fingers", "L1:rtmw+hand", False, 0.753).to_dict()        # 小指比例临近阈值
    fingers_l2 = CheckResult("fingers", "L2:x", False, 0.95).to_dict()
    unverified = CheckResult("fingers", "L1:rtmw+hand+contour-skin", None, 0.0, note="有手指蜷起或并拢").to_dict()
    nails_far = CheckResult("nails", "L1:deltaE", False, 0.75).to_dict()
    nails_l2 = CheckResult("nails", "L2:x", False, 0.95).to_dict()
    frames = [{"view": "palms", "l1_server": {"fingers": unverified}, "l2": {"fingers": fingers_l2}},
              {"view": "backs", "l1_server": {"fingers": fingers_kp, "nails": nails_far}, "l2": {"fingers": fingers_l2, "nails": nails_l2}},
              {"view": "nails_close", "l1_server": {"nails": nails_far}, "l2": {"nails": nails_l2}}]
    out = server.arbitrate_session("precision", frames, checks=["fingers", "nails"])
    assert out["by_view"]["palms"]["fingers"]["decision"] == "pass"
    assert out["by_view"]["backs"]["fingers"]["decision"] == "pass"
    assert out["by_check"]["nails"]["decision"] == "pass"


def test_nails_close_view_decides(server):
    """拍了指甲近景时以近景为准：手背远景的色差法疑似甲油只记录。"""
    far = CheckResult("nails", "L1:deltaE", True, 0.8, ["right_hand"]).to_dict()
    ok = CheckResult("nails", "L2:x", False, 0.95).to_dict()
    frames = [{"view": "backs", "l1": {"nails": far}}, {"view": "nails_close", "l1": {"nails": ok}, "l2": {"nails": ok}}]
    e = server.arbitrate_session("precision", frames, checks=["nails"])["by_check"]["nails"]
    assert e["views"]["backs"] != "pass" and e["decision"] == "pass" and e["primary_view"] == "nails_close"
    e = server.arbitrate_session("precision", frames[:1], checks=["nails"])["by_check"]["nails"]
    assert e["decision"] != "pass"
