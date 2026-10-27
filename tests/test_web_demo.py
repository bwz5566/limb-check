"""网页 demo 的 server.py：只用标准库、复用 limbcheck 的仲裁；接口在未配置大模型时也能工作。"""
import importlib.util
import json
import threading
import urllib.error
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

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
    hair = CheckResult("hair", "L2:x", True, 0.9, ["head"]).to_dict()
    h = server.arbitrate_session("precision", [{"view": "stand", "l2": {"hair": hair}}])["by_view"]["stand"]["hair"]
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
