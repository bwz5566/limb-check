"""用本地 mock 服务测试 OpenAI 兼容客户端（含结构化输出不支持时的回退）。"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from limbcheck.vlm import VLMClient

REPLY = {"items": [
    {"check": "fingers", "part": "right_hand", "label": "normal", "fingers": [], "value": 5, "confidence": 0.92},
    {"check": "fingers", "part": "left_hand", "label": "normal", "fingers": [], "value": 5, "confidence": 0.9},
], "image_quality": "good", "reason": "mock"}


class Handler(BaseHTTPRequestHandler):
    calls = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.calls.append(body)
        if "response_format" in body and self.server.reject_schema:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "response_format not supported"}')
            return
        out = json.dumps({"choices": [{"message": {"content": json.dumps(REPLY)}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    srv.reject_schema = False
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


def test_review_roundtrip(server):
    c = VLMClient(f"http://127.0.0.1:{server.server_port}/v1", "mock-vl")
    rv = c.review("palms", ["fingers"], ["data:image/jpeg;base64,AAAA"])
    assert rv.errors == []
    assert rv.results["fingers"].positive is False and rv.results["fingers"].confidence == 0.9
    sent = Handler.calls[-1]
    assert sent["temperature"] == 0 and sent["model"] == "mock-vl"
    assert sent["messages"][1]["content"][1]["type"] == "image_url"


def test_schema_fallback(server):
    server.reject_schema = True
    c = VLMClient(f"http://127.0.0.1:{server.server_port}/v1", "mock-vl", use_schema=True)
    rv = c.review("palms", ["fingers"], ["data:image/jpeg;base64,AAAA"])
    assert any("退回普通模式" in e for e in rv.errors)
    assert rv.results["fingers"].positive is False


def test_connection_error_is_recorded():
    c = VLMClient("http://127.0.0.1:9/v1", "x", timeout=2)
    rv = c.review("palms", ["fingers"], ["data:image/jpeg;base64,AAAA"])
    assert rv.results["fingers"].positive is None and rv.errors
