"""用本地 mock 服务测试 OpenAI 兼容客户端（含结构化输出不支持时的回退）。"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from limbcheck.vlm import VLMClient, env_options, parse_thinking

REPLY = {"items": [
    {"check": "fingers", "part": "right_hand", "label": "normal", "fingers": [], "value": 5, "confidence": 0.92},
    {"check": "fingers", "part": "left_hand", "label": "normal", "fingers": [], "value": 5, "confidence": 0.9},
], "image_quality": "good", "reason": "mock"}


class Handler(BaseHTTPRequestHandler):
    calls = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.calls.append(body)
        if len(body["messages"][1]["content"]) - 1 > self.server.max_images:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "At most 1 image(s) may be provided in one prompt."}')
            return
        if "response_format" in body and self.server.reject_schema:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "response_format not supported"}')
            return
        if body["max_tokens"] < self.server.need_tokens:
            out = json.dumps({"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}).encode()
        else:
            out = json.dumps({"choices": [{"message": {"content": json.dumps(REPLY)}, "finish_reason": "stop"}]}).encode()
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
    srv.max_images = 6
    srv.need_tokens = 0
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


def test_other_400_not_blamed_on_schema(server):
    server.max_images = 1
    c = VLMClient(f"http://127.0.0.1:{server.server_port}/v1", "mock-vl", use_schema=True)
    rv = c.review("palms", ["fingers"], ["data:image/jpeg;base64,AAAA"] * 2)
    assert not any("JSON Schema" in e for e in rv.errors)
    assert any("At most 1 image" in e for e in rv.errors)
    assert rv.results["fingers"].positive is None


def test_connection_error_is_recorded():
    c = VLMClient("http://127.0.0.1:9/v1", "x", timeout=2)
    rv = c.review("palms", ["fingers"], ["data:image/jpeg;base64,AAAA"])
    assert rv.results["fingers"].positive is None and rv.errors


def test_thinking_switch(server):
    url = f"http://127.0.0.1:{server.server_port}/v1"
    c = VLMClient(url, "mock-vl", thinking="fingers,nails")
    assert c.timeout == 600 and c.max_tokens == 8192
    c.review("palms", ["fingers", "tattoo"], ["data:image/jpeg;base64,AAAA"])
    assert Handler.calls[-1]["chat_template_kwargs"] == {"enable_thinking": True}
    c.review("neck", ["tattoo", "scar"], ["data:image/jpeg;base64,AAAA"])
    assert "chat_template_kwargs" not in Handler.calls[-1]
    off = VLMClient(url, "mock-vl")
    assert off.timeout == 90 and off.max_tokens == 2048
    off.review("palms", ["fingers"], ["data:image/jpeg;base64,AAAA"])
    assert "chat_template_kwargs" not in Handler.calls[-1]


def test_parse_thinking_and_env(monkeypatch):
    assert parse_thinking("1") is True and parse_thinking("all") is True
    assert parse_thinking("") is False and parse_thinking(None) is False and parse_thinking("0") is False
    assert parse_thinking("nails, fingers") == {"nails", "fingers"} and parse_thinking(["hair"]) == {"hair"}
    monkeypatch.setenv("VLM_THINKING", "1")
    monkeypatch.setenv("VLM_TIMEOUT", "300")
    monkeypatch.setenv("VLM_MAX_TOKENS", "4096")
    assert env_options() == {"use_schema": False, "thinking": True, "timeout": 300.0, "max_tokens": 4096}


def test_truncated_output_is_reported(server):
    server.need_tokens = 4096
    c = VLMClient(f"http://127.0.0.1:{server.server_port}/v1", "mock-vl", use_schema=True)
    rv = c.review("palms", ["fingers"], ["data:image/jpeg;base64,AAAA"])
    assert any("截断" in e for e in rv.errors) and not any("JSON Schema" in e for e in rv.errors)
    assert rv.results["fingers"].positive is None
    rv = VLMClient(f"http://127.0.0.1:{server.server_port}/v1", "mock-vl", max_tokens=8192).review("palms", ["fingers"], ["data:image/jpeg;base64,AAAA"])
    assert rv.errors == [] and rv.results["fingers"].positive is False
