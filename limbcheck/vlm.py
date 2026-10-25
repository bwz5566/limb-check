"""OpenAI 兼容接口的视觉大模型客户端（仅标准库）。

适用于 vLLM / vLLM-Ascend 私有部署、阿里云百炼 DashScope 兼容模式、Ollama 等。
环境变量：VLM_BASE_URL（如 http://昇腾服务器:8001/v1）、VLM_MODEL、VLM_API_KEY（私有部署可不填）、
VLM_SCHEMA=1（服务端支持结构化输出时用 JSON Schema 强制格式）。
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .core import CheckResult
from .prompts import build_messages, parse_reply, prompt_version, schema, to_check_results


def file_to_data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode("ascii")


@dataclass
class Review:
    view: str
    checks: list[str]
    model: str
    prompt_version: str = field(default_factory=prompt_version)
    text: str = ""
    parsed: dict | None = None
    errors: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    results: dict[str, CheckResult] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "view": self.view, "checks": self.checks, "model": self.model, "prompt_version": self.prompt_version,
            "text": self.text, "parsed": self.parsed, "errors": self.errors, "latency_ms": round(self.latency_ms, 1),
            "results": {k: v.to_dict() for k, v in self.results.items()},
        }


class VLMError(RuntimeError):
    pass


class VLMClient:
    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 90, use_schema: bool = False,
                 temperature: float = 0.0, max_tokens: int = 1024, max_images: int = 6):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or "EMPTY"
        self.timeout = timeout
        self.use_schema = use_schema
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_images = max_images

    @classmethod
    def from_env(cls) -> "VLMClient | None":
        base = os.environ.get("VLM_BASE_URL", "")
        model = os.environ.get("VLM_MODEL", "")
        if not base or not model:
            return None
        return cls(base, model, os.environ.get("VLM_API_KEY", ""), use_schema=os.environ.get("VLM_SCHEMA") == "1")

    def chat(self, messages: list[dict], schema: dict | None = None) -> str:
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature, "max_tokens": self.max_tokens}
        if schema is not None:
            payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "review", "schema": schema}}
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.api_key},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read()[:300].decode("utf-8", errors="ignore")
            raise VLMError(f"模型服务返回 {e.code}: {detail}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise VLMError(f"无法连接模型服务：{e}") from e
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise VLMError(f"返回格式不符合 OpenAI 接口：{str(data)[:200]}") from e
        if isinstance(text, list):
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        return text or ""

    def review(self, view: str, checks: list[str], images: list[str]) -> Review:
        """对一个视角的证据帧做一次复核。images 为 data URL 或 http(s) 地址。失败不抛异常，记录在 errors。"""
        rv = Review(view=view, checks=list(checks), model=self.model)
        messages = build_messages(view, rv.checks, images[: self.max_images])
        t0 = time.perf_counter()
        try:
            try:
                rv.text = self.chat(messages, schema(rv.checks) if self.use_schema else None)
            except VLMError as e:
                if not self.use_schema or "400" not in str(e):
                    raise
                rv.errors.append("服务端不支持 JSON Schema，已退回普通模式")
                rv.text = self.chat(messages, None)
        except VLMError as e:
            rv.errors.append(str(e))
        rv.latency_ms = (time.perf_counter() - t0) * 1000
        if rv.text:
            rv.parsed, errs = parse_reply(rv.text, rv.checks)
            rv.errors += errs
        rv.results = to_check_results(rv.parsed, rv.checks, view, f"L2:{self.model}", rv.errors)
        return rv
