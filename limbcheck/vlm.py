"""OpenAI 兼容接口的视觉大模型客户端（仅标准库）。

适用于 vLLM / vLLM-Ascend 私有部署、阿里云百炼 DashScope 兼容模式、Ollama 等。
环境变量：VLM_BASE_URL（如 http://昇腾服务器:8001/v1）、VLM_MODEL、VLM_API_KEY（私有部署可不填）、
VLM_SCHEMA=1（服务端支持结构化输出时用 JSON Schema 强制格式）、
VLM_THINKING（深度思考：1 = 全部开启，或逗号分隔的检测项如 fingers,nails = 只对这些项开启；
用 vLLM 的 chat_template_kwargs.enable_thinking，Qwen3 系列适用）、
VLM_TIMEOUT（秒）、VLM_MAX_TOKENS（开启思考时默认 600 秒、8192，否则 90 秒、2048）。
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


def parse_thinking(v) -> bool | frozenset[str]:
    """VLM_THINKING / 流水线配置的 thinking：True / "1" / "all" = 全部开启；检测项列表或 "fingers,nails" = 只对这些项开启。"""
    if isinstance(v, bool) or v is None:
        return bool(v)
    if isinstance(v, str):
        v = v.strip()
        if v.lower() in ("", "0", "false", "off", "no"):
            return False
        if v.lower() in ("1", "true", "on", "yes", "all"):
            return True
        v = v.split(",")
    return frozenset(x.strip() for x in v if str(x).strip())


def env_options() -> dict:
    """从环境变量读客户端选项（VLM_SCHEMA / VLM_THINKING / VLM_TIMEOUT / VLM_MAX_TOKENS）。"""
    opts: dict = {"use_schema": os.environ.get("VLM_SCHEMA") == "1", "thinking": parse_thinking(os.environ.get("VLM_THINKING"))}
    if os.environ.get("VLM_TIMEOUT"):
        opts["timeout"] = float(os.environ["VLM_TIMEOUT"])
    if os.environ.get("VLM_MAX_TOKENS"):
        opts["max_tokens"] = int(os.environ["VLM_MAX_TOKENS"])
    return opts


class VLMClient:
    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float | None = None, use_schema: bool = False,
                 temperature: float = 0.0, max_tokens: int | None = None, max_images: int = 6, thinking=False):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or "EMPTY"
        self.thinking = parse_thinking(thinking)
        # 思考会先输出几千 token 的推理，2048 会被截断、90 秒也不够
        self.timeout = timeout or (600 if self.thinking else 90)
        self.use_schema = use_schema
        self.temperature = temperature
        self.max_tokens = max_tokens or (8192 if self.thinking else 2048)
        self.max_images = max_images

    @classmethod
    def from_env(cls) -> "VLMClient | None":
        base = os.environ.get("VLM_BASE_URL", "")
        model = os.environ.get("VLM_MODEL", "")
        if not base or not model:
            return None
        return cls(base, model, os.environ.get("VLM_API_KEY", ""), **env_options())

    def thinks_for(self, checks: list[str]) -> bool:
        return self.thinking is True or bool(self.thinking and self.thinking & set(checks))

    def chat(self, messages: list[dict], schema: dict | None = None, thinking: bool = False) -> str:
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature, "max_tokens": self.max_tokens}
        if thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": True}
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
            choice = data["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise VLMError(f"返回格式不符合 OpenAI 接口：{str(data)[:200]}") from e
        if choice.get("finish_reason") == "length":
            raise VLMError(f"模型输出被截断（已用满 max_tokens={self.max_tokens}，开启思考时更容易发生），请调大 VLM_MAX_TOKENS")
        if isinstance(text, list):
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        return text or ""

    def review(self, view: str, checks: list[str], images: list[str], meta: dict | None = None) -> Review:
        """对一个视角的证据帧做一次复核。images 为 data URL 或 http(s) 地址。失败不抛异常，记录在 errors。
        meta：图像比例与岗位参数（见 prompts.to_check_results），纹身 / 疤痕按录用标准判定时用。"""
        rv = Review(view=view, checks=list(checks), model=self.model)
        messages = build_messages(view, rv.checks, images[: self.max_images])
        think = self.thinks_for(rv.checks)
        t0 = time.perf_counter()
        try:
            try:
                rv.text = self.chat(messages, schema(rv.checks) if self.use_schema else None, think)
            except VLMError as e:
                if not self.use_schema or "400" not in str(e):
                    raise
                # 400 不一定是 Schema 不支持（如图片数超过服务端上限）：不带 Schema 重试成功才算 Schema 的问题
                rv.text = self.chat(messages, None, think)
                rv.errors.append("服务端不支持 JSON Schema，已退回普通模式")
        except VLMError as e:
            rv.errors.append(str(e))
        rv.latency_ms = (time.perf_counter() - t0) * 1000
        if rv.text:
            rv.parsed, errs = parse_reply(rv.text, rv.checks)
            rv.errors += errs
        elif not rv.errors:
            rv.errors.append("模型返回了空内容")
        rv.results = to_check_results(rv.parsed, rv.checks, view, f"L2:{self.model}", rv.errors, meta)
        return rv
