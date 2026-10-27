#!/usr/bin/env python3
"""网页验证 demo 的本地服务：静态文件 + 大模型复核代理 + 仲裁。

只用 Python 标准库（不需要 pip install），直接复用仓库里的 limbcheck.prompts / vlm / arbiter，
保证网页、离线评测、将来的服务端用同一套提示词、输出解析和仲裁规则。

环境变量：
  VLM_API_KEY   模型服务的 API Key（本地 vLLM 可不设）
  VLM_BASE_URL  OpenAI 兼容接口地址；只设了 VLM_API_KEY 时默认 https://dashscope.aliyuncs.com/compatible-mode/v1
  VLM_MODEL     模型名；默认 qwen-vl-max（DashScope）
  VLM_SCHEMA    设为 1 时用 JSON Schema 约束输出（vLLM 支持；不支持的服务会自动退回）
  LIMBCHECK_PROMPT_VERSION  提示词版本（configs/prompts/<版本>.json），默认见 configs/checks.json
  PORT          默认 8000
  HOST          默认 127.0.0.1（局域网内用手机访问时设为 0.0.0.0，但浏览器只允许 https 或 localhost 打开摄像头）
"""
from __future__ import annotations

import json
import os
import sys
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("LIMBCHECK_CONFIG_DIR", str(REPO / "configs"))

from limbcheck.arbiter import arbitrate, worst  # noqa: E402
from limbcheck.core import CHECK_DEFS, VIEW_CN, VIEW_CHECKS, CheckResult  # noqa: E402
from limbcheck.marks import session_review, set_policy  # noqa: E402
from limbcheck.prompts import l2_checks_for, prompt_version  # noqa: E402
from limbcheck.vlm import VLMClient  # noqa: E402

MODELS = Path(os.environ.get("LIMBCHECK_MODELS_DIR") or REPO / "models")
WEB_CONFIG = json.loads((ROOT / "config.generated.json").read_text(encoding="utf-8"))
JOBS = {j["id"]: j for j in WEB_CONFIG["jobs"]}
set_policy(WEB_CONFIG["policies"]["marks"])      # 录用标准与网页配置同源（configs/policies/marks.yaml 生成）
DECISION_OF_ACTION = {"block": "fail_suggested", "review": "review", "clean": "redo", "pass": "pass"}
PORT = int(os.environ.get("PORT", "8000"))
HOST = os.environ.get("HOST", "127.0.0.1")
MAX_BODY = 40 * 1024 * 1024


def make_client() -> VLMClient | None:
    key = os.environ.get("VLM_API_KEY", "")
    base = os.environ.get("VLM_BASE_URL") or ("https://dashscope.aliyuncs.com/compatible-mode/v1" if key else "")
    model = os.environ.get("VLM_MODEL") or ("qwen-vl-max" if base else "")
    if not base or not model:
        return None
    return VLMClient(base, model, key, use_schema=os.environ.get("VLM_SCHEMA") == "1")


CLIENT = make_client()


def mode_of(job: dict | None, check: str) -> str:
    if job is None:
        return "judge"
    c = job["checks"].get(check)
    if not c or not c.get("enabled", True):
        return "off"
    return c.get("mode", "judge")


def arbitrate_session(job_id: str, frames: list[dict]) -> dict:
    """frames: [{view, l1: {check: CheckResult dict}, l2: {check: CheckResult dict}}]。
    与离线评测一致：按最后一次尝试仲裁（attempt = max_redo），另给出首次尝试的结论供统计重做率。"""
    job = JOBS.get(job_id)
    cfg = WEB_CONFIG["thresholds"].get("arbitration", {})
    final_attempt = cfg.get("max_redo", 1)
    by_view: dict = {}
    for fr in frames:
        view = fr["view"]
        l1s, l2s = fr.get("l1") or {}, fr.get("l2") or {}
        out = {}
        for c in VIEW_CHECKS.get(view, ()):
            mode = mode_of(job, c)
            a, b = CheckResult.from_dict(l1s.get(c)), CheckResult.from_dict(l2s.get(c))
            if mode == "off" or (a is None and b is None):
                continue
            final = arbitrate(c, a, b, cfg, mode, final_attempt)
            first = arbitrate(c, a, b, cfg, mode, 0)
            out[c] = {**final.to_dict(), "shadow_first_attempt": first.shadow}
        by_view[view] = out
    by_check: dict = {}
    for view, res in by_view.items():
        for c, r in res.items():
            e = by_check.setdefault(c, {"mode": r["mode"], "views": {}, "decision": None, "shadow": None, "notes": []})
            e["views"][view] = r["decision"]
            e["decision"] = worst([e["decision"], r["decision"]])
            e["shadow"] = worst([e["shadow"], r["shadow"]])

    def add(c: str, shadow: str, note: str) -> None:
        e = by_check.get(c)
        if e is None:
            return
        e["notes"].append(note)
        e["shadow"] = worst([e["shadow"], shadow])
        if e["mode"] == "judge":
            e["decision"] = worst([e["decision"], shadow])

    # 必拍视角（如手部要同时看掌心和手背）缺了 → 人工复核
    seen = {fr["view"] for fr in frames}
    for c, d in CHECK_DEFS["checks"].items():
        miss = [v for v in d.get("required_views", []) if v not in seen]
        if miss and mode_of(job, c) != "off":
            by_check.setdefault(c, {"mode": mode_of(job, c), "views": {}, "decision": None, "shadow": None, "notes": []})
            add(c, "review", f"未采集到{'、'.join(VIEW_CN[v] for v in miss)}，无法确认手掌、手背均正常")
    # 纹身 / 疤痕的跨视角规则（手指纹身 ≥ 2 处、手背多处、疤痕搭配纹身）和清洁提醒
    sessions = session_review({fr["view"]: fr.get("l2") or {} for fr in frames})
    for c, sr in sessions.items():
        if c in by_check:
            by_check[c]["policy"] = sr.to_dict()
            for r in sr.rules:
                add(c, DECISION_OF_ACTION[r["action"]], r["text"])
    overall = worst([e["decision"] for e in by_check.values() if e["mode"] == "judge"])
    return {"by_view": by_view, "by_check": by_check, "overall": overall, "job": job_id, "prompt_version": prompt_version()}


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".mjs": "text/javascript",
        ".js": "text/javascript",
        ".wasm": "application/wasm",
        ".task": "application/octet-stream",
        ".tflite": "application/octet-stream",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def translate_path(self, path):
        # /models/ 指向仓库根目录的 models/，网页和离线评测共用同一份模型文件
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean.startswith("/models/"):
            name = Path(clean[len("/models/"):]).name
            return str(MODELS / name)
        return super().translate_path(path)

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length > MAX_BODY:
            raise ValueError("请求过大")
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        if self.path.startswith("/api/config"):
            return self._json(200, {
                "vlm": CLIENT is not None, "model": CLIENT.model if CLIENT else None,
                "schema": bool(CLIENT and CLIENT.use_schema), "prompt_version": prompt_version(),
                "web": WEB_CONFIG,
            })
        return super().do_GET()

    def do_POST(self):
        try:
            req = self._body()
        except (ValueError, json.JSONDecodeError) as e:
            return self._json(400, {"error": f"请求格式错误：{e}"})
        if self.path == "/api/review":
            if CLIENT is None:
                return self._json(503, {"error": "未配置大模型（VLM_API_KEY 或 VLM_BASE_URL + VLM_MODEL），跳过 L2 复核"})
            view = req.get("view")
            if view not in VIEW_CHECKS:
                return self._json(400, {"error": f"未知视角 {view!r}"})
            checks = l2_checks_for(view, req.get("enabled"))
            if not checks:
                return self._json(200, {"view": view, "checks": [], "results": {}, "errors": ["该视角没有需要 L2 复核的项目"]})
            job = JOBS.get(req.get("job"))
            meta = {k: v for k, v in (req.get("meta") or {}).items() if k in ("mm_per_px", "img_w", "img_h", "scale_source")}
            meta["params"] = {c: (job["checks"].get(c) or {}).get("params") or {} for c in ("tattoo", "scar")} if job else {}
            rv = CLIENT.review(view, checks, req.get("images") or [], meta)
            return self._json(200, rv.to_dict())
        if self.path == "/api/arbitrate":
            t0 = time.perf_counter()
            out = arbitrate_session(req.get("job"), req.get("frames") or [])
            out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return self._json(200, out)
        return self._json(404, {"error": "not found"})

    def log_message(self, fmt, *args):
        if args and "/api/" in str(args[0]):
            super().log_message(fmt, *args)


def main():
    missing = [n for n in ("hand_landmarker.task", "pose_landmarker_full.task") if not (MODELS / n).exists()]
    if missing:
        print(f"警告：{MODELS} 下缺少模型文件 {missing}，网页将无法加载模型")
    server = ThreadingHTTPServer((HOST, PORT), partial(Handler, directory=str(ROOT)))
    status = f"已启用（{CLIENT.model} @ {CLIENT.base_url}）" if CLIENT else "未启用"
    print(f"Demo 已启动：在 Chrome 或 Edge 中打开 http://localhost:{PORT}")
    print(f"大模型复核（L2）：{status}   提示词版本：{prompt_version()}")
    print("按 Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
