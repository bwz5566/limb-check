#!/usr/bin/env python3
"""网页验证 demo 的本地服务：静态文件 + 服务端判定 + 大模型复核代理 + 仲裁。

基础功能只用 Python 标准库，直接复用仓库里的 limbcheck.prompts / vlm / arbiter，
保证网页、离线评测、将来的服务端用同一套提示词、输出解析和仲裁规则。
服务端判定（/api/analyze）：对浏览器上传的高清截图跑 Python 流水线（关键点 + 轮廓复核），
需要 pip install -e ".[mediapipe]" 或 ".[rtmw]"；装不上时自动退回浏览器端结果。
浏览器端的关键点只用来引导受检者做动作。

配置：写在同目录的 .env 文件里（复制 .env.example 改名为 .env 再填写；.env 不会提交到 git），
或启动前设置同名环境变量（环境变量优先）：
  VLM_API_KEY   模型服务的 API Key（本地 vLLM 可不设）
  VLM_BASE_URL  OpenAI 兼容接口地址；只设了 VLM_API_KEY 时默认 https://dashscope.aliyuncs.com/compatible-mode/v1
  VLM_MODEL     模型名；默认 qwen3.7-plus（DashScope）
  VLM_SCHEMA    设为 1 时用 JSON Schema 约束输出（vLLM 支持；不支持的服务会自动退回）
  LIMBCHECK_PROMPT_VERSION  提示词版本（configs/prompts/<版本>.json），默认见 configs/checks.json
  LIMBCHECK_PIPELINE  服务端判定用的流水线（configs/pipelines/ 下的名字或 yaml 路径）；
                默认 auto：先试 rtmw-skin-cpu（需 pip install -e ".[rtmw]"，首次运行下载模型），不行再用 mediapipe-skin；
                设为 off 则只用浏览器端结果
  LIMBCHECK_RECORDS_DIR  检测记录（每次检测每一步成功 / 失败、卡在哪个提示、使用者反馈）保存目录，默认 apps/web-demo/records/
  PORT          默认 8000
  HOST          默认 127.0.0.1（局域网内用手机访问时设为 0.0.0.0，但浏览器只允许 https 或 localhost 打开摄像头）
"""
from __future__ import annotations

import json
import os
import sys
import statistics
import threading
import time
from collections import Counter
from datetime import datetime, timedelta
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent


def load_env_file(path: Path) -> None:
    """读取 KEY=VALUE 格式的 .env（# 开头为注释）；已设置的环境变量不覆盖。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v:
            os.environ.setdefault(k, v)


load_env_file(ROOT / ".env")
sys.path.insert(0, str(REPO))
os.environ.setdefault("LIMBCHECK_CONFIG_DIR", str(REPO / "configs"))

from limbcheck.arbiter import arbitrate, worst  # noqa: E402
from limbcheck.core import CHECK_DEFS, VIEW_CN, VIEW_CHECKS, CheckResult  # noqa: E402
from limbcheck.marks import session_review, set_policy  # noqa: E402
from limbcheck.motion import analyze as analyze_motion, motion_result  # noqa: E402
from limbcheck.strictness import arbitration_cfg, default_level, levels, set_config  # noqa: E402
from limbcheck.prompts import l2_checks_for, prompt_version  # noqa: E402
from limbcheck.vlm import VLMClient  # noqa: E402

MODELS = Path(os.environ.get("LIMBCHECK_MODELS_DIR") or REPO / "models")
WEB_CONFIG = json.loads((ROOT / "config.generated.json").read_text(encoding="utf-8"))
JOBS = {j["id"]: j for j in WEB_CONFIG["jobs"]}
set_policy(WEB_CONFIG["policies"]["marks"])      # 录用标准、严格度与网页配置同源（由 configs/ 生成）
set_config(WEB_CONFIG["strictness"])
DECISION_OF_ACTION = {"review": "review", "clean": "redo", "pass": "pass"}


def strictness_of(job: dict | None, requested: str | None) -> str:
    """本场严格度：请求里指定 > 岗位画像 > 默认。"""
    if requested in levels():
        return requested
    return (job or {}).get("strictness") or default_level()
PORT = int(os.environ.get("PORT", "8000"))
HOST = os.environ.get("HOST", "127.0.0.1")
MAX_BODY = 40 * 1024 * 1024


def make_client() -> VLMClient | None:
    key = os.environ.get("VLM_API_KEY", "")
    base = os.environ.get("VLM_BASE_URL") or ("https://dashscope.aliyuncs.com/compatible-mode/v1" if key else "")
    model = os.environ.get("VLM_MODEL") or ("qwen3.7-plus" if base else "")
    if not base or not model:
        return None
    return VLMClient(base, model, key, use_schema=os.environ.get("VLM_SCHEMA") == "1")


CLIENT = make_client()


class ServerJudge:
    """服务端判定：模型只加载一次，按岗位画像、严格度为每个请求组装 Pipeline（模型节点共用）。
    MediaPipe / ONNX 推理对象不保证线程安全，推理串行执行。"""

    AUTO = ("rtmw-skin-cpu", "mediapipe-skin")

    def __init__(self, choice: str | None = None):
        self.choice = (choice or os.environ.get("LIMBCHECK_PIPELINE") or "auto").strip()
        self.base = None
        self.error: str | None = None
        self.loaded = False
        self._lock = threading.Lock()
        self._th = None
        self._profiles: dict = {}

    @property
    def enabled(self) -> bool:
        return self.choice.lower() not in ("off", "none", "0", "")

    def _spec_path(self, name: str) -> Path:
        p = Path(name)
        if p.suffix in (".yaml", ".yml") and p.exists():
            return p
        return REPO / "configs" / "pipelines" / f"{name.removesuffix('.yaml')}.yaml"

    def load(self) -> None:
        with self._lock:
            if self.loaded or not self.enabled:
                return
            self.loaded = True
            errors = []
            try:
                from limbcheck.config import load_job_profiles, load_thresholds
                from limbcheck.pipeline import Pipeline, load_pipeline_spec
            except ImportError as e:
                self.error = f'缺少依赖（{e}）。请在仓库根目录（{REPO}）运行：python -m pip install -e ".[mediapipe,rtmw]"'
                return
            self._th = load_thresholds()
            self._profiles = load_job_profiles()
            for name in (self.AUTO if self.choice.lower() == "auto" else (self.choice,)):
                try:
                    spec = load_pipeline_spec(self._spec_path(name))
                    self.base = Pipeline.from_spec(spec, self._th, None, vlm="none")   # 大模型走 /api/review，这里只做 L1
                    self.error = None
                    return
                except Exception as e:   # 依赖没装、模型下载失败等 → 试下一个
                    errors.append(f"{name}: {type(e).__name__}: {e}"[:300])
            self.error = "；".join(errors)
            if "ModuleNotFoundError" in self.error:
                self.error += f'。请在仓库根目录（{REPO}）运行：python -m pip install -e ".[mediapipe,rtmw]"'

    def status(self) -> dict:
        return {"enabled": self.base is not None, "pipeline": self.base.name if self.base else None,
                "error": self.error if self.enabled else "LIMBCHECK_PIPELINE=off"}

    def analyze(self, view: str, image: str, job_id: str | None, strictness: str | None) -> dict:
        from limbcheck.pipeline import Pipeline

        self.load()
        if self.base is None:
            raise RuntimeError(self.error or "服务端判定未启用")
        img = decode_image(image)
        job = JOBS.get(job_id)
        b = self.base
        p = Pipeline(self._th, keypoints=b.kp, contour=b.contour, segmenter=b.seg, profile=self._profiles.get(job_id),
                     mirrored=b.mirrored, l1=b.l1_names, name=b._name, strictness=strictness_of(job, strictness))
        t0 = time.perf_counter()
        with self._lock:
            r = p.run(img, view)
        l1 = {c: v["l1"] for c, v in r["results"].items() if v.get("l1")}
        return {"view": view, "pipeline": b.name, "l1": l1, "quality": r["quality"], "errors": r["errors"],
                "latency_ms": {**r["latency_ms"], "total": round((time.perf_counter() - t0) * 1000, 1)},
                "img_size": [int(img.shape[1]), int(img.shape[0])]}


def decode_image(data: str):
    """data:image/jpeg;base64,... → RGB ndarray。"""
    import base64

    import cv2
    import numpy as np

    raw = base64.b64decode(data.split(",", 1)[1] if data.startswith("data:") else data)
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("图片无法解码")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


JUDGE = ServerJudge()

# ---------- 检测记录：每次检测一行 JSON（不含图片），用来统计哪些步骤容易失败、卡在哪个提示、哪里需要优化 ----------
RECORDS_DIR = Path(os.environ.get("LIMBCHECK_RECORDS_DIR") or ROOT / "records")
RECORD_TYPES = ("session", "feedback")
FEEDBACK_CN = {"accurate": "结果准确", "false_alarm": "有误判（正常被判有问题）", "missed": "有漏判（有问题没查出）",
               "confusing": "步骤看不懂 / 太难做", "other": "其他"}
_records_lock = threading.Lock()


def append_record(rec: dict, directory: Path | None = None) -> Path:
    if rec.get("type") not in RECORD_TYPES:
        raise ValueError(f"type 只能是 {RECORD_TYPES}")
    sid = rec.get("session_id")
    if not isinstance(sid, str) or not 0 < len(sid) <= 64:
        raise ValueError("缺少 session_id")
    rec = {**rec, "server_at": datetime.now().isoformat(timespec="seconds")}
    d = directory or RECORDS_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{datetime.now():%Y-%m}.jsonl"
    line = json.dumps(rec, ensure_ascii=False)
    if len(line) > 512 * 1024:
        raise ValueError("记录过大（记录里不应包含图片）")
    with _records_lock, path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    return path


def load_records(directory: Path | None = None, days: int | None = None) -> list[dict]:
    d = directory or RECORDS_DIR
    since = (datetime.now() - timedelta(days=days)).isoformat() if days else ""
    rows = []
    for path in sorted(d.glob("*.jsonl")) if d.exists() else []:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("server_at", "") >= since:
                rows.append(r)
    return rows


def summarize_records(rows: list[dict], job: str | None = None) -> dict:
    """按步骤统计完成率、用时、最常卡住的提示；按检测项统计结论分布；汇总使用者反馈。
    同一会话重拍后会再存一条，只取最后一条。"""
    sessions: dict = {}
    feedback = []
    for r in rows:
        if r.get("type") == "session":
            sessions[r["session_id"]] = r
        elif r.get("type") == "feedback":
            feedback.append(r)
    ss = [s for s in sessions.values() if not job or (s.get("job") or {}).get("id") == job]
    keep = {s["session_id"] for s in ss}
    feedback = [f for f in feedback if not job or f["session_id"] in keep]
    steps: dict = {}
    for s in ss:
        for st in s.get("steps") or []:
            e = steps.setdefault(st["id"], {"id": st["id"], "title": st.get("title") or st["id"], "n": 0, "status": Counter(),
                                            "ms": [], "first_ok_ms": [], "hints": Counter(), "hint_sessions": Counter(),
                                            "skipped": 0, "confusing": 0})
            e["n"] += 1
            e["status"][st.get("status") or "?"] += 1
            if st.get("ms") is not None:
                e["ms"].append(st["ms"])
            if st.get("first_ok_ms") is not None:
                e["first_ok_ms"].append(st["first_ok_ms"])
            if "手动跳过" in (st.get("note") or ""):
                e["skipped"] += 1
            for h in st.get("hints") or []:
                e["hints"][h["text"]] += h.get("ms") or 0
                e["hint_sessions"][h["text"]] += 1
    for f in feedback:
        for sid in f.get("steps") or []:
            if sid in steps:
                steps[sid]["confusing"] += 1

    def med(xs):
        return round(statistics.median(xs) / 1000, 1) if xs else None

    step_rows = []
    for e in steps.values():
        n = e["n"]
        step_rows.append({
            "id": e["id"], "title": e["title"], "n": n,
            "pass_rate": round(e["status"]["pass"] / n, 3) if n else None,
            "status": dict(e["status"]), "skipped": e["skipped"], "confusing": e["confusing"],
            "median_s": med(e["ms"]), "median_first_ok_s": med(e["first_ok_ms"]),
            "top_hints": [{"text": t, "total_s": round(ms / 1000, 1), "sessions": e["hint_sessions"][t]}
                          for t, ms in e["hints"].most_common(3)],
        })
    step_rows.sort(key=lambda r: (r["pass_rate"] if r["pass_rate"] is not None else 1, -r["n"]))
    checks: dict = {}
    for s in ss:
        for c, v in (s.get("checks") or {}).items():
            checks.setdefault(c, Counter())[v.get("decision") or "none"] += 1
    recent = sorted(ss, key=lambda s: s.get("server_at", ""), reverse=True)[:30]
    return {
        "sessions": len(ss), "feedback_n": len(feedback),
        "from": min((s.get("server_at", "") for s in ss), default=None), "to": max((s.get("server_at", "") for s in ss), default=None),
        "jobs": dict(Counter((s.get("job") or {}).get("name") or "?" for s in ss)),
        "overall": dict(Counter(s.get("overall") or "none" for s in ss)),
        "median_duration_s": med([s["duration_ms"] for s in ss if s.get("duration_ms")]),
        "steps": step_rows, "checks": {c: dict(v) for c, v in checks.items()},
        "feedback": {"by_rating": {FEEDBACK_CN.get(k, k): v for k, v in Counter(f.get("rating") for f in feedback).items()},
                     "recent": [{"session_id": f["session_id"], "at": f.get("server_at"), "rating": FEEDBACK_CN.get(f.get("rating"), f.get("rating")),
                                 "steps": f.get("steps") or [], "note": f.get("note") or ""} for f in feedback[-30:][::-1]]},
        "recent": [{"session_id": s["session_id"], "at": s.get("server_at"), "job": (s.get("job") or {}).get("name"),
                    "overall": s.get("overall"), "duration_s": round((s.get("duration_ms") or 0) / 1000),
                    "failed_steps": [st.get("title") or st["id"] for st in s.get("steps") or [] if st.get("status") != "pass"]}
                   for s in recent],
    }


def mode_of(job: dict | None, check: str) -> str:
    if job is None:
        return "judge"
    c = job["checks"].get(check)
    if not c or not c.get("enabled", True):
        return "off"
    return c.get("mode", "judge")


def arbitrate_session(job_id: str, frames: list[dict], strictness: str | None = None, motion: dict | None = None,
                      checks: list[str] | None = None) -> dict:
    """frames: [{view, l1: {check: CheckResult dict}, l1_server: {...}, l2: {check: CheckResult dict}}]。
    l1_server 是服务端对高清截图的判定（/api/analyze），有就用它；没有（服务端未启用或失败）才用浏览器端 l1。
    checks：本次检测范围内的检测项（网页上选"只测身体 / 只测手部"时），范围外的不因缺视角判"需复核"。
    motion：动作步骤的关键点序列（格式见 limbcheck/motion.py），用于"肢体协调性"。
    与离线评测一致：按最后一次尝试仲裁（attempt = max_redo），另给出首次尝试的结论供统计重做率。"""
    job = JOBS.get(job_id)
    level = strictness_of(job, strictness)
    cfg = arbitration_cfg(WEB_CONFIG["thresholds"].get("arbitration", {}), level)
    final_attempt = cfg.get("max_redo", 1)
    by_view: dict = {}
    for fr in frames:
        view = fr["view"]
        l1s, l2s = fr.get("l1") or {}, fr.get("l2") or {}
        srv = fr.get("l1_server")
        out = {}
        for c in VIEW_CHECKS.get(view, ()):
            mode = mode_of(job, c)
            use_srv = srv is not None and c in srv
            a = CheckResult.from_dict((srv if use_srv else l1s).get(c))
            b = CheckResult.from_dict(l2s.get(c))
            if mode == "off" or (a is None and b is None):
                continue
            final = arbitrate(c, a, b, cfg, mode, final_attempt)
            first = arbitrate(c, a, b, cfg, mode, 0)
            out[c] = {**final.to_dict(), "shadow_first_attempt": first.shadow,
                      "l1_from": "server" if use_srv else "browser"}
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
        if miss and mode_of(job, c) != "off" and (checks is None or c in checks):
            by_check.setdefault(c, {"mode": mode_of(job, c), "views": {}, "decision": None, "shadow": None, "notes": []})
            add(c, "review", f"未采集到{'、'.join(VIEW_CN[v] for v in miss)}，无法确认手掌、手背均正常")
    # 纹身 / 疤痕的跨视角规则（手指纹身 ≥ 2 处、手背多处、疤痕搭配纹身）和清洁提醒
    sessions = session_review({fr["view"]: fr.get("l2") or {} for fr in frames}, level)
    auto_fail = cfg.get("auto_fail_conf")
    for c, sr in sessions.items():
        if c in by_check:
            by_check[c]["policy"] = sr.to_dict()
            for r in sr.rules:
                d = DECISION_OF_ACTION.get(r["action"]) or (
                    "fail" if auto_fail is not None and r.get("confidence", 0) >= auto_fail else "fail_suggested")
                add(c, d, r["text"])
    # 肢体协调性：分析动作序列（只有关键点一路，发现问题 → 疑似问题·待人工确认）
    analysis = None
    if motion and mode_of(job, "coordination") != "off":
        analysis = analyze_motion(motion, WEB_CONFIG["thresholds"].get("motion"))
        r = motion_result(analysis, WEB_CONFIG["thresholds"].get("motion"))
        if r is not None:
            mode = mode_of(job, "coordination")
            final = arbitrate("coordination", r, None, cfg, mode, final_attempt)
            by_check["coordination"] = {"mode": mode, "views": {}, "decision": final.decision, "shadow": final.shadow,
                                        "notes": [], "reason": final.reason, "l1": r.to_dict(), "motion": analysis}
    overall = worst([e["decision"] for e in by_check.values() if e["mode"] == "judge"])
    return {"by_view": by_view, "by_check": by_check, "overall": overall, "job": job_id, "strictness": level,
            "prompt_version": prompt_version()}


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
        if self.path.startswith("/api/records/summary"):
            from urllib.parse import parse_qs, urlparse

            q = parse_qs(urlparse(self.path).query)
            days = int(q["days"][0]) if q.get("days", [""])[0].isdigit() else None
            return self._json(200, {**summarize_records(load_records(days=days), (q.get("job") or [None])[0] or None),
                                    "dir": str(RECORDS_DIR)})
        if self.path.startswith("/api/config"):
            return self._json(200, {
                "vlm": CLIENT is not None, "model": CLIENT.model if CLIENT else None,
                "schema": bool(CLIENT and CLIENT.use_schema), "prompt_version": prompt_version(),
                "server_judge": JUDGE.status(), "web": WEB_CONFIG,
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
            meta["strictness"] = strictness_of(job, req.get("strictness"))
            rv = CLIENT.review(view, checks, req.get("images") or [], meta)
            return self._json(200, rv.to_dict())
        if self.path == "/api/analyze":
            view = req.get("view")
            if view not in VIEW_CHECKS:
                return self._json(400, {"error": f"未知视角 {view!r}"})
            if not req.get("image"):
                return self._json(400, {"error": "缺少 image"})
            try:
                return self._json(200, JUDGE.analyze(view, req["image"], req.get("job"), req.get("strictness")))
            except ValueError as e:
                return self._json(400, {"error": str(e)})
            except RuntimeError as e:
                return self._json(503, {"error": str(e)})
        if self.path == "/api/records":
            try:
                append_record(req)
            except ValueError as e:
                return self._json(400, {"error": str(e)})
            return self._json(200, {"ok": True})
        if self.path == "/api/arbitrate":
            t0 = time.perf_counter()
            out = arbitrate_session(req.get("job"), req.get("frames") or [], req.get("strictness"), req.get("motion"), req.get("checks"))
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
    print("正在加载服务端判定模型……")
    JUDGE.load()
    js = JUDGE.status()
    server = ThreadingHTTPServer((HOST, PORT), partial(Handler, directory=str(ROOT)))
    status = f"已启用（{CLIENT.model} @ {CLIENT.base_url}）" if CLIENT else "未启用"
    print(f"Demo 已启动：在 Chrome 或 Edge 中打开 http://localhost:{PORT}")
    print(f"服务端判定（L1）：{'已启用（' + js['pipeline'] + '）' if js['enabled'] else '未启用，只用浏览器端结果（' + str(js['error']) + '）'}")
    print(f"大模型复核（L2）：{status}   提示词版本：{prompt_version()}")
    print("按 Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
