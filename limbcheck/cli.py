"""命令行入口：limbcheck <命令> ...   （python -m limbcheck.cli 同效）

常用流程见 README.md 与 docs/VALIDATION.md。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path


def _out(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def _git() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5,
                              cwd=Path(__file__).parent).stdout.strip() or "—"
    except Exception:
        return "—"


def _profile(job: str | None):
    from .config import load_job_profiles

    if not job or job == "all":
        return None
    profiles = load_job_profiles()
    if job not in profiles:
        sys.exit(f"没有岗位画像 {job}，可选：{', '.join(profiles)} 或 all")
    return profiles[job]


def _pipeline(args, use_vlm: bool = True):
    from .config import load_thresholds
    from .pipeline import Pipeline, load_pipeline_spec

    spec = load_pipeline_spec(args.pipeline)
    over = {}
    for k in ("keypoints", "contour", "segmenter"):
        v = getattr(args, k, None)
        if v:
            over[k] = v
    if getattr(args, "vlm_base_url", None) and getattr(args, "vlm_model", None):
        over["vlm"] = {"base_url": args.vlm_base_url, "model": args.vlm_model, "api_key": args.vlm_api_key or "",
                       "use_schema": bool(args.vlm_schema)}
    if not use_vlm or getattr(args, "no_vlm", False):
        over["vlm"] = "none"
    if getattr(args, "strictness", None):
        over["strictness"] = args.strictness
    th = load_thresholds(getattr(args, "thresholds", None))
    return Pipeline.from_spec(spec, th, _profile(getattr(args, "job", None)), **over), th


def _write_report(data_dir: Path, run_dir: Path, preds: dict, run_info: dict, acceptance: str | None):
    from .config import load_acceptance
    from .dataset import load_samples, load_sessions
    from .metrics import compute, evaluate_acceptance
    from .report import render

    samples = load_samples(data_dir)
    m = compute(samples, preds, load_sessions(data_dir))
    rows = evaluate_acceptance(m, load_acceptance(acceptance)) if acceptance != "none" else None
    (run_dir / "metrics.json").write_text(json.dumps({"metrics": m, "acceptance": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "report.md").write_text(render(m, rows, run_info, str(data_dir)), encoding="utf-8")
    return m, rows


def cmd_eval(args):
    from tqdm import tqdm

    from .dataset import load_image, load_samples, write_jsonl
    from .prompts import prompt_version

    data_dir = Path(args.data)
    samples = load_samples(data_dir)
    if args.views:
        samples = [s for s in samples if s.view in args.views.split(",")]
    if args.limit:
        samples = samples[: args.limit]
    pipe, th = _pipeline(args)
    run_dir = Path(args.out or f"runs/{data_dir.name}-{pipe.name.replace('/', '_')}-{dt.datetime.now():%m%d-%H%M}")
    run_dir.mkdir(parents=True, exist_ok=True)
    preds = {}
    for s in tqdm(samples, desc="评测", unit="张"):
        try:
            rec = pipe.run(load_image(data_dir / s.image), s.view, s.meta)
        except Exception as e:
            rec = {"view": s.view, "pipeline": pipe.name, "results": {}, "latency_ms": {}, "errors": [f"{type(e).__name__}: {e}"]}
        preds[s.id] = {"id": s.id, "session_id": s.session_id, **rec}
    write_jsonl(run_dir / "predictions.jsonl", preds.values())
    info = {"name": run_dir.name, "pipeline": pipe.describe(), "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "git": _git(), "prompt_version": prompt_version(), "thresholds": th, "data": str(data_dir), "samples": len(samples)}
    (run_dir / "run.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    m, rows = _write_report(data_dir, run_dir, preds, info, args.acceptance)
    errs = sum(bool(p.get("errors")) for p in preds.values())
    print(f"\n完成：{len(preds)} 张，{errs} 张有错误记录。报告：{run_dir / 'report.md'}")
    if rows:
        from .report import STATUS_CN
        for r in rows:
            print(f"  [{STATUS_CN[r['status']]}] {r.get('name') or r['id']}: {r.get('value')}")


def cmd_score(args):
    from .dataset import read_jsonl

    data_dir, run_dir = Path(args.data), Path(args.run)
    pred_path = Path(args.predictions) if args.predictions else run_dir / "predictions.jsonl"
    if not args.predictions and not pred_path.exists() and (data_dir / "predictions" / "demo_browser.jsonl").exists():
        pred_path = data_dir / "predictions" / "demo_browser.jsonl"   # 网页 demo 导入的数据：直接评浏览器端结果
    preds = {p["id"]: p for p in read_jsonl(pred_path)}
    if not preds:
        sys.exit(f"{pred_path} 为空")
    run_dir.mkdir(parents=True, exist_ok=True)
    info_path = run_dir / "run.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {"name": run_dir.name, "pipeline": pred_path.stem}
    _write_report(data_dir, run_dir, preds, info, args.acceptance)
    print(f"报告：{run_dir / 'report.md'}")


def cmd_calibrate(args):
    from .calibrate import calibrate, render
    from .config import load_thresholds
    from .dataset import load_samples, read_jsonl

    data_dir, run_dir = Path(args.data), Path(args.run)
    pred_path = Path(args.predictions) if args.predictions else run_dir / "predictions.jsonl"
    preds = {p["id"]: p for p in read_jsonl(pred_path)}
    if not preds:
        sys.exit(f"{pred_path} 为空")
    th = load_thresholds(args.thresholds)
    rep = calibrate(load_samples(data_dir), preds, th.get("arbitration", {}),
                    args.checks.split(",") if args.checks else None,
                    args.levels.split(",") if args.levels else None, _profile(args.job))
    md = render(rep)
    out = Path(args.out) if args.out else run_dir / "calibration.md"
    out.write_text(md, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(md)
    print(f"\n已写入 {out}")


def cmd_motion(args):
    """重新分析网页 demo 导出报告里的动作序列（改了 motion 阈值后不用重新采集）。"""
    from .config import load_thresholds
    from .motion import LABEL_CN, analyze, motion_result

    th = load_thresholds(args.thresholds).get("motion")
    files = [f for p in args.reports for f in (sorted(Path(p).glob("*.json")) if Path(p).is_dir() else [Path(p)])]
    out = []
    for f in files:
        rep = json.loads(f.read_text(encoding="utf-8"))
        a = analyze(rep.get("motion"), th)
        r = motion_result(a, th)
        out.append({"file": str(f), "session_id": rep.get("session_id"), "analysis": a, "result": r.to_dict() if r else None})
        verdict = "未采集到动作序列" if r is None else "无法判断" if r.positive is None else "发现问题" if r.positive else "正常"
        print(f"\n## {f.name}（{rep.get('session_id', '')}）：{verdict}")
        for x in a:
            notes = "；".join(f"{LABEL_CN[y['label']]}：{y['note']}" for y in x["findings"]) or ("正常" if x["verdict"] == "normal" else "帧数不足")
            print(f"- {x['title']}（{x['frames']} 帧）：{notes}  {json.dumps(x['metrics'], ensure_ascii=False)}")
    if args.out:
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {args.out}")


def cmd_compare(args):
    from .report import compare

    runs = []
    for r in args.runs:
        p = Path(r) / "metrics.json"
        runs.append((Path(r).name, json.loads(p.read_text(encoding="utf-8"))["metrics"]))
    md = compare(runs)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"已写入 {args.out}")
    else:
        print(md)


def cmd_import_demo(args):
    from .dataset import import_demo_reports

    files = []
    for p in args.reports:
        p = Path(p)
        files += sorted(p.glob("*.json")) if p.is_dir() else [p]
    s, sess = import_demo_reports(files, args.out)
    print(f"导入 {len(sess)} 个会话、{len(s)} 张图到 {args.out}")


def cmd_import_folder(args):
    from .dataset import import_folder

    s = import_folder(args.images, args.labels, args.out, source=args.source)
    print(f"导入 {len(s)} 张图到 {args.out}")


def cmd_labels_template(args):
    from .dataset import write_labels_template

    n = write_labels_template(args.images, args.out)
    print(f"已为 {n} 张图生成标注表 {args.out}（用 Excel 打开填写，保存为 UTF-8 CSV）")


def cmd_synth(args):
    from .registry import create
    from .synth import synthesize

    kp = create("keypoints", args.keypoints)
    fingers = args.fingers.split(",") if args.fingers else None
    cuts = args.cuts.split(",") if args.cuts else None
    kw = {k: v for k, v in {"fingers": fingers, "cuts": cuts}.items() if v}
    out = synthesize(args.data, args.out, kp, max_per_image=args.max_per_image, seed=args.seed, **kw)
    pos = sum(1 for s in out if s.meta.get("synth") != "original")
    print(f"生成 {len(out)} 张（缺指 {pos} 张，原图 {len(out) - pos} 张）到 {args.out}")


def cmd_bench(args):
    from .bench import bench_pipeline
    from .dataset import load_image, load_samples

    pipe, _ = _pipeline(args, use_vlm=False)
    if args.data:
        samples = load_samples(args.data)[: args.limit or None]
        items = [(load_image(Path(args.data) / s.image), s.view) for s in samples]
    else:
        files = sorted(p for p in Path(args.images).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[: args.limit or None]
        items = [(load_image(p), args.view) for p in files]
    if not items:
        sys.exit("没有图片")
    res = {"pipeline": pipe.describe(), **bench_pipeline(pipe, items, repeat=args.repeat)}
    _out(res)
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")


def _vlm_client(args):
    from .vlm import VLMClient

    if args.vlm_base_url and args.vlm_model:
        return VLMClient(args.vlm_base_url, args.vlm_model, args.vlm_api_key or "", use_schema=bool(args.vlm_schema))
    c = VLMClient.from_env()
    if c is None:
        sys.exit("请用 --vlm-base-url / --vlm-model 或环境变量 VLM_BASE_URL / VLM_MODEL 指定大模型服务")
    return c


def cmd_bench_vlm(args):
    from .bench import bench_vlm
    from .prompts import l2_checks_for
    from .vlm import file_to_data_url

    client = _vlm_client(args)
    imgs = [file_to_data_url(p) for p in args.image]
    checks = args.checks.split(",") if args.checks else l2_checks_for(args.view)
    rows = bench_vlm(client, args.view, checks, imgs, [int(x) for x in args.concurrency.split(",")], args.requests)
    print(f"模型 {client.model} @ {client.base_url}，视角 {args.view}，检测项 {checks}")
    print("| 并发 | 请求数 | 吞吐（次/秒） | p50 ms | p95 ms | 成功率 | JSON 合格率 |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        print(f"| {r['concurrency']} | {r['requests']} | {r['throughput_per_s']} | {r['p50_ms']} | {r['p95_ms']} | {r['ok_rate']} | {r['json_ok_rate']} |")
    if args.out:
        Path(args.out).write_text(json.dumps({"model": client.model, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_vlm_probe(args):
    from .prompts import build_user_prompt, l2_checks_for
    from .vlm import file_to_data_url

    client = _vlm_client(args)
    checks = args.checks.split(",") if args.checks else l2_checks_for(args.view)
    if args.show_prompt:
        print(build_user_prompt(args.view, checks), "\n")
    rv = client.review(args.view, checks, [file_to_data_url(p) for p in args.image])
    print(f"耗时 {rv.latency_ms:.0f} ms\n--- 原始输出 ---\n{rv.text}\n--- 问题 ---\n{rv.errors or '无'}\n--- 判定 ---")
    for c, r in rv.results.items():
        print(f"{c}: positive={r.positive} confidence={r.confidence} parts={r.parts} note={r.note}")


def cmd_viz(args):
    from .dataset import load_image, load_samples, read_jsonl, save_image
    from .viz import draw

    pipe, _ = _pipeline(args, use_vlm=False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.image:
        img = load_image(args.image)
        rec = pipe.run(img, args.view)
        save_image(out / (Path(args.image).stem + "_viz.jpg"), draw(img, pipe, args.view, rec))
        _out(rec["results"])
        return
    samples = load_samples(args.data)
    ids = None
    if args.run:
        m = json.loads((Path(args.run) / "metrics.json").read_text(encoding="utf-8"))["metrics"]
        key = {"misses": "misses", "false_alarms": "false_alarms"}.get(args.only)
        if key:
            ids = {i for c in m["checks"].values() for lv in c.values() for i in lv.get(key, [])}
        pred_path = Path(args.run) / "predictions.jsonl"
        if not pred_path.exists():   # score 网页 demo 数据得到的运行目录：预测在数据集里
            pred_path = Path(args.data) / "predictions" / "demo_browser.jsonl"
        preds = {p["id"]: p for p in read_jsonl(pred_path)}
    else:
        preds = {}
    n = 0
    for s in samples:
        if ids is not None and s.id not in ids and s.session_id not in ids:
            continue
        img = load_image(Path(args.data) / s.image)
        save_image(out / f"{s.id}.jpg", draw(img, pipe, s.view, preds.get(s.id)))
        n += 1
        if args.limit and n >= args.limit:
            break
    print(f"已输出 {n} 张到 {out}")


def cmd_export_web_config(args):
    from .config import export_web_config

    print(f"已生成 {export_web_config(args.out)}")


def cmd_list(args):
    from . import registry
    from .config import config_dir, load_job_profiles
    from .core import CHECK_CN, VIEW_CHECKS

    print("可替换节点（registry）：")
    for k in registry.KINDS:
        print(f"  {k}: {', '.join(registry.names(k))}")
    print("检测项：", "、".join(f"{v}({k})" for k, v in CHECK_CN.items()))
    print("视角：", "；".join(f"{v}: {','.join(c)}" for v, c in VIEW_CHECKS.items()))
    print("岗位画像：", "、".join(f"{p['name']}({k})" for k, p in load_job_profiles().items()))
    d = config_dir()
    print("流水线配置：", "、".join(p.stem for p in sorted((d / "pipelines").glob("*.yaml"))))
    print("提示词版本：", "、".join(p.stem for p in sorted((d / "prompts").glob("*.json"))))


def _add_pipeline_args(p, vlm: bool = True):
    p.add_argument("--pipeline", default="configs/pipelines/mediapipe-skin.yaml", help="流水线配置文件")
    p.add_argument("--keypoints", help="覆盖关键点模型（如 mediapipe / rtmw+hand / 包.模块:类）")
    p.add_argument("--contour", help="覆盖轮廓复核后端（skin / mpseg / sam2 / none）")
    p.add_argument("--segmenter", help="覆盖分割模型（mpseg / none）")
    p.add_argument("--thresholds", help="阈值文件，默认 configs/thresholds.yaml")
    p.add_argument("--job", default="precision", help="岗位画像（决定启用哪些检测项、record_only），all 表示全部按判定模式")
    p.add_argument("--strictness", help="筛选严格度 strict / standard / lenient（默认取岗位画像，见 configs/strictness.yaml）")
    if vlm:
        _add_vlm_args(p)
        p.add_argument("--no-vlm", action="store_true", help="不调用 L2")


def _add_vlm_args(p):
    p.add_argument("--vlm-base-url", help="OpenAI 兼容接口地址，如 http://10.0.0.8:8001/v1（默认读 VLM_BASE_URL）")
    p.add_argument("--vlm-model", help="模型名（默认读 VLM_MODEL）")
    p.add_argument("--vlm-api-key", help="API Key（默认读 VLM_API_KEY）")
    p.add_argument("--vlm-schema", action="store_true", help="用 JSON Schema 强制输出格式（需服务端支持）")


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(prog="limbcheck", description="远程肢体完整性检测 · M0 验证工具")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("eval", help="在数据集上跑流水线并生成报告")
    p.add_argument("--data", required=True, help="数据集目录（含 manifest.jsonl）")
    p.add_argument("--out", help="输出目录，默认 runs/<数据集>-<流水线>-<时间>")
    p.add_argument("--limit", type=int, help="只跑前 N 张（调试用）")
    p.add_argument("--views", help="只跑这些视角，逗号分隔")
    p.add_argument("--acceptance", default=None, help="验收标准文件，默认 configs/acceptance.yaml；none 表示不判定")
    _add_pipeline_args(p)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("score", help="对已有预测重新计算指标和报告（改了验收标准或想评估浏览器端结果时用）")
    p.add_argument("--data", required=True)
    p.add_argument("--run", required=True, help="运行目录（输出 report.md / metrics.json）")
    p.add_argument("--predictions", help="预测文件，默认 <run>/predictions.jsonl；浏览器端结果用 <data>/predictions/demo_browser.jsonl")
    p.add_argument("--acceptance", default=None)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("calibrate", help="按严格度校准 high_conf / auto_fail_conf（模型更换或微调后用，不重新调用模型）")
    p.add_argument("--data", required=True, help="带标注的验证集")
    p.add_argument("--run", required=True, help="在该数据集上 eval 的运行目录")
    p.add_argument("--predictions", help="预测文件，默认 <run>/predictions.jsonl")
    p.add_argument("--checks", help="只校准这些检测项，逗号分隔")
    p.add_argument("--levels", help="只校准这些严格度，逗号分隔")
    p.add_argument("--job", default="precision", help="岗位画像（取检测项参数，如疤痕的 fine_work）")
    p.add_argument("--thresholds", help="阈值文件，默认 configs/thresholds.yaml")
    p.add_argument("-o", "--out", help="输出 Markdown，默认 <run>/calibration.md（同名 .json 为完整网格）")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("motion", help="重新分析网页 demo 报告里的动作序列（肢体协调性），改了 motion 阈值后用")
    p.add_argument("reports", nargs="+", help="导出的报告 JSON（文件或目录）")
    p.add_argument("--thresholds", help="阈值文件，默认 configs/thresholds.yaml")
    p.add_argument("-o", "--out", help="把完整分析写入 JSON")
    p.set_defaults(func=cmd_motion)

    p = sub.add_parser("compare", help="多次运行横向对比")
    p.add_argument("runs", nargs="+")
    p.add_argument("-o", "--out")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("import-demo", help="导入网页 demo 验证模式导出的报告 JSON（文件或目录）")
    p.add_argument("reports", nargs="+")
    p.add_argument("-o", "--out", required=True)
    p.set_defaults(func=cmd_import_demo)

    p = sub.add_parser("import-folder", help="导入图片文件夹 + 标注表")
    p.add_argument("images")
    p.add_argument("--labels", required=True)
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--source", default="folder", help="写入 meta.source，用于分组统计（如 real / partner / internal）")
    p.set_defaults(func=cmd_import_folder)

    p = sub.add_parser("labels-template", help="为图片文件夹生成标注表模板（CSV）")
    p.add_argument("images")
    p.add_argument("-o", "--out", default="labels.csv")
    p.set_defaults(func=cmd_labels_template)

    p = sub.add_parser("synth-missing", help="在健全手图片上合成缺指样本")
    p.add_argument("--data", required=True)
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--keypoints", default="mediapipe", help="用于定位手指的关键点模型")
    p.add_argument("--fingers", help="只合成这些手指，逗号分隔（默认五指）")
    p.add_argument("--cuts", help="mcp（整指）/ pip（远端两节），逗号分隔，默认两种")
    p.add_argument("--max-per-image", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("bench", help="L1 模型耗时与吞吐压测")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--data", help="数据集目录")
    g.add_argument("--images", help="图片目录（需配合 --view）")
    p.add_argument("--view", default="palms")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("-o", "--out")
    _add_pipeline_args(p, vlm=False)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("bench-vlm", help="L2 大模型并发压测")
    p.add_argument("--image", nargs="+", required=True)
    p.add_argument("--view", default="palms")
    p.add_argument("--checks", help="逗号分隔，默认该视角全部可评估项")
    p.add_argument("--concurrency", default="1,2,4,8")
    p.add_argument("--requests", type=int, default=16)
    p.add_argument("-o", "--out")
    _add_vlm_args(p)
    p.set_defaults(func=cmd_bench_vlm)

    p = sub.add_parser("vlm-probe", help="对一张图调用一次大模型，查看原始输出和解析结果")
    p.add_argument("--image", nargs="+", required=True)
    p.add_argument("--view", default="palms")
    p.add_argument("--checks")
    p.add_argument("--show-prompt", action="store_true")
    _add_vlm_args(p)
    p.set_defaults(func=cmd_vlm_probe)

    p = sub.add_parser("viz", help="把关键点、手部掩码和结论画到图上，排查漏检 / 误报")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image")
    g.add_argument("--data")
    p.add_argument("--view", default="palms", help="配合 --image 使用")
    p.add_argument("--run", help="运行目录：叠加该次预测结果，并可用 --only 只画漏检或误报")
    p.add_argument("--only", choices=["all", "misses", "false_alarms"], default="all")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("-o", "--out", default="viz")
    _add_pipeline_args(p, vlm=False)
    p.set_defaults(func=cmd_viz)

    p = sub.add_parser("export-web-config", help="由 configs/ 生成网页 demo 的 config.generated.json")
    p.add_argument("-o", "--out")
    p.set_defaults(func=cmd_export_web_config)

    p = sub.add_parser("list", help="列出可替换节点、检测项、岗位画像、流水线和提示词版本")
    p.set_defaults(func=cmd_list)

    args = ap.parse_args(argv)
    from .registry import close_all

    try:
        args.func(args)
    finally:
        close_all()


if __name__ == "__main__":
    main()
