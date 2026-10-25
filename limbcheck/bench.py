"""压测：L1 模型单张耗时 / 吞吐，L2 大模型在不同并发下的延迟与吞吐。

容量估算（docs/METRICS.md D 节）：
    单卡可支撑的实时路数 ≈ 单卡吞吐（张/秒） ÷ 每路每秒需要分析的帧数（建议 5–10 帧/秒）
    L2 每次检测约调用 4–6 次（掌心、手背、正面、背面、双臂等），单次检测的 L2 时长 ≈ 调用次数 × p50 / 并发
"""
from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from .metrics import percentile


def bench_pipeline(pipeline, items: list[tuple], repeat: int = 3, warmup: int = 2) -> dict:
    """items: [(img, view), ...]。不含 L2（请先把 pipeline.vlm 设为 None）。"""
    for img, view in items[:warmup]:
        pipeline.run(img, view)
    stages: dict[str, list[float]] = {}
    total = []
    t_all = time.perf_counter()
    for _ in range(repeat):
        for img, view in items:
            t0 = time.perf_counter()
            rec = pipeline.run(img, view)
            total.append((time.perf_counter() - t0) * 1000)
            for k, v in rec["latency_ms"].items():
                stages.setdefault(k, []).append(v)
    wall = time.perf_counter() - t_all
    n = len(total)
    out = {"images": n, "throughput_per_s": round(n / wall, 2) if wall else None,
           "total": {"p50_ms": percentile(total, 50), "p95_ms": percentile(total, 95), "mean_ms": round(statistics.fmean(total), 1)}}
    for k, v in stages.items():
        out[k] = {"p50_ms": percentile(v, 50), "p95_ms": percentile(v, 95), "mean_ms": round(statistics.fmean(v), 1)}
    return out


def bench_vlm(client, view: str, checks: list[str], images: list[str], concurrency: list[int], requests: int) -> list[dict]:
    """images 为 data URL 列表（轮流使用）。每个并发级别发 requests 次请求。"""
    rows = []
    for c in concurrency:
        jobs = [images[i % len(images)] for i in range(requests)]

        def one(img):
            rv = client.review(view, checks, [img])
            return rv.latency_ms, bool(rv.text), rv.parsed is not None

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=c) as ex:
            res = list(ex.map(one, jobs))
        wall = time.perf_counter() - t0
        lat = [r[0] for r in res]
        rows.append({
            "concurrency": c, "requests": requests, "wall_s": round(wall, 2),
            "throughput_per_s": round(requests / wall, 3) if wall else None,
            "p50_ms": percentile(lat, 50), "p95_ms": percentile(lat, 95),
            "ok_rate": round(sum(r[1] for r in res) / requests, 3),
            "json_ok_rate": round(sum(r[2] for r in res) / requests, 3),
        })
    return rows
