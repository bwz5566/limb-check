"""筛选严格度（仅标准库）：configs/strictness.yaml 的严格 / 标准 / 宽松三档。

每档覆盖两类东西：录用标准里的动作（marks，见 limbcheck.marks.policy）和仲裁阈值（arbitration）。
网页 demo 的 server.py 没有 PyYAML，读 export-web-config 生成的 JSON 里的同一份配置。
"""
from __future__ import annotations

import copy
import json
from functools import lru_cache

from .core import config_dir

_OVERRIDE: dict | None = None


@lru_cache(maxsize=1)
def _load() -> dict:
    try:
        import yaml
    except ImportError:
        web = config_dir().parent / "apps" / "web-demo" / "config.generated.json"
        return json.loads(web.read_text(encoding="utf-8"))["strictness"]
    with open(config_dir() / "strictness.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config() -> dict:
    return _OVERRIDE if _OVERRIDE is not None else _load()


def set_config(cfg: dict | None) -> None:
    global _OVERRIDE
    _OVERRIDE = cfg


def default_level() -> str:
    return config().get("default", "standard")


def levels() -> list[str]:
    return list(config()["levels"])


def level_def(name: str | None) -> dict:
    lv = config()["levels"]
    name = name or default_level()
    if name not in lv:
        raise ValueError(f"未知严格度 {name!r}，可选 {list(lv)}")
    return lv[name]


def deep_merge(base: dict, over: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def arbitration_cfg(base: dict | None, level: str | None) -> dict:
    """仲裁参数 = thresholds.yaml 的 arbitration 段 + 该档的覆盖（顶层值覆盖到每个检测项，per_check 里显式写的除外）。"""
    over = level_def(level).get("arbitration") or {}
    cfg = copy.deepcopy(base or {})
    per = cfg.get("per_check") or {}
    for k, v in over.items():
        if k == "per_check":
            continue
        cfg[k] = v
        for c in per.values():          # 档位覆盖统一阈值，但保留各检测项的策略名
            if k in c and k != "strategy":
                c[k] = max(c[k], v) if k == "high_conf" else v
    if "per_check" in over:
        cfg["per_check"] = deep_merge(per, over["per_check"])
    return cfg
