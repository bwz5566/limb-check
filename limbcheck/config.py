"""读取 configs/ 下的阈值、岗位画像、验收标准，并导出网页 demo 用的 JSON。"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import yaml

from .core import APPEARANCE_CHECKS, BASIS_REQUIRED, CHECK_DEFS, CHECKS, config_dir as _config_dir
from .prompts import prompt_version

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_CONFIG_PATH = REPO_ROOT / "apps" / "web-demo" / "config.generated.json"
MODELS_DIR = Path(os.environ.get("LIMBCHECK_MODELS_DIR", REPO_ROOT / "models"))

MODES = ("judge", "record_only")


def config_dir() -> Path:
    d = _config_dir()
    if not d.exists():
        raise FileNotFoundError(f"找不到配置目录 {d}。请在仓库内用 `pip install -e .` 安装，或设置 LIMBCHECK_CONFIG_DIR。")
    return d


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_thresholds(path: str | Path | None = None) -> dict:
    return _load_yaml(Path(path) if path else config_dir() / "thresholds.yaml")


def load_acceptance(path: str | Path | None = None) -> dict:
    return _load_yaml(Path(path) if path else config_dir() / "acceptance.yaml")


def normalize_profile(p: dict) -> dict:
    """补全缺省项，并强制执行"外观项无书面依据只能 record_only"。"""
    p = dict(p)
    checks = {}
    for c in CHECKS:
        spec = dict((p.get("checks") or {}).get(c) or {"enabled": False})
        spec.setdefault("enabled", False)
        spec.setdefault("mode", "judge")
        spec.setdefault("params", {})
        if spec["mode"] not in MODES:
            raise ValueError(f"岗位 {p.get('id')} 的 {c}.mode 只能是 {MODES}，实际为 {spec['mode']}")
        checks[c] = spec
    covered = {c for b in p.get("basis") or [] for c in b.get("checks") or []}
    for c in BASIS_REQUIRED:
        if checks[c]["enabled"] and checks[c]["mode"] == "judge" and c not in covered:
            warnings.warn(f"岗位 {p.get('id')} 的 {c} 设为 judge 但没有书面依据（basis），已降级为 record_only")
            checks[c]["mode"] = "record_only"
    p["checks"] = checks
    p.setdefault("steps", [])
    p.setdefault("requirements", [])
    return p


def load_job_profiles(directory: str | Path | None = None) -> dict[str, dict]:
    d = Path(directory) if directory else config_dir() / "job_profiles"
    profiles = {}
    for f in sorted(d.glob("*.yaml")):
        p = normalize_profile(_load_yaml(f))
        profiles[p["id"]] = p
    return profiles


def check_mode(profile: dict | None, check: str) -> str:
    """返回 judge / record_only / off。没有岗位画像时全部按 judge 评测。"""
    if profile is None:
        return "judge"
    spec = profile["checks"].get(check) or {}
    if not spec.get("enabled"):
        return "off"
    return spec.get("mode", "judge")


def web_config(thresholds: dict | None = None, profiles: dict | None = None) -> dict:
    """网页 demo 使用的配置（单一来源：configs/*.yaml）。"""
    th = thresholds or load_thresholds()
    profiles = profiles or load_job_profiles()
    order = ["precision", "line", "warehouse", "food"]
    jobs = [profiles[k] for k in order if k in profiles] + [v for k, v in profiles.items() if k not in order]
    return {
        "_comment": "由 `limbcheck export-web-config` 从 configs/ 生成，请勿手改",
        "prompt_version": prompt_version(),
        "checks": CHECK_DEFS["checks"],
        "views": CHECK_DEFS["views"],
        "thresholds": th,
        "jobs": jobs,
        "appearance_checks": list(APPEARANCE_CHECKS),
    }


def export_web_config(path: str | Path | None = None) -> Path:
    path = Path(path) if path else WEB_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(web_config(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def model_path(name: str) -> Path:
    p = MODELS_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"找不到模型文件 {p}（可用 LIMBCHECK_MODELS_DIR 指定目录）")
    return p
