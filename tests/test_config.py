import json
import warnings

from limbcheck.config import WEB_CONFIG_PATH, check_mode, load_job_profiles, load_thresholds, normalize_profile, web_config
from limbcheck.core import CHECKS, VIEW_CHECKS


def test_profiles_load_and_appearance_needs_basis():
    profiles = load_job_profiles()
    assert set(profiles) >= {"precision", "line", "warehouse", "food"}
    for p in profiles.values():
        covered = {c for b in p.get("basis") or [] for c in b.get("checks") or []}
        for c in ("tattoo", "scar", "hair"):
            assert check_mode(p, c) in ("record_only", "off") or c in covered
        assert check_mode(p, "hair") == "judge" and "hair" in covered     # 发色 / 发长标准已由招聘方提供
        assert set(p["checks"]) == set(CHECKS)


def test_judge_without_basis_is_downgraded():
    p = {"id": "t", "checks": {"tattoo": {"enabled": True, "mode": "judge"}}}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = normalize_profile(p)
    assert out["checks"]["tattoo"]["mode"] == "record_only" and w
    p["basis"] = [{"checks": ["tattoo"], "doc": "员工手册", "clause": "3.2"}]
    assert normalize_profile(p)["checks"]["tattoo"]["mode"] == "judge"


def test_thresholds_have_required_sections():
    th = load_thresholds()
    for k in ("capture", "quality", "pose", "fingers", "contour", "nails", "hair", "arbitration"):
        assert k in th


def test_views_cover_all_image_checks():
    covered = {c for cs in VIEW_CHECKS.values() for c in cs}
    assert covered == {"limbs", "fingers", "nails", "tattoo", "scar", "hair"}


def test_web_config_in_sync():
    """网页 demo 读取的配置必须由 configs/ 生成。失败时运行：limbcheck export-web-config"""
    current = json.loads(WEB_CONFIG_PATH.read_text(encoding="utf-8"))
    assert current == json.loads(json.dumps(web_config(), ensure_ascii=False)), "请运行 limbcheck export-web-config"
