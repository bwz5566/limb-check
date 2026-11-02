"""扩展性：只改配置（configs/checks.json + 提示词 + 岗位画像）就能新增一个检测项，核心代码不动。

在子进程里用 LIMBCHECK_CONFIG_DIR 指向一份改过的配置副本，验证新检测项能贯通
L2 提示词 → 输出解析 → 仲裁 → 标注表导入 → 指标计算。示例检测项：首饰（jewelry）。
"""
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


def _make_config(tmp: Path) -> Path:
    cfg = tmp / "configs"
    shutil.copytree(REPO / "configs", cfg)
    checks = json.loads((cfg / "checks.json").read_text(encoding="utf-8"))
    checks["checks"]["jewelry"] = {"name": "首饰", "views": ["backs", "hand"], "basis_required": False, "appearance": True}
    (cfg / "checks.json").write_text(json.dumps(checks, ensure_ascii=False), encoding="utf-8")
    pfile = cfg / "prompts" / f"{checks['default_prompt_version']}.json"
    prompts = json.loads(pfile.read_text(encoding="utf-8"))
    prompts["checks"]["jewelry"] = {
        "rule": "jewelry：对每只手各输出 1 条，part 为 \"right_hand\" 或 \"left_hand\"；label 为 \"none\"（未佩戴）或 \"ring\"（戒指、手链等）。",
        "positive": ["ring"], "negative": ["none"], "per_hand": True}
    pfile.write_text(json.dumps(prompts, ensure_ascii=False), encoding="utf-8")
    p = cfg / "job_profiles" / "food.yaml"
    prof = yaml.safe_load(p.read_text(encoding="utf-8"))
    prof["checks"]["jewelry"] = {"enabled": True, "mode": "judge"}
    p.write_text(yaml.safe_dump(prof, allow_unicode=True), encoding="utf-8")
    return cfg


SCRIPT = textwrap.dedent("""
    import json, sys
    from pathlib import Path
    import numpy as np
    from limbcheck.core import IMAGE_CHECKS, CheckResult
    from limbcheck.prompts import l2_checks_for, build_user_prompt, parse_reply, to_check_results
    from limbcheck.arbiter import arbitrate
    from limbcheck.config import load_job_profiles, check_mode
    from limbcheck.dataset import import_folder, save_image, load_samples
    from limbcheck.metrics import compute

    tmp = Path(sys.argv[1])
    assert "jewelry" in IMAGE_CHECKS
    checks = l2_checks_for("backs")
    assert "jewelry" in checks and "首饰（jewelry）" in build_user_prompt("backs", checks)
    reply = json.dumps({"items": [{"check": "jewelry", "part": "right_hand", "label": "ring", "confidence": 0.9},
                                  {"check": "jewelry", "part": "left_hand", "label": "none", "confidence": 0.9}],
                        "image_quality": "good"})
    parsed, errs = parse_reply(reply, ["jewelry"])
    res = to_check_results(parsed, ["jewelry"], "backs", "L2:test", errs)["jewelry"]
    assert res.positive is True and res.parts == ["right_hand"], res
    prof = load_job_profiles()["food"]
    assert check_mode(prof, "jewelry") == "judge"
    a = arbitrate("jewelry", None, res, {}, check_mode(prof, "jewelry"), 1)
    assert a.decision == "fail_suggested"

    (tmp / "raw").mkdir()
    for n in ("a_backs.jpg", "b_backs.jpg"):
        save_image(tmp / "raw" / n, np.full((20, 20, 3), 128, np.uint8))
    (tmp / "labels.csv").write_text("image,view,jewelry\\na_backs.jpg,backs,是\\nb_backs.jpg,backs,否\\n", encoding="utf-8")
    import_folder(tmp / "raw", tmp / "labels.csv", tmp / "ds")
    samples = load_samples(tmp / "ds")
    assert [s.truth["jewelry"] for s in samples] == [True, False] and "jewelry" not in samples[0].meta
    preds = {s.id: {"results": {"jewelry": {"l2": CheckResult("jewelry", "L2:test", s.truth["jewelry"], 0.9).to_dict(),
                                            "shadow": "fail_suggested" if s.truth["jewelry"] else "pass"}}} for s in samples}
    m = compute(samples, preds)
    assert m["checks"]["jewelry"]["image"]["system"]["recall"]["value"] == 1.0
    print("ok")
""")


def test_add_check_by_config_only(tmp_path):
    cfg = _make_config(tmp_path)
    env = {**os.environ, "LIMBCHECK_CONFIG_DIR": str(cfg), "PYTHONPATH": str(REPO)}
    r = subprocess.run([sys.executable, "-c", SCRIPT, str(tmp_path)], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-3000:]
