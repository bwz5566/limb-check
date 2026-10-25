"""MediaPipe 实跑冒烟测试（需要 mediapipe，Linux 还需要 libegl1 / libgles2）。

使用 MediaPipe 官方测试图（运行时下载，不随仓库分发）；无法联网时跳过。
这组测试同时固定了两条容易搞反的约定：
- Tasks 版 HandLandmarker 对未镜像照片给出的左右就是本人左右；
- 手性：右手手背朝前 orient_sign = +1。
"""
import os
import urllib.request
from pathlib import Path

import pytest

mp = pytest.importorskip("mediapipe")

CACHE = Path(os.environ.get("LIMBCHECK_TEST_ASSETS", Path(__file__).parent / ".assets"))
BASE = "https://storage.googleapis.com/mediapipe-assets/"


def asset(name: str) -> Path:
    p = CACHE / name
    if not p.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(BASE + name, p)
        except Exception as e:
            pytest.skip(f"无法下载测试图 {name}：{e}")
    return p


@pytest.fixture(scope="module")
def kp():
    from limbcheck.registry import create

    try:
        m = create("keypoints", "mediapipe")
    except OSError as e:   # 缺 libEGL 等系统库
        pytest.skip(f"MediaPipe 无法加载：{e}")
    yield m
    m.close()


@pytest.mark.parametrize("name,side", [("right_hands.jpg", "right"), ("left_hands.jpg", "left")])
def test_handedness_and_facing(kp, name, side):
    from limbcheck.dataset import load_image
    from limbcheck.geometry import facing

    hands = kp.hands(load_image(asset(name)))
    assert len(hands) == 2
    for h in hands:
        assert h.side == side
        assert facing(h.pts, h.side) == "back"     # 两张图都是手背朝向镜头


def test_pipeline_on_real_photo(kp, thresholds):
    from limbcheck.dataset import load_image
    from limbcheck.pipeline import Pipeline
    from limbcheck.registry import create

    p = Pipeline(thresholds, keypoints=kp, contour=create("contour", "skin"))
    rec = p.run(load_image(asset("victory.jpg")), "hand", {"side": "right"})
    assert rec["results"]["fingers"]["l1"]["positive"] is False
