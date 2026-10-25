"""可替换节点的注册表（仅标准库）。

后续换模型、加检测项、改仲裁策略时，不改主干代码，只需在自己的模块里注册，再在流水线配置里引用：

    from limbcheck.registry import register

    @register("keypoints", "my-hand-model")
    class MyHandModel:                     # 实现 hands(img) / bodies(img)
        name = "my-hand-model"
        def __init__(self, weights: str, device: str = "cuda"): ...

然后在 configs/pipelines/xxx.yaml 里写 `keypoints: {type: my-hand-model, weights: /models/x.onnx}`，
并在 `plugins:` 中列出该模块。也可以不注册，直接写完整路径 `type: mypkg.models:MyHandModel`。

节点类型（kind）：
    keypoints   关键点模型：hands(img) -> list[Hand]、bodies(img) -> list[Body]
    contour     手指轮廓复核后端：hand_mask(img, hand, th) -> (H, W) uint8
    segmenter   人体部位分割：mask(img, classes) -> (H, W) uint8
    l1          L1 判定函数（按检测项）：fn(ctx: L1Context) -> CheckResult | None
    arbiter     仲裁策略：fn(l1, l2, cfg, attempt) -> (decision, reason)
详见 docs/EXTENDING.md。
"""
from __future__ import annotations

import importlib
from collections import defaultdict
from typing import Any, Callable

_REGISTRY: dict[str, dict[str, Any]] = defaultdict(dict)
_META: dict[str, dict[str, dict]] = defaultdict(dict)
_BUILTINS = {
    "keypoints": "limbcheck.adapters",
    "contour": "limbcheck.adapters",
    "segmenter": "limbcheck.adapters",
    "l1": "limbcheck.l1checks",
    "arbiter": "limbcheck.arbiter",
}
KINDS = tuple(_BUILTINS)
_INSTANCES: list = []


def register(kind: str, name: str | None = None, **meta) -> Callable:
    if kind not in _BUILTINS:
        raise ValueError(f"未知节点类型 {kind}，可选 {KINDS}")

    def deco(obj):
        key = name or getattr(obj, "name", None) or obj.__name__
        _REGISTRY[kind][key] = obj
        _META[kind][key] = meta
        return obj

    return deco


def _ensure(kind: str) -> None:
    importlib.import_module(_BUILTINS[kind])


def load_plugins(modules: list[str] | None) -> None:
    """导入插件模块（模块内用 @register 注册自己的节点）。"""
    for m in modules or []:
        importlib.import_module(m)


def get(kind: str, name: str):
    if ":" in name:   # 完整路径：包.模块:属性
        mod, attr = name.split(":", 1)
        return getattr(importlib.import_module(mod), attr)
    _ensure(kind)
    if name not in _REGISTRY[kind]:
        raise KeyError(f"没有注册名为 {name!r} 的 {kind}，可选：{', '.join(names(kind))}")
    return _REGISTRY[kind][name]


def meta(kind: str, name: str) -> dict:
    _ensure(kind)
    return _META[kind].get(name, {})


def names(kind: str) -> list[str]:
    _ensure(kind)
    return sorted(_REGISTRY[kind])


def create(kind: str, spec: Any, **defaults):
    """按配置创建节点实例。spec 可以是 None / "none" / "名称" / "包.模块:类" / {"type": ..., 其他参数}。"""
    if spec in (None, "", "none", False):
        return None
    if isinstance(spec, str):
        spec = {"type": spec}
    spec = dict(spec)
    factory = get(kind, spec.pop("type"))
    obj = factory(**{**defaults, **spec})
    if hasattr(obj, "close"):
        _INSTANCES.append(obj)
    return obj


def close_all() -> None:
    """释放 create 创建的、带 close() 的模型节点（MediaPipe 1.x 需在解释器退出前关闭，否则退出时打印析构报错）。"""
    while _INSTANCES:
        try:
            _INSTANCES.pop().close()
        except Exception:   # 释放失败不影响已经完成的结果
            pass
