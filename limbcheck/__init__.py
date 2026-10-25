"""远程肢体完整性检测：M0 验证框架。

本包刻意保持顶层轻量：`limbcheck.core`、`limbcheck.registry`、`limbcheck.prompts`、`limbcheck.vlm`、`limbcheck.arbiter`
只依赖 Python 标准库，网页 demo 的 server.py 会直接复用它们；
其余模块（几何、模型适配、评测）依赖 numpy / opencv，按需导入。
"""

__version__ = "0.1.0"
