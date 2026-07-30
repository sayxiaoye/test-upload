"""兼容层：保留旧的 src.batch_score 导入与执行路径。"""

from __future__ import annotations

from src.eval.batch_score import batch_score

if __name__ == "__main__":
    batch_score()
