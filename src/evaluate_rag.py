"""兼容层：保留旧的 src.evaluate_rag 导入与执行路径。"""

from __future__ import annotations

from src.eval.evaluate_rag import (  # noqa: F401
    DEFAULT_DOCUMENT,
    DEFAULT_LOG_DIR,
    DEFAULT_QUESTIONS,
    RAGEvaluator,
    RAGPipeline,
    analyze_results,
    build_parser,
    generate_logs,
    main,
)

if __name__ == "__main__":
    raise SystemExit(main())
