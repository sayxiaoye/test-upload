"""
统一日志模块（E3 新增）

提供配置驱动的日志系统，支持控制台 + 文件双输出，
与 config.yaml 中的 logging 配置集成。

核心设计:
- 配置驱动: 日志级别、格式从 config.yaml 读取
- 模块独立: 通过 get_logger(__name__) 获取专用 logger，互不干扰
- 双输出: 控制台（开发调试）+ 文件（持久化审计）
- 结构化: RAG 问答记录序列化为 JSON，方便离线分析和排查 bad case

面试话术:
"I built a unified logging layer on top of Python's standard logging module.
It's config-driven — log level, format, and output targets are read from
config.yaml. Each module gets its own logger via get_logger(__name__).
For RAG evaluation, I added structured JSON logging: every Q&A pair is
serialized with timestamp, answer, and sources for offline analysis."
"""

# ---- 导入区 ----
# logging: Python 标准库的日志模块，提供了 Logger、Handler、Formatter 等核心类
import logging
import sys  # sys.stdout: 标准输出流，控制台日志写到这
from datetime import datetime  # 给日志条目打时间戳
from pathlib import Path  # 面向对象的文件路径操作
from typing import Any  # Any 类型，用于灵活的结构化日志字段

# ---- 常量（默认值）----
# 这些默认值在 config.yaml 不可用时兜底
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"  # 日志行格式
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"  # 时间戳格式
DEFAULT_LEVEL = "INFO"  # 默认只输出 INFO 及以上级别

# 全局标记: 防止重复初始化（多次调用 setup_logging 只生效第一次）
_initialized: bool = False


def setup_logging(
    level: str | None = None,  # 日志级别，None 则从 config.yaml 读取
    log_file: str | None = None,  # 日志文件路径，None 则只输出到控制台
    log_format: str | None = None,  # 日志格式，None 则用默认格式
    force: bool = False,  # 是否强制重新初始化（切换环境时用）
) -> None:
    """初始化全局日志配置。

    只会执行一次（除非 force=True），避免重复添加 handler 导致日志重复。

    设计思路:
    - 优先从 config.yaml 读取配置，读取失败则用硬编码默认值
    - 控制台 handler 始终添加（开发时实时查看）
    - 文件 handler 可选（持久化审计日志）

    Args:
        level: 日志级别字符串 (DEBUG/INFO/WARNING/ERROR)，None 自动读取配置
        log_file: 日志文件路径，None 则只输出到控制台
        log_format: 日志格式字符串，None 用默认格式
        force: True 时强制重新初始化（清空旧 handler 重来）
    """
    global _initialized  # noqa: PLW0603
    if _initialized and not force:
        return  # 已初始化且不是强制模式，直接退出

    # --- 解析参数: 优先用传入值，其次从 config.yaml 读取，最后用默认值 ---
    resolved_level = level or DEFAULT_LEVEL
    resolved_format = log_format or DEFAULT_FORMAT

    try:
        # 尝试从项目配置中读取日志设定
        from src.core.config import get_config

        cfg = get_config()
        resolved_level = level or cfg.get("logging.level", DEFAULT_LEVEL)
        resolved_format = log_format or cfg.get("logging.format", DEFAULT_FORMAT)
    except Exception:
        pass  # 配置不可用时静默降级，用默认值

    # --- 配置根 logger ---
    root_logger = logging.getLogger()  # 获取根 logger（所有 logger 的祖先）
    root_logger.setLevel(
        getattr(logging, resolved_level.upper(), logging.INFO)
    )  # 字符串 → logging 常量

    # 清空已有 handler（force=True 时避免重复）
    root_logger.handlers.clear()

    # --- 控制台 handler ---
    # StreamHandler: 把日志输出到流（这里是 sys.stdout，即终端）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)  # 控制台看到所有级别
    console_formatter = logging.Formatter(resolved_format, DEFAULT_DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # --- 文件 handler（可选）---
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
        # FileHandler: 把日志追加写入文件
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)  # 文件只记录 INFO 及以上
        file_formatter = logging.Formatter(resolved_format, DEFAULT_DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """获取指定模块的 logger。

    首次调用自动执行 setup_logging()，后续调用复用已有配置。
    每个模块传入 __name__ 作为名称，logger 会按层级继承配置。

    用法:
        from src.utils.logging_utils import get_logger
        logger = get_logger(__name__)
        logger.info("RAG pipeline 初始化完成")

    Args:
        name: logger 名称，通常传 __name__（当前模块的完整路径名）

    Returns:
        配置好的 logging.Logger 实例
    """
    setup_logging()  # 确保至少初始化一次（有 _initialized 标记，不会重复）
    return logging.getLogger(name)


# ============================================================================
# 结构化日志（RAG 评估专用）
# ============================================================================
def log_rag_query(
    log_dir: str,  # 日志输出目录
    question: str,  # 用户问题
    answer: str,  # 模型回答
    sources: list[dict[str, Any]],  # 参考来源列表 [{"content":..., "score":...}, ...]
    context: str = "",  # 拼接后的完整上下文
    metadata: dict[str, Any] | None = None,  # 附加元数据（model alias、chunk 策略等）
) -> Path:
    """将一次 RAG 问答持久化为结构化 JSON 日志。

    与普通日志的区别:
    - 普通日志是自由文本，人阅读方便但程序解析困难
    - 结构化日志是 JSON，方便脚本批量分析、统计、可视化

    面试时可以说: "每次 RAG 问答都会生成结构化日志，
    包含 timestamp / question / answer / sources / metadata，
    方便后续做失败 case 分析和模型效果对比。"

    Args:
        log_dir: 日志目录路径
        question: 用户问题原文
        answer: 模型生成的回答
        sources: 参考来源列表
        context: 拼接后的检索上下文
        metadata: 额外信息（如使用的模型别名、切分策略等）

    Returns:
        写入的 JSON 日志文件路径
    """
    # 确保日志目录存在
    dir_path = Path(log_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    # 生成唯一文件名: 时间戳 + 问题哈希，避免重名
    timestamp = datetime.now()
    # hash() 可能为负数，用 & 0x7FFFFFFF 转为非负整数
    filename = (
        f"{timestamp.strftime('%Y%m%d_%H%M%S')}_"
        f"{hash(question) & 0x7FFFFFFF}.json"
    )

    # 组装结构化日志条目
    import json

    entry: dict[str, Any] = {
        "timestamp": timestamp.isoformat(),  # ISO 8601 格式，便于机器解析
        "question": question,
        "answer": answer,
        "sources": sources,  # 完整来源列表，含 content 和 score
        "context": context,
        "num_sources": len(sources),  # 来源数量，方便统计
    }
    if metadata:
        entry["metadata"] = metadata  # 附加信息

    output_path = dir_path / filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)  # indent=2 生成可读 JSON

    return output_path


# ============================================================================
# 便捷环境切换函数
# ============================================================================
def set_debug() -> None:
    """切换所有 logger 到 DEBUG 级别。

    排查问题时用: 可以看到所有级别的日志，包括详细的调试信息。
    面试时可以说: "开发调试时一键切到 DEBUG 模式，生产环境用 WARNING。"
    """
    setup_logging(level="DEBUG", force=True)  # force=True 强制覆盖已有配置


def set_quiet() -> None:
    """切换所有 logger 到 WARNING 级别。

    生产环境用: 只输出警告和错误，减少日志噪声。
    """
    setup_logging(level="WARNING", force=True)


# ============================================================================
# 演示入口
# ============================================================================
def demo() -> None:
    """运行日志模块功能演示。"""
    print("=" * 60)
    print("E3-4 统一日志模块演示")
    print("=" * 60)

    # 1. 控制台日志
    setup_logging(level="DEBUG")  # 先切到 DEBUG 看全貌
    logger = get_logger("demo")  # 获取名为 "demo" 的 logger

    logger.debug("这条是 DEBUG 级别 — 详细的调试信息")  # DEBUG: 最低级别
    logger.info("这条是 INFO 级别 — 关键流程节点")  # INFO: 比 DEBUG 高一档
    logger.warning("这条是 WARNING — 需要注意但不影响运行")  # WARNING: 警告

    # 2. 结构化日志演示
    print()
    print("─" * 60)
    path = log_rag_query(
        log_dir="logs/demo",  # 输出到 logs/demo 目录
        question="什么是向量数据库？",  # 模拟用户问题
        answer="向量数据库是专门存储和检索向量嵌入的数据库系统。",  # 模拟回答
        sources=[{"id": 1, "content": "向量数据库...", "score": 0.95}],  # 模拟来源
        metadata={"model_alias": "pro", "strategy": "semantic_chunk"},  # 附加入参信息
    )
    logger.info("结构化日志已保存到: %s", path)

    # 3. 展示所有公开 API
    print()
    print("─" * 60)
    print("公开 API 一览:")
    print("  setup_logging()  — 初始化全局日志配置（控制台+文件）")
    print("  get_logger(name) — 获取模块级 logger")
    print("  log_rag_query()  — RAG 问答结构化 JSON 日志")
    print("  set_debug()      — 一键切到 DEBUG 模式（排查问题用）")
    print("  set_quiet()      — 一键切到 WARNING 模式（生产环境用）")
    print("─" * 60)


if __name__ == "__main__":
    demo()
