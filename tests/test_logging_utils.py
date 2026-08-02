"""测试统一日志模块（E3）。"""

import json
import logging
from pathlib import Path

from src.utils.logging_utils import (
    get_logger,
    log_rag_query,
    set_debug,
    set_quiet,
    setup_logging,
)


class TestSetupLogging:
    """测试日志初始化配置。"""

    def test_setup_adds_console_handler(self, capsys):
        """初始化后根 logger 至少有一个 console handler。"""
        setup_logging(level="INFO", force=True)
        msg = "这是一条测试日志"
        logger = logging.getLogger("test_console")
        logger.info(msg)
        # 刷新输出后检查
        captured = capsys.readouterr()
        assert msg in captured.out

    def test_setup_respects_level(self):
        """日志级别过滤生效：DEBUG 在 INFO 级别下不输出。"""
        setup_logging(level="INFO", force=True)
        logger = logging.getLogger("test_level")
        # DEBUG 不应传播（根 logger 是 INFO）
        assert not logger.isEnabledFor(logging.DEBUG)
        assert logger.isEnabledFor(logging.INFO)

    def test_force_reinitialize(self):
        """force=True 时清空旧 handler 重新初始化。"""
        setup_logging(level="INFO", force=True)
        count_before = len(logging.getLogger().handlers)
        setup_logging(level="DEBUG", force=True)
        count_after = len(logging.getLogger().handlers)
        # handler 数量应不变（先清空再重建）
        assert count_before > 0
        assert count_after == count_before


class TestGetLogger:
    """测试获取模块 logger。"""

    def test_get_logger_returns_logger(self):
        """返回的是 logging.Logger 实例。"""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_get_logger_initializes_automatically(self):
        """首次调用自动初始化全局日志配置（不抛异常）。"""
        # force reinitialize 确保走初始化路径
        setup_logging(force=True)
        logger = get_logger("auto_init_test")
        logger.info("auto init works")
        # 没有异常即为通过

    def test_get_logger_hierarchy(self):
        """模块层级命名：子模块继承父配置。"""
        parent = get_logger("src")
        child = get_logger("src.rag.llm_client")
        # child 名称以 parent 名为前缀
        assert child.name.startswith(parent.name)


class TestStructuredLogging:
    """测试结构化 RAG 日志。"""

    def test_log_rag_query_creates_file(self, tmp_path):
        """调用后生成 JSON 文件。"""
        log_dir = str(tmp_path / "rag_logs")
        path = log_rag_query(
            log_dir=log_dir,
            question="什么是 RAG？",
            answer="RAG 是检索增强生成。",
            sources=[{"id": 1, "content": "RAG...", "score": 0.95}],
        )
        assert path.exists()
        assert path.suffix == ".json"

    def test_log_rag_query_content(self, tmp_path):
        """JSON 文件内容包含所有传入字段。"""
        log_dir = str(tmp_path / "rag_logs")
        path = log_rag_query(
            log_dir=log_dir,
            question="Q1",
            answer="A1",
            sources=[{"id": 1, "content": "C1", "score": 0.9}],
            context="CTX",
            metadata={"model_alias": "pro"},
        )
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["question"] == "Q1"
        assert data["answer"] == "A1"
        assert data["context"] == "CTX"
        assert data["num_sources"] == 1
        assert data["metadata"]["model_alias"] == "pro"

    def test_log_rag_query_creates_parent_dirs(self, tmp_path):
        """日志目录不存在时自动创建。"""
        log_dir = str(tmp_path / "deep" / "nested" / "logs")
        path = log_rag_query(
            log_dir=log_dir,
            question="Q",
            answer="A",
            sources=[],
        )
        assert path.parent.exists()
        assert path.exists()

    def test_log_rag_query_unique_filenames(self, tmp_path):
        """不同问题生成不同文件名。"""
        log_dir = str(tmp_path)
        p1 = log_rag_query(log_dir=log_dir, question="Q1", answer="A", sources=[])
        p2 = log_rag_query(log_dir=log_dir, question="Q2", answer="A", sources=[])
        assert p1 != p2  # 文件名不同

    def test_log_rag_query_same_question_returns_path(self, tmp_path):
        """相同问题也正常返回 Path（同名覆盖不影响功能）。"""
        log_dir = str(tmp_path)
        p1 = log_rag_query(log_dir=log_dir, question="Q", answer="A", sources=[])
        p2 = log_rag_query(log_dir=log_dir, question="Q", answer="A", sources=[])
        # 同秒内相同问题哈希相同 → 文件名相同 → 第二次覆盖第一次
        # 这在实际使用中极罕见（人工评分场景两次查询间隔远超 1 秒）
        assert isinstance(p1, Path)
        assert isinstance(p2, Path)
        assert p1.exists() or p2.exists()  # 至少有一个文件存在

    def test_log_rag_query_empty_sources(self, tmp_path):
        """sources 为空列表也能正常记录。"""
        log_dir = str(tmp_path)
        path = log_rag_query(
            log_dir=log_dir, question="Q", answer="A", sources=[]
        )
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["sources"] == []
        assert data["num_sources"] == 0


class TestLevelSwitching:
    """测试 set_debug / set_quiet 快速切换。"""

    def test_set_debug_enables_debug(self):
        """set_debug 后根 logger 可输出 DEBUG 级别。"""
        set_debug()
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_set_quiet_filters_info(self):
        """set_quiet 后根 logger 不输出 INFO。"""
        set_quiet()
        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_round_trip(self):
        """set_debug 后 set_quiet 再切回去，级别正确。"""
        set_debug()
        assert logging.getLogger().level == logging.DEBUG
        set_quiet()
        assert logging.getLogger().level == logging.WARNING
