"""测试模型对比演示模块（E3）。"""

from unittest.mock import patch

from src.rag.model_compare import show_alias_system


class TestShowAliasSystem:
    """测试别名系统展示函数（纯本地，不调 API）。"""

    def test_show_alias_runs_without_error(self, capsys):
        """show_alias_system 正常执行，输出包含关键信息。"""
        show_alias_system()
        captured = capsys.readouterr()
        assert "模型别名系统说明" in captured.out
        assert "已配置别名" in captured.out
        assert "默认别名" in captured.out
        assert "三级路由优先级" in captured.out
        assert "config.yaml" in captured.out

    def test_show_alias_lists_configured_aliases(self, capsys):
        """输出包含配置中的别名（fast、pro）。"""
        show_alias_system()
        captured = capsys.readouterr()
        assert "fast" in captured.out
        assert "pro" in captured.out
        assert "deepseek-v4" in captured.out


class TestCompareRagQaWithMocks:
    """用 mock 测试 RAG 对比逻辑，避免消费 API。"""

    def test_compare_rag_qa_mocked_llm(self, capsys):
        """当 LLM 返回固定值时，函数正常完成。"""
        from src.rag.model_compare import compare_rag_qa

        # Mock LLMClient.chat_with_template 返回固定值
        with patch(
            "src.rag.model_compare.LLMClient",
            autospec=True,
        ) as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.chat_with_template.side_effect = [
                "fast 模型的回答",  # 第一次调用 (fast)
                "pro 模型的回答会更详细一些",  # 第二次调用 (pro)
            ]

            compare_rag_qa()

        captured = capsys.readouterr()
        assert "模型对比演示" in captured.out
        assert "fast 模型" in captured.out or "fast" in captured.out.lower()

    def test_compare_rag_qa_handles_api_error(self, capsys):
        """API 调用失败时打印错误但不崩溃。"""
        from src.rag.model_compare import compare_rag_qa

        with patch(
            "src.rag.model_compare.LLMClient",
            autospec=True,
        ) as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.chat_with_template.side_effect = RuntimeError("API 不可用")

            # 不应抛出异常
            compare_rag_qa()

        captured = capsys.readouterr()
        # 应包含调用失败提示
        assert "调用失败" in captured.out


class TestCompareViaBatchMethod:
    """用 mock 测试批量对比方法。"""

    def test_batch_compare_mocked(self, capsys):
        """compare_models 返回固定结果时正常展示。"""
        from src.rag.model_compare import compare_via_batch_method

        with patch(
            "src.rag.model_compare.LLMClient",
            autospec=True,
        ) as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.compare_models.return_value = [
                {"alias": "fast", "model": "deepseek-v4-flash", "answer": "fast 的摘要"},
                {"alias": "pro", "model": "deepseek-v4-pro", "answer": "pro 的摘要"},
            ]

            compare_via_batch_method()

        captured = capsys.readouterr()
        assert "批量对比演示" in captured.out
        assert "fast" in captured.out
        assert "pro" in captured.out

    def test_batch_compare_handles_error_entry(self, capsys):
        """结果中含 error 字段时正确显示错误信息。"""
        from src.rag.model_compare import compare_via_batch_method

        with patch(
            "src.rag.model_compare.LLMClient",
            autospec=True,
        ) as MockClient:
            mock_instance = MockClient.return_value
            mock_instance.compare_models.return_value = [
                {
                    "alias": "fast",
                    "model": "deepseek-v4-flash",
                    "answer": "",
                    "error": "连接超时",
                },
            ]

            compare_via_batch_method()

        captured = capsys.readouterr()
        assert "错误" in captured.out
        assert "连接超时" in captured.out


class TestMainFunction:
    """测试 main 入口函数（模拟用户选择不调 API）。"""

    def test_main_exits_gracefully_on_no(self, monkeypatch, capsys):
        """用户输入 n 时，main 展示别名系统后正常退出。"""
        from src.rag.model_compare import main

        # 模拟用户输入：两次都选 n
        inputs = iter(["n", "n"])
        monkeypatch.setattr("builtins.input", lambda _="": next(inputs))

        main()

        captured = capsys.readouterr()
        assert "E3-3 模型切换演示" in captured.out
        assert "模型别名系统说明" in captured.out
        assert "演示结束" in captured.out
