"""测试 LLMClient 纯逻辑方法（不调 API，E5 补测试）。"""

from src.rag.llm_client import LLMClient


class TestValidateMessages:
    """消息验证逻辑测试。"""

    def test_empty_list_raises(self):
        """空消息列表抛出 ValueError。"""
        client = LLMClient()
        try:
            client._validate_messages([])
            raise AssertionError("应该抛出 ValueError")
        except ValueError as e:
            assert "不能为空" in str(e)

    def test_valid_messages_pass(self):
        """合法消息不抛异常。"""
        client = LLMClient()
        client._validate_messages([
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ])

    def test_missing_role_raises(self):
        """缺少 role 字段抛出 ValueError。"""
        client = LLMClient()
        try:
            client._validate_messages([{"content": "hello"}])
            raise AssertionError("应该抛出 ValueError")
        except ValueError as e:
            assert "role" in str(e)

    def test_missing_content_raises(self):
        """缺少 content 字段抛出 ValueError。"""
        client = LLMClient()
        try:
            client._validate_messages([{"role": "user"}])
            raise AssertionError("应该抛出 ValueError")
        except ValueError as e:
            assert "content" in str(e)

    def test_invalid_role_raises(self):
        """非法 role 值抛出 ValueError。"""
        client = LLMClient()
        try:
            client._validate_messages([{"role": "invalid", "content": "x"}])
            raise AssertionError("应该抛出 ValueError")
        except ValueError as e:
            assert "role" in str(e)

    def test_non_dict_raises(self):
        """消息不是 dict 时抛出 ValueError。"""
        client = LLMClient()
        try:
            client._validate_messages(["not a dict"])  # type: ignore
            raise AssertionError("应该抛出 ValueError")
        except ValueError as e:
            assert "字典" in str(e)

    def test_non_string_content_raises(self):
        """content 不是字符串时抛出 ValueError。"""
        client = LLMClient()
        try:
            client._validate_messages([{"role": "user", "content": 123}])  # type: ignore
            raise AssertionError("应该抛出 ValueError")
        except ValueError as e:
            assert "字符串" in str(e)


class TestResolveModel:
    """模型解析逻辑测试。"""

    def test_direct_model_highest_priority(self):
        """直接传 model 名称优先级最高。"""
        client = LLMClient()
        result = client._resolve_model(model="custom-model", model_alias=None)
        assert result == "custom-model"

    def test_model_overrides_alias(self):
        """同时传 model 和 alias 时 model 优先。"""
        client = LLMClient()
        client.model_aliases = {"pro": "deepseek-v4-pro"}
        result = client._resolve_model(model="explicit-model", model_alias="pro")
        assert result == "explicit-model"

    def test_alias_not_in_config_raises(self):
        """别名不存在时抛出 ValueError 并列出可用别名。"""
        client = LLMClient()
        client.model_aliases = {"fast": "deepseek-v4-flash"}
        try:
            client._resolve_model(model=None, model_alias="unknown")
            raise AssertionError("应该抛出 ValueError")
        except ValueError as e:
            assert "未知模型别名" in str(e)
            assert "fast" in str(e)

    def test_no_args_uses_default(self):
        """不传参数时使用默认模型。"""
        client = LLMClient()
        client.model_aliases = {"fast": "deepseek-v4-flash"}
        client.default_alias = "fast"
        result = client._resolve_model(model=None, model_alias=None)
        assert result == "deepseek-v4-flash"


class TestInitDefaults:
    """初始化默认值测试。"""

    def test_empty_aliases_handled(self):
        """aliases 为空时不会崩溃。"""
        client = LLMClient()
        client.model_aliases = {}
        # 不传任何参数时使用 default_model
        assert client._resolve_model(None, None)
