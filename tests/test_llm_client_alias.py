from typing import Any, cast

from src.rag.llm_client import LLMClient


class _DummyCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        class _Message:
            content = "ok"

        class _Choice:
            message = _Message()

        class _Response:
            choices = [_Choice()]

        return _Response()


class _DummyChat:
    def __init__(self, completions):
        self.completions = completions


class _DummyClient:
    def __init__(self, completions):
        self.chat = _DummyChat(completions)


def test_chat_uses_model_alias():
    llm = LLMClient()
    llm.model_aliases = {"fast": "deepseek-v4-flash", "pro": "deepseek-v4-flash"}
    llm.default_alias = "fast"

    completions = _DummyCompletions()
    llm.client = cast(Any, _DummyClient(completions))

    result = llm.chat([{"role": "user", "content": "hi"}], model_alias="pro")

    assert result == "ok"
    assert completions.calls[0]["model"] == "deepseek-v4-flash"


def test_chat_raises_for_unknown_alias():
    llm = LLMClient()
    llm.model_aliases = {"fast": "deepseek-v4-flash"}

    # 注入 mock 客户端以跳过 API Key 检查
    completions = _DummyCompletions()
    llm.client = cast(Any, _DummyClient(completions))

    try:
        llm.chat([{"role": "user", "content": "hi"}], model_alias="unknown")
    except ValueError as error:
        assert "未知模型别名" in str(error)
        assert "fast" in str(error)
    else:
        raise AssertionError("expected ValueError for unknown alias")
