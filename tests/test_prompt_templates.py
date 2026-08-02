"""测试提示词模板管理模块（E3）。"""

import pytest

from src.rag.prompt_templates import (
    CODE_REVIEW_TEMPLATE,
    DOCUMENT_SUMMARY_TEMPLATE,
    FAILURE_CATALOG,
    RAG_QA_TEMPLATE,
    RELEVANCE_EVAL_TEMPLATE,
    PromptRegistry,
    PromptTemplate,
    get_registry,
    render_document_summary,
    render_rag_qa,
    render_relevance_eval,
)


# ============================================================================
# PromptTemplate 单元测试
# ============================================================================
class TestPromptTemplate:
    """测试单个模板的变量提取、渲染、预览等能力。"""

    def test_variables_extraction(self):
        """变量提取: 正确提取 system + user 中的所有 {占位符}。"""
        tmpl = PromptTemplate(
            name="test",
            description="测试",
            system_prompt="你是{role}",
            user_template="问题：{query}，参考：{context}",
        )
        assert tmpl.variables == {"role", "query", "context"}

    def test_variables_empty(self):
        """无占位符时返回空集合。"""
        tmpl = PromptTemplate(
            name="test",
            description="测试",
            system_prompt="你是助手",
            user_template="请回答问题",
        )
        assert tmpl.variables == set()

    def test_render_basic(self):
        """基本渲染: 替换所有占位符，返回 OpenAI 格式。"""
        tmpl = PromptTemplate(
            name="test",
            description="测试",
            system_prompt="你是{role}",
            user_template="问题：{query}",
        )
        messages = tmpl.render(role="助手", query="你好")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "你是助手"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "问题：你好"

    def test_render_missing_variable(self):
        """缺少变量时抛出 KeyError，错误信息列出缺失变量名。"""
        tmpl = PromptTemplate(
            name="test",
            description="测试",
            system_prompt="你是{role}",
            user_template="{query}，参考{context}",
        )
        with pytest.raises(KeyError, match="缺少变量"):
            tmpl.render(role="助手")  # 缺少 query 和 context

    def test_render_missing_variable_message(self):
        """KeyError 信息包含模板名和缺失变量名。"""
        tmpl = PromptTemplate(
            name="rag_qa",
            description="测试",
            system_prompt="你是助手",
            user_template="{query}——{context}",
        )
        with pytest.raises(KeyError) as exc_info:
            tmpl.render(query="你好")  # 缺少 context
        assert "rag_qa" in str(exc_info.value)
        assert "context" in str(exc_info.value)

    def test_preview_contains_info(self):
        """模板预览包含名称、版本、描述和变量列表。"""
        tmpl = PromptTemplate(
            name="test",
            description="测试模板",
            system_prompt="sys",
            user_template="{x}",
            version="2.0.0",
        )
        preview = tmpl.preview()
        assert "test" in preview
        assert "2.0.0" in preview
        assert "测试模板" in preview
        assert "x" in preview  # 变量名出现在预览中

    def test_version_default(self):
        """默认版本号为 1.0.0。"""
        tmpl = PromptTemplate(
            name="test", description="", system_prompt="", user_template=""
        )
        assert tmpl.version == "1.0.0"

    def test_metadata_default(self):
        """metadata 默认是独立空字典（非共享）。"""
        a = PromptTemplate(name="a", description="", system_prompt="", user_template="")
        b = PromptTemplate(name="b", description="", system_prompt="", user_template="")
        a.metadata["key"] = "val"
        assert "key" not in b.metadata  # 互不影响


# ============================================================================
# PromptRegistry 单元测试
# ============================================================================
class TestPromptRegistry:
    """测试注册中心的增删查改、渲染和概览。"""

    def test_register_and_get(self):
        """注册后可按名获取。"""
        reg = PromptRegistry()
        tmpl = PromptTemplate(name="t", description="", system_prompt="", user_template="")
        reg.register(tmpl)
        assert reg.get("t") is tmpl

    def test_get_missing_raises(self):
        """获取不存在的模板时抛出 KeyError，列出可用模板。"""
        reg = PromptRegistry()
        reg.register(
            PromptTemplate(name="a", description="", system_prompt="", user_template="")
        )
        with pytest.raises(KeyError, match="未找到模板"):
            reg.get("nonexistent")
        with pytest.raises(KeyError, match="a"):  # 错误信息列出可用模板
            reg.get("nonexistent")

    def test_list_names_sorted(self):
        """list_names 返回排序后的名称列表。"""
        reg = PromptRegistry()
        reg.register(PromptTemplate(name="c", description="", system_prompt="", user_template=""))
        reg.register(PromptTemplate(name="a", description="", system_prompt="", user_template=""))
        reg.register(PromptTemplate(name="b", description="", system_prompt="", user_template=""))
        assert reg.list_names() == ["a", "b", "c"]

    def test_render_delegates(self):
        """注册中心的 render 正确代理到模板的 render。"""
        reg = PromptRegistry()
        reg.register(
            PromptTemplate(
                name="t",
                description="",
                system_prompt="你是{role}",
                user_template="{msg}",
            )
        )
        messages = reg.render("t", role="助手", msg="你好")
        assert messages[0]["content"] == "你是助手"
        assert messages[1]["content"] == "你好"

    def test_summary_content(self):
        """概况输出包含模板数量和名称。"""
        reg = PromptRegistry()
        reg.register(
            PromptTemplate(
                name="test_tmpl",
                description="一个测试",
                system_prompt="",
                user_template="",
            )
        )
        text = reg.summary()
        assert "test_tmpl" in text
        assert "一个测试" in text
        assert "共 1 个模板" in text

    def test_register_overwrite(self):
        """同名注册覆盖旧模板。"""
        reg = PromptRegistry()
        old = PromptTemplate(
            name="t", description="旧", system_prompt="", user_template=""
        )
        new = PromptTemplate(
            name="t", description="新", system_prompt="", user_template=""
        )
        reg.register(old)
        reg.register(new)
        assert reg.get("t").description == "新"


# ============================================================================
# 预置模板测试
# ============================================================================
class TestPresetTemplates:
    """验证 4 个预置模板的结构完整性。"""

    @pytest.mark.parametrize(
        "tmpl,expected_vars",
        [
            (RAG_QA_TEMPLATE, {"query", "context"}),
            (DOCUMENT_SUMMARY_TEMPLATE, {"title", "document_text"}),
            (RELEVANCE_EVAL_TEMPLATE, {"query", "documents"}),
            (CODE_REVIEW_TEMPLATE, {"context", "code"}),
        ],
    )
    def test_preset_variables(self, tmpl, expected_vars):
        """预置模板的变量名与预期一致。"""
        assert tmpl.variables == expected_vars

    def test_rag_qa_render(self):
        """RAG 问答模板可正确渲染。"""
        messages = RAG_QA_TEMPLATE.render(query="Q", context="C")
        assert "Q" in messages[1]["content"]
        assert "C" in messages[1]["content"]
        assert "参考文档" in messages[0]["content"]  # system prompt 含规则

    def test_document_summary_render(self):
        """文档摘要模板可正确渲染。"""
        messages = DOCUMENT_SUMMARY_TEMPLATE.render(title="T", document_text="D")
        assert "T" in messages[1]["content"]
        assert "D" in messages[1]["content"]

    def test_relevance_eval_render(self):
        """相关性评估模板可正确渲染。"""
        messages = RELEVANCE_EVAL_TEMPLATE.render(query="Q", documents="D")
        assert "Q" in messages[1]["content"]
        assert "D" in messages[1]["content"]
        # system prompt 要求输出 JSON
        assert "JSON" in messages[0]["content"]

    def test_code_review_render(self):
        """代码审查模板可正确渲染。"""
        messages = CODE_REVIEW_TEMPLATE.render(context="ctx", code="print(1)")
        assert "ctx" in messages[1]["content"]
        assert "print(1)" in messages[1]["content"]


# ============================================================================
# 全局注册中心与便捷函数
# ============================================================================
class TestGlobalRegistry:
    """测试 get_registry 单例和便捷函数。"""

    def test_get_registry_returns_same_instance(self):
        """多次调用返回同一实例（单例）。"""
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_get_registry_has_four_templates(self):
        """全局注册中心预置 4 个模板。"""
        reg = get_registry()
        assert len(reg.list_names()) == 4

    def test_render_rag_qa_convenience(self):
        """便捷函数 render_rag_qa 可正常渲染。"""
        messages = render_rag_qa(query="Q", context="C")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "Q" in messages[1]["content"]

    def test_render_document_summary_convenience(self):
        """便捷函数 render_document_summary 可正常渲染。"""
        messages = render_document_summary(title="T", document_text="D")
        assert len(messages) == 2

    def test_render_relevance_eval_convenience(self):
        """便捷函数 render_relevance_eval 可正常渲染。"""
        messages = render_relevance_eval(query="Q", documents="D")
        assert len(messages) == 2


# ============================================================================
# 失败样例清单
# ============================================================================
class TestFailureCatalog:
    """验证失败样例清单的结构完整性。"""

    def test_catalog_not_empty(self):
        """失败样例清单非空。"""
        assert len(FAILURE_CATALOG) >= 3

    def test_catalog_has_required_fields(self):
        """每条记录包含 problem / cause / fix / template 四个字段。"""
        for item in FAILURE_CATALOG:
            assert "problem" in item
            assert "cause" in item
            assert "fix" in item
            assert "template" in item
            assert item["template"] in {
                "rag_qa",
                "document_summary",
                "relevance_eval",
            }
