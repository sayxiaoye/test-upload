"""
提示词模板管理模块

提供模板注册、参数化渲染、版本管理等能力。
将业务逻辑与提示词文本分离，方便复用、调试和 A/B 测试。

设计思路：
- 每个模板是独立单元，包含名字、描述、system/user 消息模板
- 支持 {variable} 占位符语法，渲染时替换为实际值
- PromptRegistry 作为全局注册中心，统一管理所有模板

面试话术（如何向面试官解释这个模块）：
"I designed a prompt template management system that separates prompt text
from business logic. Each template has a name, description, and version.
It supports variable interpolation so prompts can be parameterized without
string concatenation in code. This enables centralized management, easier
A/B testing, and allows non-engineers to tune prompts without touching code."
"""

# ---- 导入区 ----
# from __future__ import annotations: 开启"延迟求值注解"（PEP 563），
# 让所有类型注解被当作字符串处理，避免循环引用问题，同时提升导入速度
from __future__ import annotations

from dataclasses import (  # dataclass: 自动生成 __init__/__repr__ 等方法的装饰器; field: 定义字段默认值
    dataclass,
    field,
)
from string import (
    Formatter,  # Python 标准库中的字符串格式化解析器，用于提取 {变量} 占位符
)

# ============================================================================
# 失败样例清单（面试时可以说"我们记录了这些失败 case 来驱动 prompt 优化"）
# ============================================================================
# FAILURE_CATALOG: 记录历史上 prompt 出过的问题、原因和修复方案
# 这是一个"经验知识库"，当修改模板时可回溯之前的坑，避免重复犯错
FAILURE_CATALOG: list[dict[str, str]] = [
    {
        "problem": "模型在 RAG 问答中编造了文档里不存在的数据",
        "cause": "system prompt 未强调「只根据参考文档回答」，给了模型太大自由度",
        "fix": "在 system prompt 中加入硬约束：'如果文档中没有相关信息，请直接回答：未找到相关信息'",
        "template": "rag_qa",  # 关联到哪个模板
    },
    {
        "problem": "文档摘要输出过长，超过 3 句话",
        "cause": "prompt 只说'简洁'但没给出具体长度约束",
        "fix": "将约束改为'用 2-3 句话总结，不超过 100 字'",
        "template": "document_summary",
    },
    {
        "problem": "相关性评分中，部分相关文档被评了 10 分",
        "cause": "评分标准缺少具体锚点，模型倾向于给高分",
        "fix": "为每个分数段补具体示例，让模型有参照",
        "template": "relevance_eval",
    },
    {
        "problem": "多轮对话中丢失了前文的上下文",
        "cause": "user template 未包含对话历史占位符",
        "fix": "在 user_template 中加入 {conversation_history} 变量",
        "template": "rag_qa",
    },
]


# ============================================================================
# 模板数据类
# ============================================================================
# @dataclass 装饰器: 告诉 Python 这是一个"数据类"——
# 自动生成 __init__、__repr__、__eq__ 等方法，减少样板代码
@dataclass
class PromptTemplate:
    """单个提示词模板。

    每个模板包含 system 消息和 user 消息两部分，
    支持 {variable} 占位符，渲染时替换为实际值。

    字段说明:
        name (str):           模板唯一标识名，如 "rag_qa"
        description (str):    人类可读的模板用途描述
        system_prompt (str):  系统提示词——定义模型的角色、行为规则和输出格式
        user_template (str):  用户消息模板——包含占位符，如 "参考文档：{context}"
        version (str):        语义化版本号 (SemVer)，格式 "主版本.次版本.修订号"
        metadata (dict):      附加元数据（分类、作者、适用场景等），用于检索和管理
    """

    name: str  # 模板唯一名称，注册中心的 key
    description: str  # 模板用途的一句话描述
    system_prompt: str  # system 角色的提示词文本（可含占位符）
    user_template: str  # user 角色的提示词文本（可含占位符）
    version: str = "1.0.0"  # 默认版本号，遵循语义化版本规范
    metadata: dict[str, str] = field(default_factory=dict)  # field(default_factory=dict): 每个实例独立的空字典，避免多实例共享同一可变对象

    def _get_variables(self, text: str) -> set[str]:
        """提取文本中的占位符变量名。

        使用 Python 标准库的 Formatter().parse() 解析文本，
        找出所有 {变量名} 形式的占位符。

        Formatter().parse() 返回 (literal_text, field_name, format_spec, conversion) 四元组，
        这里只取 field_name（变量名），筛掉 field_name 为 None 的纯文本片段。

        Args:
            text: 待解析的模板文本

        Returns:
            变量名的集合（去重），如 {"query", "context"}
        """
        return {
            field_name  # 提取变量名
            for _, field_name, _, _ in Formatter().parse(text)  # 遍历解析结果，解包为 (前缀文本, 变量名, 格式化规格, 转换函数)
            if field_name is not None  # 只保留有变量名的片段（纯文本片段的 field_name 为 None）
        }

    @property
    def variables(self) -> set[str]:
        """返回 system + user 模板中所有变量名集合。

        @property: 把这个方法变成"只读属性"，调用方写 tmpl.variables 即可，
        无需加括号写成 tmpl.variables()。集合的并集操作 | 自动去重。
        """
        return self._get_variables(self.system_prompt) | self._get_variables(self.user_template)

    def render(self, **kwargs: str) -> list[dict[str, str]]:
        """用实际参数渲染模板，返回 OpenAI 格式的消息列表。

        **kwargs: 关键字参数，参数名对应模板中的占位符名。
        例如 render(query="什么是RAG", context="RAG是...")。

        执行流程:
        1. 检查是否缺少必需变量（模板中有的、但 kwargs 没提供的）
        2. 用 str.format(**kwargs) 替换模板中的 {变量} 为实际值
        3. 返回 OpenAI Chat Completions API 标准格式的消息列表

        Args:
            **kwargs: 模板变量值，key 对应占位符名

        Returns:
            [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            这是 OpenAI/大多数 LLM API 的标准消息格式

        Raises:
            KeyError: 缺少必需变量时抛出，错误信息明确列出缺失的变量名
        """
        # 计算缺失变量: 模板需要的变量 - 调用者提供的变量
        missing = self.variables - set(kwargs)
        if missing:
            # 提前失败，给出清晰的错误提示，而不是等到 format() 报 KeyError
            raise KeyError(
                f"模板 '{self.name}' 缺少变量: {', '.join(sorted(missing))}"
            )

        # 构造 OpenAI 标准格式的消息列表:
        # - system 消息: 定义 AI 的角色和行为规范
        # - user 消息: 包含具体的问题/任务
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt.format(**kwargs)},  # str.format() 将 {变量} 替换为实际值
            {"role": "user", "content": self.user_template.format(**kwargs)},
        ]
        return messages

    def preview(self) -> str:
        """纯文本预览模板结构（不含渲染）。

        用于开发调试——快速查看模板的全貌:
        名称、版本、描述、变量列表、system/user 文本。

        注: 这里展示的是原始模板（含占位符），而非渲染后的结果。
        """
        lines = [
            f"━━━ {self.name} v{self.version} ━━━",  # 模板头部: 名称 + 版本
            f"描述: {self.description}",  # 人类可读的描述
            f"变量: {', '.join(sorted(self.variables)) if self.variables else '(无)'}",  # 列出所有占位符变量
            "--- System ---",
            self.system_prompt,  # 原始 system prompt（含占位符）
            "--- User ---",
            self.user_template,  # 原始 user template（含占位符）
            "━━━━━━━━━━━━━━━━━━━━━━━",
        ]
        return "\n".join(lines)  # 用换行符连接所有行


# ============================================================================
# 模板注册中心
# ============================================================================
class PromptRegistry:
    """模板注册中心，集中管理所有 PromptTemplate。

    设计模式: 注册表模式 (Registry Pattern)
    - 内部用 dict 存储 name → template 的映射
    - 提供 CRUD（增删改查）+ 渲染 + 概览等能力
    - 配合下方的 get_registry() 懒加载函数实现全局单例

    用法：
        registry = PromptRegistry()
        registry.register(rag_qa_template)
        messages = registry.render("rag_qa", query="...", context="...")
    """

    def __init__(self) -> None:
        # _templates: 私有属性，用 _ 前缀表示"内部使用，不要直接访问"
        # 存储 模板名 → PromptTemplate 实例 的映射
        self._templates: dict[str, PromptTemplate] = {}

    # --- CRUD 操作 ---
    def register(self, template: PromptTemplate) -> None:
        """注册一个模板（同名会覆盖）。

        设计选择: 同名覆盖而非报错——
        方便开发时动态重载模板，也支持 A/B 测试时替换不同版本。
        """
        self._templates[template.name] = template  # 以模板 name 为 key 存入字典

    def get(self, name: str) -> PromptTemplate:
        """按名称获取模板。

        当模板不存在时，抛出 KeyError 并列出所有可用模板名，
        帮助调用方快速发现拼写错误或缺失注册的问题。
        """
        if name not in self._templates:
            # 友好错误: 告诉调用者有哪些模板可用
            available = ", ".join(sorted(self._templates)) or "<empty>"
            raise KeyError(
                f"未找到模板 '{name}'，可用模板: {available}"
            )
        return self._templates[name]

    def list_names(self) -> list[str]:
        """列出所有已注册模板名称（按字母排序）。

        用于动态展示可选模板列表，例如 CLI 工具中让用户选择。
        """
        return sorted(self._templates)  # sorted(dict) 默认对 key 排序

    def list_templates(self) -> list[PromptTemplate]:
        """列出所有已注册模板对象。

        与 list_names() 的区别: 返回完整的 PromptTemplate 对象而非仅名称，
        用于需要读取模板详情的场景（如导出、迁移、对比）。
        """
        return list(self._templates.values())

    # --- 批量渲染 ---
    def render(self, name: str, **kwargs: str) -> list[dict[str, str]]:
        """渲染指定模板。

        这是调用方最常用的方法——先按名称查找模板，再传入参数渲染。

        Example:
            registry.render("rag_qa", query="什么是RAG?", context="RAG是...")
            # 返回: [{"role": "system", ...}, {"role": "user", ...}]
        """
        return self.get(name).render(**kwargs)  # 链式调用: 获取 → 渲染

    # --- 总结 ---
    def summary(self) -> str:
        """打印注册中心概况。

        输出所有已注册模板的摘要信息:
        - 模板名称和版本
        - 变量数量
        - 一句话描述

        用于开发调试和运维巡检。
        """
        lines = ["=" * 60, "📋 PromptRegistry 概况", "=" * 60]  # 表格头部
        for name in self.list_names():  # 遍历所有模板名
            tmpl = self._templates[name]
            var_count = len(tmpl.variables)  # 统计变量个数
            lines.append(f"  [{name}] v{tmpl.version} ({var_count} 个变量)")
            lines.append(f"    → {tmpl.description}")
        lines.append(f"\n共 {len(self._templates)} 个模板\n" + "=" * 60)  # 总数统计
        return "\n".join(lines)


# ============================================================================
# 预置模板（面试时可以说"我预置了 3 类核心模板，覆盖主要场景"）
# ============================================================================
# 以下 4 个 PromptTemplate 实例是预置模板，在 get_registry() 首次调用时自动注册
# 覆盖了 RAG 场景的 3 个核心环节: 检索问答、文档摘要、相关性评估 + 1 个辅助场景: 代码审查

# ---- 模板 1: RAG 问答 ----
# 这是 RAG（Retrieval-Augmented Generation, 检索增强生成）的核心模板
# 工作流程: 先检索相关文档 → 将文档作为 context 传入 → 模型基于文档回答问题
RAG_QA_TEMPLATE = PromptTemplate(
    name="rag_qa",  # 模板唯一标识
    description="RAG 问答：基于参考文档回答问题，要求引用来源",
    version="1.1.0",  # 已迭代过（初始 1.0.0 → 修复幻觉问题后升到 1.1.0）
    system_prompt=(
        "你是一个严谨的知识问答助手。请**只根据**以下参考文档回答问题。\n"  # 角色定义 + 核心约束
        "规则：\n"
        "1. 如果文档中有明确答案，请引用相关片段并给出回答\n"  # 有答案 → 引用 + 回答
        "2. 如果文档中没有相关信息，请直接回答：「根据现有资料，未找到相关信息」\n"  # 无答案 → 诚实告知（防幻觉）
        "3. 回答要简洁、准确，用中文输出\n"  # 输出风格控制
        "4. 如果答案涉及多条文档，请分条列出"  # 格式化指引
    ),
    user_template=(
        "参考文档：\n"
        "{context}\n\n"  # {context}: 检索到的参考文档内容，渲染时替换
        "用户问题：{query}"  # {query}: 用户的实际问题，渲染时替换
    ),
    metadata={
        "category": "qa",
        "author": "E3-prompt-templates",
        "best_for": "单轮 RAG 问答场景",
        "failure_note": "见 FAILURE_CATALOG 第 1、4 条",  # 引用了上方的失败样例，形成可追溯的改进记录
    },
)

# ---- 模板 2: 文档摘要 ----
# 用于快速摘要场景: 给定一篇文档，用 2-3 句话概括核心内容
# 关键设计: 将抽象约束（"简洁"）具象化为数字约束（"2-3 句话，不超过 100 字"）
DOCUMENT_SUMMARY_TEMPLATE = PromptTemplate(
    name="document_summary",
    description="文档摘要：用 2-3 句话概括文档核心内容",
    version="1.0.0",
    system_prompt=(
        "你是一个专业的文档分析师。请用**2-3 句话**总结以下文档的核心内容，"
        "总字数不超过 100 字。\n"  # 两个硬约束: 句数 + 字数——让模型有明确的行为边界
        "要求：\n"
        "- 抓住文档的核心主题，而非细节\n"  # 引导模型区分主题 vs 细节
        "- 使用中文输出\n"
        "- 如果文档内容不足以做摘要，请如实说明"  # 空文档 / 极短文档的兜底逻辑
    ),
    user_template=(
        "文档标题：{title}\n"  # {title}: 文档标题，提供上下文线索
        "文档内容：\n{document_text}"  # {document_text}: 文档正文
    ),
    metadata={
        "category": "summary",
        "author": "E3-prompt-templates",
        "best_for": "单文档快速摘要",
        "failure_note": "见 FAILURE_CATALOG 第 2 条",
    },
)

# ---- 模板 3: 相关性评估 ----
# 用于评估检索结果的质量: 给每个召回的文档片段打分 (0-10)
# 关键设计: 用"锚定示例"替代纯文字描述，让评分标准具体可参照
RELEVANCE_EVAL_TEMPLATE = PromptTemplate(
    name="relevance_eval",
    description="相关性评估：对召回片段与查询的相关性打分（0-10）",
    version="1.0.0",
    system_prompt=(
        "你是一个检索质量评估专家。请评估每个文档片段与用户查询的相关性。\n\n"
        "评分标准（锚定示例）：\n"  # "锚定"是 prompt 工程技巧: 用具体 case 让模型有参照系
        "- 10 分：片段直接完整回答了查询（如查询「Python 是什么」→ 片段「Python 是一门解释型语言…」）\n"
        "- 7-9 分：片段回答了部分查询，但不够完整\n"
        "- 4-6 分：片段与查询相关但未直接回答问题\n"
        "- 0-3 分：片段与查询几乎无关\n\n"
        "请以 JSON 格式输出评分，格式：\n"  # 要求结构化输出，方便下游代码解析
        '{{"scores": [{{"doc_id": 1, "score": 8, "reason": "…"}}]}}'  # 双花括号: 在 Python f-string 场景下转义为单花括号；此处虽是普通字符串但保留转义可防意外
    ),
    user_template=(
        "用户查询：{query}\n\n"  # {query}: 用户的原始查询
        "待评估的文档片段：\n{documents}"  # {documents}: 召回的所有文档片段
    ),
    metadata={
        "category": "evaluation",
        "author": "E3-prompt-templates",
        "best_for": "RAG 召回质量评估、reranker 对比",
        "failure_note": "见 FAILURE_CATALOG 第 3 条",
    },
)

# ---- 模板 4: 代码审查 ----
# 代码审查辅助模板: 对 Python 代码片段做系统性审查
# 覆盖 4 个审查维度: bug、规范、性能、类型安全
CODE_REVIEW_TEMPLATE = PromptTemplate(
    name="code_review",
    description="代码审查：对 Python 代码片段做审查，指出问题和改进建议",
    version="1.0.0",
    system_prompt=(
        "你是一个资深 Python 代码审查员。请审查以下代码片段，关注：\n"
        "1. 潜在 bug 或逻辑错误\n"  # 正确性维度
        "2. 不符合 PEP 8 或项目规范的写法\n"  # 规范性维度
        "3. 性能问题或可简化之处\n"  # 性能/简洁性维度
        "4. 类型安全问题\n\n"  # 类型安全维度
        "请以分条方式输出，每条以「[类别]」开头。"  # 统一输出格式，例如 "[Bug] 第5行存在空指针风险"
    ),
    user_template=(
        "代码上下文：{context}\n\n"  # {context}: 代码的上下文说明（如"这段代码用于处理用户登录"）
        "待审查代码：\n```python\n{code}\n```"  # {code}: 实际代码内容，用 markdown 代码块包裹
    ),
    metadata={
        "category": "development",
        "author": "E3-prompt-templates",
        "best_for": "代码审查、教学辅助",
    },
)


# ============================================================================
# 全局注册中心（单例）
# ============================================================================
# _registry: 模块级私有变量，存储全局唯一的 PromptRegistry 实例
# 使用 None 标记"尚未初始化"状态，配合 get_registry() 实现懒加载
_registry: PromptRegistry | None = None  # 类型注解: PromptRegistry 或 None


def get_registry() -> PromptRegistry:
    """获取全局模板注册中心（懒加载，自动注册预置模板）。

    懒加载 (Lazy Initialization) 模式:
    - 首次调用时才创建 PromptRegistry 并注册预置模板
    - 后续调用直接返回同一个实例（单例）
    - 优点: 导入模块时不产生任何开销，只在真正使用时才初始化

    线程安全说明:
    - 在 CPython 中，由于 GIL（全局解释器锁）的存在，
      这个简单的 if _registry is None 检查是线程安全的
    - 如果需要在多进程或有并发要求的场景使用，
      可以考虑用 threading.Lock 加锁保护
    """
    global _registry  # noqa: PLW0603 — 声明要修改模块级全局变量 _registry
    if _registry is None:  # 首次调用: 初始化
        _registry = PromptRegistry()  # 创建注册中心实例
        # 注册 4 个预置模板，覆盖 RAG 核心场景
        _registry.register(RAG_QA_TEMPLATE)  # 模板 1: RAG 问答
        _registry.register(DOCUMENT_SUMMARY_TEMPLATE)  # 模板 2: 文档摘要
        _registry.register(RELEVANCE_EVAL_TEMPLATE)  # 模板 3: 相关性评估
        _registry.register(CODE_REVIEW_TEMPLATE)  # 模板 4: 代码审查
    return _registry  # 返回单例实例


# ============================================================================
# 便捷函数（给外部模块用）
# ============================================================================
# 这些函数的目的是简化调用——
# 外部模块不需要先获取 registry 再调用 render，只需一行函数调用即可
# 每个函数内部调用 get_registry().render() 完成模板渲染

def render_rag_qa(query: str, context: str) -> list[dict[str, str]]:
    """快捷渲染 RAG 问答模板。

    适用场景: 用户提出问题 → 系统检索相关文档 → 调用本函数生成带上下文的问答提示词

    Args:
        query: 用户原始问题，如 "什么是机器学习？"
        context: 检索到的参考文档内容，作为模型回答的依据

    Returns:
        OpenAI 格式消息列表，可直接传给 LLM API 的 messages 参数
    """
    return get_registry().render("rag_qa", query=query, context=context)


def render_document_summary(title: str, document_text: str) -> list[dict[str, str]]:
    """快捷渲染文档摘要模板。

    Args:
        title: 文档标题，如 "Python 入门指南"
        document_text: 文档全文内容

    Returns:
        OpenAI 格式消息列表
    """
    return get_registry().render(
        "document_summary", title=title, document_text=document_text
    )


def render_relevance_eval(query: str, documents: str) -> list[dict[str, str]]:
    """快捷渲染相关性评估模板。

    适用场景: 评估 RAG 检索质量——召回的内容和用户问题到底有多相关？

    Args:
        query: 用户的原始查询
        documents: 被召回的文档片段列表（序列化为字符串）

    Returns:
        OpenAI 格式消息列表，模型会按要求返回 JSON 格式的评分
    """
    return get_registry().render("relevance_eval", query=query, documents=documents)


# ============================================================================
# 演示入口
# ============================================================================
def demo() -> None:
    """运行提示词模板演示。

    这个函数通过 4 个演示展示模块的核心功能:
    1. 模板概况概览 (summary)
    2. 模板的正常渲染流程
    3. 缺少变量时的错误提示（防御性编程的体现）
    4. 失败样例追溯（工程经验的体现）

    运行方式: python prompt_templates.py
    """
    registry = get_registry()  # 获取全局注册中心（首次调用会触发懒加载）

    # 打印所有已注册模板的概况
    print(registry.summary())
    print()

    # ---- 演示 1: RAG 问答渲染 ----
    print("─" * 60)
    print("📌 演示 1: RAG 问答模板渲染")
    print("─" * 60)
    messages = registry.render(
        "rag_qa",
        query="什么是 RAG？",  # 模拟用户问题
        context="RAG（检索增强生成）是一种结合检索和生成的技术。",  # 模拟检索到的文档
    )
    for msg in messages:
        # 截取前 150 个字符预览，避免输出过长
        print(f"[{msg['role']}]\n{msg['content'][:150]}...\n")

    # ---- 演示 2: 文档摘要渲染 ----
    print("─" * 60)
    print("📌 演示 2: 文档摘要模板渲染")
    print("─" * 60)
    messages = registry.render(
        "document_summary",
        title="Python 入门指南",
        document_text="Python 是一门解释型、面向对象的编程语言...",
    )
    for msg in messages:
        print(f"[{msg['role']}]\n{msg['content'][:150]}...\n")

    # ---- 演示 3: 缺少变量的报错 ----
    print("─" * 60)
    print("📌 演示 3: 缺少变量时的错误提示")
    print("─" * 60)
    try:
        # 故意只传 query、不传 context，触发 KeyError
        registry.render("rag_qa", query="你好")  # 缺少 context
    except KeyError as e:
        # 验证: 错误信息中应明确列出缺少哪个变量
        print(f"✅ 预期错误: {e}")

    # ---- 演示 4: 失败样例清单 ----
    print()
    print("─" * 60)
    print("📌 失败样例清单（面试时可以说'我们记录了这些来驱动 prompt 优化'）")
    print("─" * 60)
    # 遍历 FAILURE_CATALOG，展示历史上踩过的坑和修复方案
    for i, case in enumerate(FAILURE_CATALOG, 1):  # enumerate(..., 1): 序号从 1 开始（更符合人类阅读习惯）
        print(f"\n  [{i}] 问题: {case['problem']}")
        print(f"     原因: {case['cause']}")
        print(f"     修复: {case['fix']}")
        print(f"     关联模板: {case['template']}")


# ---- 脚本直接运行时的入口 ----
# if __name__ == "__main__": 当 python prompt_templates.py 直接执行时，__name__ 为 "__main__"，执行 demo()
# 当被 import 导入时，__name__ 为 "prompt_templates"，不执行 demo() —— 避免被导入时自动运行
if __name__ == "__main__":
    demo()
