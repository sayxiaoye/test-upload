"""
大模型客户端模块
支持 DeepSeek API 调用、模型别名切换、提示词模板集成

核心能力:
- 封装 OpenAI 兼容的 LLM 调用
- 模型别名系统: 配置中定义别名(如 fast/pro)，调用时一键切换
- 模板集成: chat_with_template() 直接使用 prompt_templates 模块
- 模型对比: compare_models() 一次请求对比多个模型表现

面试话术（如何向面试官解释多模型切换）:
"I built a model alias system in the LLM client. Aliases like 'fast' and 'pro'
are defined in config.yaml and map to actual model names. This means you can
switch models by changing config, not code. Simple queries go to the fast model
to save cost, complex reasoning goes to the pro model for quality."
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

from src.core.config import get_config  # 项目配置管理，读取 config.yaml

# 加载 .env 文件中的环境变量（API Key 等敏感信息）
load_dotenv()


class LLMClient:
    """大模型客户端

    封装 OpenAI 兼容的 API 调用，支持:
    - 模型别名切换 (fast/pro)
    - 消息格式验证
    - 提示词模板集成
    - 多模型对比
    """

    # 允许的消息角色类型，用于 _validate_messages() 校验
    VALID_ROLES = {"system", "user", "assistant", "tool"}

    def __init__(self):
        # --- 初始化 OpenAI 兼容客户端 ---
        # api_key: 从 .env 中读取 DEEPSEEK_API_KEY，不硬编码在代码里
        # base_url: DeepSeek API 地址，也允许通过环境变量覆盖
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        )

        # --- 读取配置 ---
        config = get_config()  # 获取全局配置单例

        # --- 模型别名系统 ---
        # 从 config.yaml 的 models.aliases 段读取别名映射
        # 例如: {"fast": "deepseek-v4-flash", "pro": "deepseek-v4-pro"}
        aliases = config.get("models.aliases", {}) or {}
        if isinstance(aliases, dict):
            # 确保 key 和 value 都是字符串
            self.model_aliases = {str(key): str(value) for key, value in aliases.items()}
        else:
            self.model_aliases = {}

        # 默认使用的别名（对应 config.yaml 的 models.default_alias）
        self.default_alias = str(config.get("models.default_alias", "fast"))

        # 兜底模型名：如果别名没找到，回退到环境变量或硬编码值
        fallback_model = os.getenv("DEEPSEEK_MODEL_FLASH", "deepseek-v4-flash")
        self.default_model = (
            os.getenv("DEEPSEEK_MODEL")  # 优先级 1: 环境变量直接指定
            or self.model_aliases.get(self.default_alias)  # 优先级 2: 默认别名对应的模型
            or fallback_model  # 优先级 3: 兜底值
        )

        # --- 生成参数默认值（从配置读取，没有则用硬编码兜底）---
        self.default_temperature = float(
            config.get("models.generation.temperature", 0.7)
        )
        self.default_max_tokens = int(
            config.get("models.generation.max_tokens", 1000)
        )

    def _resolve_model(self, model: str | None, model_alias: str | None) -> str:
        """解析最终使用的模型名称。

        三级优先级（从高到低）:
        1. model 参数 — 直接传入模型名（最高优先）
        2. model_alias 参数 — 通过别名查找（如 "fast" → "deepseek-v4-flash"）
        3. 默认别名对应的模型 — 从 config.yaml 读取

        这样设计的好处:
        - 调用方可以精确指定模型名（model="deepseek-v4-pro"）
        - 也可以只用别名切换（model_alias="pro"），由配置决定实际模型
        - 都不传就用默认配置，新手开箱即用

        Args:
            model: 直接指定的模型名称，优先级最高
            model_alias: 模型别名，如 "fast" 或 "pro"

        Returns:
            解析后的实际模型名称字符串

        Raises:
            ValueError: 提供的别名在配置中不存在时抛出
        """
        # 优先级 1: 直接指定了 model 名称，直接用
        if model:
            return model

        # 优先级 2: 传了别名，从 model_aliases 字典中查找实际模型名
        if model_alias:
            if model_alias in self.model_aliases:
                return self.model_aliases[model_alias]
            # 别名不存在时，列出可用别名帮助排查
            available = ", ".join(sorted(self.model_aliases)) or "<empty>"
            raise ValueError(
                f"未知模型别名: {model_alias}，可用别名: {available}"
            )

        # 优先级 3: 使用默认别名对应的模型
        if self.default_alias in self.model_aliases:
            return self.model_aliases[self.default_alias]

        # 最终兜底: 使用 __init__ 中计算好的 default_model
        return self.default_model

    def _validate_messages(self, messages: list[dict[str, str]]) -> None:
        """
        验证消息格式是否正确

        Args:
            messages: 消息列表

        Raises:
            ValueError: 消息格式不正确时抛出
        """
        if not messages:
            raise ValueError("消息列表不能为空")

        for i, msg in enumerate(messages):
            # 检查是否是字典
            if not isinstance(msg, dict):
                raise ValueError(f"消息 {i} 必须是字典类型，实际类型: {type(msg)}")

            # 检查 role 字段
            if "role" not in msg:
                raise ValueError(f"消息 {i} 缺少 'role' 字段")

            # 检查 content 字段
            if "content" not in msg:
                raise ValueError(f"消息 {i} 缺少 'content' 字段")

            # 检查 role 值是否合法
            if msg["role"] not in self.VALID_ROLES:
                raise ValueError(
                    f"消息 {i} 的 role 必须是 {', '.join(self.VALID_ROLES)}，"
                    f"当前值: {msg['role']}"
                )

            # 检查 content 类型
            if not isinstance(msg["content"], str):
                raise ValueError(
                    f"消息 {i} 的 content 必须是字符串，"
                    f"实际类型: {type(msg['content'])}"
                )

            # 检查 content 是否为空（非必须，但建议）
            if not msg["content"].strip():
                # 发出警告，但不中断执行
                print(f"⚠️ 警告: 消息 {i} 的 content 为空或只有空白字符")

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,  # 直接指定模型名，优先级最高
        model_alias: str | None = None,  # 模型别名（fast/pro），比 model 低一档
        temperature: float | None = None,  # None 表示使用配置默认值
        max_tokens: int | None = None,  # None 表示使用配置默认值
    ) -> str:
        """调用大模型，返回文本回答。

        参数设计思路:
        - model/model_alias 灵活切换模型，换模型不用改代码
        - temperature/max_tokens 为 None 时自动从配置读取默认值
        - 配合 chat_with_template() 可走"模板 → 模型路由"完整链路

        Args:
            messages: 消息列表 [{"role":"user","content":"..."}]
            model: 精确模型名，如 "deepseek-v4-pro"（优先级最高）
            model_alias: 模型别名，如 "fast" 或 "pro"（通过配置映射）
            temperature: 温度(0-1)，None 则用配置默认值 0.7
            max_tokens: 最大输出 token 数，None 则用配置默认值 1000

        Returns:
            模型返回的文本内容（content 为 None 时返回空字符串）

        Raises:
            ValueError: 消息格式不正确时抛出
            openai.APIError: API 调用失败时抛出
        """
        # 1. 验证消息格式（检查 role、content 等字段是否合法）
        self._validate_messages(messages)

        # 2. 解析模型名：三级优先级 model > model_alias > 默认配置
        resolved_model = self._resolve_model(model=model, model_alias=model_alias)

        # 3. 解析温度: 传了值就用，没传就用配置默认值
        resolved_temperature = (
            self.default_temperature if temperature is None else temperature
        )

        # 4. 解析 max_tokens: 传了值就用，没传就用配置默认值
        resolved_max_tokens = (
            self.default_max_tokens if max_tokens is None else max_tokens
        )

        # 5. 发起 API 调用
        try:
            response = self.client.chat.completions.create(
                model=resolved_model,
                messages=messages,  # type: ignore
                temperature=resolved_temperature,
                max_tokens=resolved_max_tokens,
            )

            # 6. 提取返回内容
            # response.choices[0].message.content 可能是 None（某些模型可能不返回文本）
            content = response.choices[0].message.content
            return str(content) if content is not None else ""
        except Exception as e:
            # 打印错误方便排查，然后向上抛出让调用方处理
            print(f"❌ API 调用失败: {e}")
            raise

    # ========================================================================
    # E3 新增: 模板集成 + 模型对比
    # ========================================================================
    def chat_with_template(
        self,
        template_name: str,  # 模板名，如 "rag_qa"
        model_alias: str | None = None,  # 模型别名，None 用默认
        temperature: float | None = None,  # 温度，None 用配置默认
        max_tokens: int | None = None,  # 最大 token，None 用配置默认
        **template_vars: str,  # 传给模板的变量（query=..., context=...等）
    ) -> str:
        """使用提示词模板调用模型（E3 新增: 桥接 prompt_templates 模块）。

        这条方法的定位: 把"选模板 → 填变量 → 选模型 → 调 API"封装成一步，
        让调用方不用关心模板渲染和模型路由的细节。

        典型调用:
            client = LLMClient()
            answer = client.chat_with_template(
                "rag_qa",
                model_alias="pro",  # 复杂问答用 pro 模型
                query="什么是 RAG？",
                context="RAG 是检索增强生成...",
            )

        Args:
            template_name: 注册在 PromptRegistry 中的模板名
            model_alias: 模型别名（fast/pro），None 则用配置中的默认别名
            temperature: 温度参数，None 则用配置默认值
            max_tokens: 最大输出 token，None 则用配置默认值
            **template_vars: 模板变量，key 对应模板中的 {占位符}

        Returns:
            模型返回的文本内容
        """
        # 延迟导入：避免模块级别的循环依赖
        # prompt_templates 内部也引用了 rag 包的模块，运行时加载最安全
        from src.rag.prompt_templates import get_registry  # noqa: PLC0415

        registry = get_registry()  # 获取全局模板注册中心
        messages = registry.render(template_name, **template_vars)  # 渲染模板 → 消息列表

        # 把渲染好的消息 + 模型配置传给 chat() 完成实际调用
        return self.chat(
            messages=messages,
            model_alias=model_alias,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def compare_models(
        self,
        messages: list[dict[str, str]] | None = None,  # 直接传消息（与 template_name 二选一）
        template_name: str | None = None,  # 使用模板（与 messages 二选一）
        aliases: list[str] | None = None,  # 要对比的别名列表，默认对比全部
        **template_vars: str,  # 模板变量（仅 template_name 模式）
    ) -> list[dict[str, object]]:
        """使用多个模型处理同一请求，对比结果差异（E3 新增: 模型效果对比）。

        面试价值: 面试时可以提到"我做了 fast 和 pro 模型的对比评估，
        fast 在简单任务上表现相当但更快更便宜，pro 在复杂推理上明显更好"。

        用法:
            client = LLMClient()
            results = client.compare_models(
                template_name="rag_qa",
                aliases=["fast", "pro"],
                query="什么是向量数据库？",
                context="向量数据库是...",
            )
            for r in results:
                print(f"[{r['alias']}] {r['model']}: {r['answer'][:100]}")

        Args:
            messages: 直接传入的消息列表（与 template_name 互斥）
            template_name: 模板名称（与 messages 互斥）
            aliases: 要对比的模型别名列表，默认对比配置中所有别名
            **template_vars: 模板变量，仅 template_name 模式使用

        Returns:
            列表，每项含 alias/model/answer，失败时含 error 字段

        Raises:
            ValueError: messages 和 template_name 都没提供时
        """
        # --- 参数解析: messages 和 template_name 必须提供其一 ---
        if template_name:
            # 模板模式: 渲染模板得到消息列表
            from src.rag.prompt_templates import get_registry  # noqa: PLC0415

            registry = get_registry()
            messages = registry.render(template_name, **template_vars)

        if not messages:
            raise ValueError("必须提供 messages 或 template_name")

        # --- 确定对比范围: 传了 aliases 就用，没传就用全部已配置别名 ---
        if aliases is None:
            aliases = list(self.model_aliases) or ["fast"]

        # --- 逐个调用模型，收集结果 ---
        results: list[dict[str, object]] = []
        for alias in aliases:
            # 解析别名 → 实际模型名
            model = self._resolve_model(model=None, model_alias=alias)
            print(f"🔄 调用模型: [{alias}] -> {model}")  # 进度提示

            try:
                answer = self.chat(messages=messages, model_alias=alias)
                results.append({"alias": alias, "model": model, "answer": answer})
            except Exception as exc:
                # 单个模型失败不中断整体流程，记录错误继续下一个
                results.append(
                    {"alias": alias, "model": model, "answer": "", "error": str(exc)}
                )

        return results
