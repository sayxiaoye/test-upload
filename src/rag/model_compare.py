"""
模型对比演示（E3 新增）

对比多个模型（fast / pro）在相同 prompt 下的回答差异。
展示"模板选择 → 模型路由 → 结果对比"的完整流程。

面试话术（当被问"你做了哪些模型层面的优化"时）:
"I built a model comparison pipeline. For simple Q&A, I route to the fast model
to save cost; for complex reasoning, I use the pro model for accuracy.
Switching is just a config alias change — no code modifications needed."
"""

from src.rag.llm_client import LLMClient  # 大模型客户端，支持别名切换和模板集成


def show_alias_system() -> None:
    """展示模型别名系统的配置和路由逻辑（纯本地，不调 API）。

    面试时可以用这里的输出向面试官说明:
    - 别名在哪里配置（config.yaml）
    - 路由优先级是什么（model > alias > default）
    - 怎么切换模型（改一行配置就行）
    """
    client = LLMClient()  # 初始化时自动从 config.yaml 读取别名配置

    print("=" * 60)
    print("模型别名系统说明")
    print("=" * 60)
    print(f"  已配置别名: {client.model_aliases}")  # 来自 config.yaml 的 models.aliases
    print(f"  默认别名:   {client.default_alias}")  # 来自 config.yaml 的 models.default_alias
    print(f"  默认模型:   {client.default_model}")  # 解析后的实际模型名
    print()
    print("  三级路由优先级（_resolve_model 方法）:")
    print("    1. model 参数直接指定（如 model='gpt-4o'）— 最高优先级")
    print("    2. model_alias 别名查找（如 alias='pro' → 'deepseek-v4-pro'）")
    print("    3. 默认别名 → 默认模型（当前: fast → deepseek-v4-flash）")
    print()
    print("  切换模型只需修改 config.yaml 中的一行:")
    print("    models.aliases.fast: \"deepseek-v4-flash\"  # 改这里就行")


def compare_rag_qa() -> None:
    """对比 fast 和 pro 模型在 RAG 问答中的表现。

    面试要点:
    - 同一个问题和上下文，分别用 fast/pro 模型回答
    - 对比内容完整性、准确度和响应长度
    - 说明"不同任务复杂度匹配不同模型"的成本控制策略
    """
    client = LLMClient()

    # --- 测试数据: 一段关于向量数据库的知识文本 ---
    # 用这段文本作为 RAG 的参考上下文
    context = """
    向量数据库是一种专门用于存储和检索向量嵌入的数据库系统。
    它通过近似最近邻（ANN）算法实现高效的相似度搜索。
    常用的向量数据库包括 Milvus、Pinecone、Weaviate 和 Qdrant。
    其中 Milvus 是开源的，支持十亿级向量检索。
    """

    question = "什么是向量数据库？常用的有哪些？"

    print()
    print("=" * 60)
    print("模型对比演示: RAG 问答")
    print("=" * 60)
    print(f"问题: {question}")
    print()

    # --- Fast 模型 ---
    # alias="fast" → 配置映射到 deepseek-v4-flash（速度快、成本低）
    # 适合: 简单问答、摘要提取、日常对话
    print("[fast] 调用中...")
    try:
        fast_answer = client.chat_with_template(
            "rag_qa",  # 使用 prompt_templates 中预置的 RAG 问答模板
            model_alias="fast",  # 别名 → 实际模型名由 _resolve_model() 完成
            query=question,  # 模板变量: 替换 user_template 中的 {query}
            context=context,  # 模板变量: 替换 user_template 中的 {context}
        )
        print(f"[fast] 回答 ({len(fast_answer)} 字符):")
        print(f"  {fast_answer}")
    except Exception as e:
        print(f"[fast] 调用失败: {e}")
        fast_answer = ""
    print()

    # --- Pro 模型 ---
    # alias="pro" → 配置映射到 deepseek-v4-pro（推理能力强、准确性高）
    # 适合: 复杂推理、深度分析、需要精确答案的场景
    print("[pro] 调用中...")
    try:
        pro_answer = client.chat_with_template(
            "rag_qa",
            model_alias="pro",  # 复杂推理用 pro 保证质量
            query=question,
            context=context,
        )
        print(f"[pro] 回答 ({len(pro_answer)} 字符):")
        print(f"  {pro_answer}")
    except Exception as e:
        print(f"[pro] 调用失败: {e}")
        pro_answer = ""
    print()

    # --- 对比分析 ---
    # 面试时可以用这些数据说明"为什么需要多模型切换"
    if fast_answer and pro_answer:
        print("─" * 60)
        print("对比总结（面试话术素材）")
        print("─" * 60)
        print(f"  fast 模型: {len(fast_answer)} 字符 — 响应更快、成本更低")
        print(f"  pro  模型: {len(pro_answer)} 字符 — 推理更深、更完整")
        print()
        print("  实际应用策略（面试时可以说）:")
        print("  - 简单问答/摘要 → 路由到 fast，降低 API 成本约 80%")
        print("  - 复杂推理/评估 → 路由到 pro，保证回答质量")
        print("  - 新模型灰度上线时 → 同时调两个，对比效果再决定切换")


def compare_via_batch_method() -> None:
    """使用 compare_models() 方法批量对比。

    这个方法封装了"同 prompt 多模型对比"的循环逻辑，
    一次调用返回所有模型的回答，适合做自动化评估流水线。
    """
    client = LLMClient()

    print()
    print("=" * 60)
    print("批量对比演示: 文档摘要")
    print("=" * 60)

    # compare_models() 内部循环调用 fast 和 pro，
    # 返回列表，每项包含 alias（别名）、model（实际模型名）、answer（回答）
    results = client.compare_models(
        template_name="document_summary",  # 文档摘要模板，来自 prompt_templates
        aliases=["fast", "pro"],  # 要对比的别名列表
        title="向量数据库选型指南",  # 模板变量 {title}
        document_text=(  # 模板变量 {document_text}
            "向量数据库是 AI 应用的核心基础设施。"
            "选型时主要考虑四个维度：性能方面要关注 QPS 和延迟；"
            "扩展性要看是否支持分布式和十亿级以上数据；"
            "功能方面要看是否支持过滤、混合检索和多模态；"
            "运维方面要看是否提供托管服务、监控和备份。"
        ),
    )

    # 逐个展示结果
    for r in results:
        alias = r.get("alias", "?")  # 模型别名（fast/pro）
        model = r.get("model", "?")  # 解析后的实际模型名
        answer = str(r.get("answer", ""))  # 模型的回答文本
        error = r.get("error")  # 出错时才会有此字段

        if error:
            print(f"\n[{alias}] {model} -> 错误: {error}")
        else:
            print(f"\n[{alias}] {model} ({len(answer)} 字符):")
            print(f"  {answer[:200]}...")


def main() -> None:
    """运行模型对比完整演示。

    演示顺序:
    1. 别名系统说明（概念理解，不调 API）
    2. 确认后进入 RAG 问答对比（实际调用 API）
    3. 确认后进入批量对比（compare_models 方法演示）
    """
    print("=" * 60)
    print("E3-3 模型切换演示")
    print("=" * 60)
    print("本演示展示: 别名系统 → 模板集成 → 模型对比 完整链路")
    print()

    # 第 1 步: 展示别名系统（纯本地输出，不调 API）
    show_alias_system()

    # 第 2 步: RAG 问答对比（会实际调两次 DeepSeek API）
    print()
    choice = input("是否进行 RAG 问答对比？（会调用 API 产生费用，输入 y 继续）: ")
    if choice.strip().lower() == "y":
        compare_rag_qa()

    # 第 3 步: 批量对比演示（会实际调两次 DeepSeek API）
    choice2 = input("\n是否进行批量对比演示？（会调用 API 产生费用，输入 y 继续）: ")
    if choice2.strip().lower() == "y":
        compare_via_batch_method()

    print()
    print("=" * 60)
    print("E3-3 演示结束")
    print("=" * 60)


if __name__ == "__main__":
    main()
