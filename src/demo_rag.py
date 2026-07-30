"""E 阶段第 1 天示例：最小 RAG 问答 Demo。

目标：
1. 输入一个问题
2. 调用 RAGPipeline 生成回答
3. 打印回答和来源
"""

from src.rag_pipeline import RAGPipeline


def run_demo(question: str) -> None:
    pipeline = RAGPipeline()

    document = """
    Python 是一种通用编程语言，适合脚本、自动化、数据分析和 AI 应用开发。
    RAG 是检索增强生成，它通过先检索相关文档，再让大模型基于这些文档做回答。
    大模型可以用于聊天、翻译、总结和问答等任务。
    """

    pipeline.index_document(document)
    result = pipeline.query(question)

    print("=" * 60)
    print("📌 问题:")
    print(question)
    print("\n📖 回答:")
    print(result["answer"])
    print("\n📚 来源:")
    for source in result["sources"]:
        print(f"- {source['content']}")
    print("=" * 60)


if __name__ == "__main__":
    run_demo("什么是 RAG？")
