"""
RAG 完整流程
串流：检索 → 重排 → 上下文拼接 → LLM生成
"""

from src.llm_client import LLMClient
from src.reranker import Reranker
from src.retriever import Retriever


class RAGPipeline:
    """RAG 完整流程"""

    def __init__(self):
        print("=" * 20 + " RAG Pipeline " + "=" * 20)

        self.retriever = Retriever()
        self.reranker = Reranker()
        self.llm = LLMClient()

    def index_document(self, text: str) -> None:
        """检索文档"""
        self.retriever.index_document(text)

    def _build_source_prompt(
        self,
        question: str,
        chunk: str,
        source_index: int,
        source_count: int,
    ) -> str:
        """为单条参考片段构造回答提示词。"""
        return f"""
            你是一个智能助手，请只根据这一条参考文档回答用户问题。

            参考文档[{source_index}/{source_count}]：
            {chunk}

            用户问题：{question}

            要求：
            - 只输出这一条参考文档对应的一个结论
            - 只输出一行
            - 不要写多个条目，不要扩写到其他参考文档
            - 回答要简洁、准确
            """

    @staticmethod
    def _normalize_answer_line(answer: str) -> str:
        """把模型输出压缩成单行答案，避免出现额外分行。"""
        for line in answer.splitlines():
            cleaned = line.strip()
            if cleaned:
                return cleaned
        return ""

    def _generate_answer_from_sources(
        self,
        question: str,
        sources: list[dict[str, object]],
    ) -> str:
        """逐条参考文档生成答案，确保输出条数和参考片段数一致。"""
        answers: list[str] = []
        source_count = len(sources)

        for index, source in enumerate(sources, start=1):
            chunk = str(source.get("content", ""))
            prompt = self._build_source_prompt(question, chunk, index, source_count)
            raw_answer = self.llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=200,
            )
            normalized = self._normalize_answer_line(raw_answer)
            if normalized:
                answers.append(f"{index}. {normalized}")

        if not answers:
            return "知识库中国未找到相关信息"

        return "\n".join(answers)

    def query(
        self,
        question: str,
        top_k_retrieve: int = 5,
        top_k_rerank: int = 3,
    ) -> dict:
        """
        执行RAG问答

        Args:
            question: 用户问题
            top_k_retrieve: 找回数量
            top_k_rerank: 重排返回数量

        Retrun:
            {
                "question": 问题
                "context":拼接上下文
                "answer": 生成的回答
                "sources": 来源文档列表
            }
        """

        # Step 1: 检索（召回）
        candidates = self.retriever.retrieve(question, top_k=top_k_retrieve)

        if not candidates:
            return {
                "question": question,
                "context": "",
                "answer": "知识库中国未找到相关信息",
                "sources": [],
            }

        # Step 2: 重排（精排）
        reranked = self.reranker.rerank(question, candidates, top_k=top_k_rerank)

        # Step 3: 拼接上下文
        context_parts = []
        sources = []
        for i, (chunk, score) in enumerate(reranked):
            context_parts.append(f"文档[{i + 1}]\n{chunk}")
            sources.append({"id": i + 1, "content": chunk, "score": score})

        context = "\n\n".join(context_parts)

        # Step 4: 逐条生成答案，保证输出条数与参考片段数一致
        answer = self._generate_answer_from_sources(question, sources)

        return {
            "question": question,
            "context": context,
            "answer": answer,
            "sources": sources,
        }


if __name__ == "__main__":
    # 测试文档
    test_doc = """
    机器学习是人工智能的一个分支，它让计算机能够从数据中学习。
    传统的编程方式需要程序员明确写出规则，而机器学习则通过算法自动发现数据中的模式。

    深度学习是机器学习的一个子集，它使用多层神经网络来学习数据的表示。
    深度学习的核心是神经网络，它由多个层组成，每一层都从前一层学习特征。

    大语言模型是基于深度学习的自然语言处理模型，它们能够理解和生成人类语言。
    GPT、Claude、DeepSeek 等都是大语言模型的代表。
    这些模型通过在海量文本数据上训练，获得了强大的语言理解和生成能力。

    RAG（检索增强生成）是一种结合检索和生成的技术。
    它先从知识库中检索相关信息，然后让大语言模型基于这些信息生成回答。
    RAG 能够显著提高回答的准确性和可解释性。
    """

    rag = RAGPipeline()
    rag.index_document(test_doc)

    questions = [
        "什么是机器学习？",
        "机器学习和深度学习是什么关系？",
        "什么是 RAG？",
        "大语言模型有哪些代表？",
    ]

    print("=" * 60)
    print("🧠 RAG 问答系统")
    print("=" * 60)

    for q in questions:
        print(f"\n📌 问题: {q}")
        result = rag.query(q)

        print(f"\n📖 回答:\n{result['answer']}")
        print(f"\n📚 来源文档数: {len(result['sources'])}")
        print("-" * 40)
