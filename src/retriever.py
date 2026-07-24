"""
检索召回模块
从切分后的 chunks 中检索最相关的片段
"""

import numpy as np  # 用于数组排序和数值计算

from src.chunker import TextChunker  # 文本切分器，把长文档切割成小块
from src.embedding_demo import EmbeddingClient  # 向量化客户端，把文本变成向量


class Retriever:
    """检索器"""

    def __init__(self, chunker: TextChunker | None = None):
        self.chunker = chunker or TextChunker(chunk_size=200)  # 文本切分器
        self.embedding_client = EmbeddingClient()
        self.chunks: list[str] = []  # 存储切分后的文本块列表
        self.chunk_embeddings: np.ndarray | None = (
            None  # 存储所有chunk的向量（二维数组）
        )

    def index_document(self, text: str) -> None:
        """索引文档：切分并生成向量"""
        self.chunks = self.chunker.semantic_chunk(text)
        if self.chunks:
            self.chunk_embeddings = self.embedding_client.encode(self.chunks)

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        """检索最相关的 top_k 个 chunk"""
        # 如果还没有索引任何文档，直接返回空列表。
        if not self.chunks or self.chunk_embeddings is None:
            return []
        # 把用户的查询问题也变成向量。
        query_vec = self.embedding_client.encode([query])[0]
        # 把查询向量和每个 chunk 向量做余弦相似度计算。
        similarities = [
            self.embedding_client.cosine_similarity(query_vec, emb)
            for emb in self.chunk_embeddings
        ]

        # 按相似度排名
        """
        np.argsort(similarities)    按相似度从小到大排序，返回索引
        [::-1]                      反转 → 从大到小（从高到低）
        [:top_k]                    只取前 top_k 个
        [(chunks[i], sim[i]) for i in ...]	返回 (文本块, 相似度) 的列表
        """
        sorted_indices = np.argsort(similarities)[::-1][:top_k]
        return [(self.chunks[i], similarities[i]) for i in sorted_indices]


def get_similarity_label(score: float) -> str:
    if score >= 0.8:
        return "非常相关"
    elif score >= 0.6:
        return "相关"
    elif score >= 0.4:
        return "部分相关"
    else:
        return "弱相关"


if __name__ == "__main__":
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

    retriever = Retriever()  # 创建检索器
    retriever.index_document(test_doc)  # 索引测试文档

    queries = [
        "什么是机器学习？",
        "什么是RAG",
        "GPT 是什么？",
    ]

    print("=" * 60)
    print("🔍 检索演示")
    print("=" * 60)

    for query in queries:
        print(f"\n📌 查询: {query}")
        results = retriever.retrieve(query, top_k=3)
        for i, (chunk, score) in enumerate(results):
            label = get_similarity_label(score)
            print(f"  结果 {i+1} (相似度: {score:.4f}) [{label}]:")
            print(f"    {chunk[:80]}...")
