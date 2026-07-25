"""
Cross-Encoder 演示
使用 Hugging Face 交叉编码器进行精排
"""

import numpy as np
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """交叉编码器 重排器"""

    def __init__(
        self,
        model_name: str = r"D:/AI_Models/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    ):
        """
        初始化交叉编码器

        Args:
            model_name: 模型名称
        """
        self.model = CrossEncoder(model_name)

    def reranker(
        self,
        query: str,
        documents: list[str],
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """
        对文档进行重排

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回数量

        Returns:
            重排后的 [(文档, 分数), ...]
        """
        # 构造 Query-Document 对儿
        pairs = [[query, doc] for doc in documents]

        # 用 Cross-Encoder 预测原始分数（logit）
        scores = self.model.predict(pairs)
        # ✅ 关键：将 logit 转换为概率（0-1 之间）
        scores = 1 / (1 + np.exp(-scores))

        # 组合并排序
        results = list(zip(documents, scores, strict=False))
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]


# 在类外部定义（顶级函数）
def get_similarity_label(score: float) -> str:
    """根据分数返回标签"""
    if score >= 0.8:
        return "非常相近"
    elif score >= 0.6:
        return "相近"
    elif score >= 0.4:
        return "有一定关联"
    elif score >= 0.2:
        return "关联较弱"
    else:
        return "不相关"


if __name__ == "__main__":
    # 测试数据
    query = "什么是机器学习？"

    documents = [
        "机器学习是人工智能的一个分支，让计算机从数据中学习。",
        "深度学习使用多层神经网络来学习数据表示。",
        "今天天气很好，适合出去散步。",
        "RAG 是检索增强生成技术。",
        "猫是一种可爱的动物，喜欢抓老鼠。",
    ]

    print("=" * 60)
    print("📌 Cross-Encoder 重排演示（本地模型）")
    print("=" * 60)
    print(f"/n查询: {query}/n")

    # 用 Cross-Encoder 重排
    reranker = CrossEncoderReranker()
    results = reranker.reranker(query, documents, top_k=3)

    print("🔍 Cross-Encoder 重排结果:")
    for i, (doc, score) in enumerate(results):
        label = get_similarity_label(score)
        print(f"  {i+1}. (分数: {score:.4f}) [{label}]")
        print(f"     {doc}")
        print()
