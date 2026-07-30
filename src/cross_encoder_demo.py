"""
Cross-Encoder 演示
使用 Hugging Face 交叉编码器进行精排
"""

from src.reranker import CrossEncoderReranker


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
    candidates = [(doc, 0.0) for doc in documents]
    results = reranker.rerank(query, candidates, top_k=3)

    print("🔍 Cross-Encoder 重排结果:")
    for i, (doc, score) in enumerate(results):
        label = get_similarity_label(score)
        print(f"  {i + 1}. (分数: {score:.4f}) [{label}]")
        print(f"     {doc}")
        print()
