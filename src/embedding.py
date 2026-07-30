"""兼容层：保留旧的 src.embedding 导入路径。"""

import numpy as np

from src.rag.embedding import EmbeddingClient, get_similarity_label

if __name__ == "__main__":
    # 初始化
    client = EmbeddingClient()

    # ============ 示例 1：相似度计算 ============
    print("=" * 60)
    print("📌 示例 1：相似度计算")
    print("=" * 60)

    texts = [
        "猫是可爱的动物",
        "猫咪很可爱",
        "汽车是一种交通工具",
        "今天天气很好",
    ]

    # 生成向量
    vectors = client.encode(texts)

    # 计算相似度
    score = client.cosine_similarity(vectors[0], vectors[1])
    label = get_similarity_label(score)
    print(f"文本1: {texts[0]}")
    print(f"文本2: {texts[1]}")
    print(f"相似度: {score:.4f} ({label})")

    score = client.cosine_similarity(vectors[0], vectors[2])
    label = get_similarity_label(score)
    print(f"\n文本1: {texts[0]}")
    print(f"文本2: {texts[2]}")
    print(f"相似度: {score:.4f} ({label})")

    score = client.cosine_similarity(vectors[0], vectors[3])
    label = get_similarity_label(score)
    print(f"\n文本1: {texts[0]}")
    print(f"文本2: {texts[3]}")
    print(f"相似度: {score:.4f} ({label})")

    # ============ 示例 2：语义搜索 ============
    print()
    print("=" * 60)
    print("📌 示例 2：语义搜索")
    print("=" * 60)

    documents = [
        "Python 是一种编程语言",
        "RAG 是检索增强生成技术",
        "猫喜欢抓老鼠",
        "向量数据库用于存储和检索向量",
        "今天下雨了",
    ]

    query = "什么是 RAG?"
    print(f"查询：{query}")

    # 生成查询向量和文档向量
    query_vec = client.encode([query])[0]
    doc_vecs = client.encode(documents)

    # 计算向量相似度并排序
    similarities = [
        client.cosine_similarity(query_vec, doc_vec) for doc_vec in doc_vecs
    ]
    sorted_indices = np.argsort(similarities)[::-1]

    print("\n🔍 搜索结果（按相关度排序）:")
    for i, idx in enumerate(sorted_indices):
        label = get_similarity_label(similarities[idx])
        print(f"  {i + 1}. {documents[idx]} (相似度: {similarities[idx]:.4f}({label}))")
