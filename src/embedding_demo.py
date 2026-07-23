"""
Embedding 演示
使用 sentence-transformers 生成向量 并计算相似度
"""

import os

import numpy as np
from sentence_transformers import SentenceTransformer

# ✅ 设置 Hugging Face 缓存目录（指向你的本地模型目录）
# os.environ["HF_HOME"] = "D:/AI_Models/huggingface"
# 如果需要镜像（下载时用）
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"


class EmbeddingClient:
    """Embedding 客户端"""

    def __init__(
        self,
        model_name: str = "D:/AI_Models/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
    ):
        """
        初始化 Embedding 模型

        Args:
            model_name: 模型名称， 默认是多语言轻量级模型
            可选：'all-MiniLM-L6-v2' (英文), 'paraphrase-multilingual-MiniLM-L12-v2' (多语言)
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        """将文本转换为向量"""
        return self.model.encode(texts, normalize_embeddings=True)

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两个向量的余弦相似度"""
        return np.dot(vec1, vec2)


# 在类外部定义（顶级函数）
def get_similarity_label(score: float) -> str:
    """根据余弦相似度值返回语义关系标签"""
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
    # 初始化
    client = EmbeddingClient()
    print(f"✅ 模型加载成功！向量维度: {client.dimension}")
    print()

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
        print(f"  {i+1}. {documents[idx]} (相似度: {similarities[idx]:.4f}({label}))")
