"""
Embedding 演示
使用 sentence-transformers 生成向量 并计算相似度
"""

import glob
import os
from pathlib import PurePath

import numpy as np
from sentence_transformers import SentenceTransformer

# ✅ 设置 Hugging Face 缓存目录（指向你的本地模型目录）
os.environ["HF_HOME"] = "D:/AI_Models/huggingface"
# 如果需要镜像（下载时用）
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"

DEFAULT_EMBEDDING_MODEL = (
    r"D:/AI_Models/huggingface/hub/"
    r"models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/"
    r"snapshots/e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
)


def get_model_display_name(model_name: str) -> str:
    """从模型路径中提取适合终端显示的目录名。"""
    for part in PurePath(model_name).parts:
        if part.startswith("models--"):
            return part
    return PurePath(model_name).name or model_name


class EmbeddingClient:
    """Embedding 客户端"""

    def __init__(
        self,
        # model_name: str = "D:/AI_Models/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ):
        print(f"{'=' * 20} EmbeddingClient {'=' * 20}")

        """
        初始化 Embedding 模型

        Args:
            model_name: 模型名称，默认使用 BAAI/bge-m3
            可选：'all-MiniLM-L6-v2' (英文), 'paraphrase-multilingual-MiniLM-L12-v2' (多语言)
        """
        # base_path = "D:/AI_Models/huggingface/hub/models--BAAI--bge-m3/snapshots/"
        base_path = "D:/AI_Models/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/"

        # 自动查找第一个快照目录
        snapshot_dirs = glob.glob(os.path.join(base_path, "*"))
        if not snapshot_dirs:
            raise FileNotFoundError(f"在 {base_path} 下未找到模型快照。")

        self.model_path = snapshot_dirs[0]  # 使用第一个找到的快照
        if self.model_path == model_name:
            r"""
            D:/AI_Models/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots\e8f8c211226b894fcb81acc59f3b34ba3efd5f42
            D:/AI_Models/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/e8f8c211226b894fcb81acc59f3b34ba3efd5f42
            """

        self.model = SentenceTransformer(model_name)
        print(f"📂 找到本地模型: {get_model_display_name(model_name)}")
        self.dimension = self.model.get_embedding_dimension()
        print(f"✅ Embedding模型加载成功！向量维度: {self.dimension}")

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
