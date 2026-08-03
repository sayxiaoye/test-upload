"""
检索召回模块
从切分后的 chunks 中检索最相关的片段
"""

import json
from pathlib import Path

import numpy as np  # 用于数组排序和数值计算

from src.rag.chunker import TextChunker  # 文本切分器，把长文档切割成小块
from src.rag.embedding import EmbeddingClient  # 向量化客户端，把文本变成向量


class Retriever:
    """检索器"""

    def __init__(self, chunker: TextChunker | None = None):
        print(f"\033[1;31m{'=' * 20} 【Retriever】 {'=' * 20}\033[0m")

        self.chunker = chunker or TextChunker(chunk_size=200)  # 文本切分器
        self.embedding_client = EmbeddingClient()
        self.chunks: list[str] = []  # 存储切分后的文本块列表
        self.chunk_embeddings: np.ndarray | None = (
            None  # 存储所有chunk的向量（二维数组）
        )
        self.index_records: list[dict] = []  # 从 JSONL 加载的原始记录

    def index_document(self, text: str) -> None:
        print("\033[1;31m【Retriever】索引文档：切分并生成向量\033[0m")

        self.chunks = self.chunker.semantic_chunk(text)
        if self.chunks:
            self.chunk_embeddings = self.embedding_client.encode(self.chunks)

    def load_from_jsonl(self, jsonl_path: str) -> None:
        print("\033[1;31m【Retriever】从知识库 JSONL 索引文件加载 chunks 并向量化。\033[0m")
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"知识库索引文件不存在: {path}")

        records: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))

        if not records:
            raise ValueError(f"知识库索引文件为空: {path}")

        self.index_records = records
        self.chunks = [
            str(r.get("chunk_text", "")) for r in records
        ]
        self.chunk_embeddings = self.embedding_client.encode(self.chunks)
        print(f"\033[1;31m已从 {path.as_posix()} 加载 {len(self.chunks)} 个 chunk\033[0m")

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        print("\033[1;31m【Retriever】问题也变成向量，检索最相关的 {top_k} 个 chunk\033[0m")

        # 如果还没有索引任何文档，直接返回空列表。
        if not self.chunks or self.chunk_embeddings is None:
            return []
        # 把用户的查询问题也变成向量。
        query_vec = self.embedding_client.encode([query])[0]
        # 把查询向量和每个 chunk 向量做余弦相似度计算。
        print("\033[1;34m【Embedding cosine_similarity】计算两个向量的余弦相似度\033[0m")

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
