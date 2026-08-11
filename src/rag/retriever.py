"""
检索召回模块（E9 增强：集成 BM25 混合检索）
从切分后的 chunks 中检索最相关的片段。

支持的检索模式：
- embedding: 纯语义向量检索（余弦相似度）
- bm25:     纯关键词检索（Okapi BM25 算法）
- hybrid:    混合检索（embedding + BM25 加权融合）
"""

import json
from pathlib import Path

import numpy as np  # 用于数组排序和数值计算

from src.rag.bm25 import BM25Retriever  # E9 新增: BM25 关键词检索
from src.rag.chunker import TextChunker  # 文本切分器，把长文档切割成小块
from src.rag.embedding import EmbeddingClient  # 向量化客户端，把文本变成向量


class Retriever:
    """检索器（支持 embedding / BM25 / hybrid 三种模式）"""

    def __init__(
        self,
        chunker: TextChunker | None = None,
        retrieval_mode: str = "embedding",
        bm25_weight: float = 0.3,
    ):
        print(f"\033[1;31m{'=' * 20} 【Retriever】 {'=' * 20}\033[0m")

        self.chunker = chunker or TextChunker(chunk_size=200)  # 文本切分器
        self.embedding_client = EmbeddingClient()
        self.bm25 = BM25Retriever()  # E9 新增: BM25 倒排索引
        self.chunks: list[str] = []  # 存储切分后的文本块列表
        self.chunk_embeddings: np.ndarray | None = (
            None  # 存储所有chunk的向量（二维数组）
        )
        self.index_records: list[dict] = []  # 从 JSONL 加载的原始记录

        # E9 新增: 检索模式配置
        self.retrieval_mode = retrieval_mode  # embedding | bm25 | hybrid
        self.bm25_weight = bm25_weight  # hybrid 模式中 BM25 的权重 [0, 1]
        print(f"\033[1;31m 检索模式: {self.retrieval_mode}\033[0m")
        if self.retrieval_mode == "hybrid":
            print(f"\033[1;31m BM25 权重: {self.bm25_weight}\033[0m")

    def index_document(self, text: str) -> None:
        print("\033[1;31m【Retriever】索引文档：切分并生成向量\033[0m")

        self.chunks = self.chunker.semantic_chunk(text)
        if self.chunks:
            self.chunk_embeddings = self.embedding_client.encode(self.chunks)
            # E9 新增: 同时构建 BM25 倒排索引
            self.bm25.build_index(self.chunks)
            print(
                "\033[1;31m【Retriever】BM25 索引已构建: "
                f"{self.bm25.doc_count} 个文档\033[0m"
            )

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
        # E9 新增: 同时构建 BM25 倒排索引
        self.bm25.build_index(self.chunks)
        print(
            f"\033[1;31m已从 {path.as_posix()} 加载 {len(self.chunks)} 个 chunk\033[0m"
        )
        print(
            f"\033[1;31mBM25 索引已构建: {self.bm25.doc_count} 个文档\033[0m"
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        mode: str | None = None,
    ) -> list[tuple[str, float]]:
        """
        检索与查询最相关的 top_k 个 chunk。

        Args:
            query: 查询文本
            top_k: 返回数量
            mode: 检索模式（可选，默认使用 self.retrieval_mode）
                  - "embedding": 纯语义向量检索
                  - "bm25": 纯关键词检索
                  - "hybrid": 混合检索（embedding + BM25 加权融合）

        Returns:
            [(chunk_text, score), ...]，按分数降序排列
        """
        actual_mode = mode or self.retrieval_mode
        print(
            f"\033[1;31m【Retriever】检索模式={actual_mode}，"
            f"查询并返回最相关的 {top_k} 个 chunk\033[0m"
        )

        # 如果还没有索引任何文档，直接返回空列表。
        if not self.chunks or self.chunk_embeddings is None:
            return []

        # ================================================================
        # 模式一：纯 Embedding 检索（现有逻辑）
        # ================================================================
        if actual_mode == "embedding":
            return self._retrieve_embedding(query, top_k)

        # ================================================================
        # 模式二：纯 BM25 检索
        # ================================================================
        if actual_mode == "bm25":
            return self._retrieve_bm25(query, top_k)

        # ================================================================
        # 模式三：混合检索（Embedding + BM25 加权融合）
        # ================================================================
        if actual_mode == "hybrid":
            return self._retrieve_hybrid(query, top_k)

        # 未知模式→回退到 embedding
        print(f"\033[1;31m⚠️ 未知检索模式 '{actual_mode}'，回退到 embedding\033[0m")
        return self._retrieve_embedding(query, top_k)

    # ------------------------------------------------------------------
    # 各检索模式的内部实现
    # ------------------------------------------------------------------

    def _retrieve_embedding(
        self, query: str, top_k: int
    ) -> list[tuple[str, float]]:
        """纯向量语义检索（现有逻辑）。"""
        if self.chunk_embeddings is None:
            return []
        query_vec = self.embedding_client.encode([query])[0]
        print(
            "\033[1;34m【Embedding cosine_similarity】"
            "计算两个向量的余弦相似度\033[0m"
        )

        similarities = [
            self.embedding_client.cosine_similarity(query_vec, emb)
            for emb in self.chunk_embeddings
        ]

        sorted_indices = np.argsort(similarities)[::-1][:top_k]
        return [(self.chunks[i], similarities[i]) for i in sorted_indices]

    def _retrieve_bm25(
        self, query: str, top_k: int
    ) -> list[tuple[str, float]]:
        """纯 BM25 关键词检索。"""
        if not self.bm25.is_built:
            print("\033[1;31m⚠️ BM25 索引未构建，回退到 embedding\033[0m")
            return self._retrieve_embedding(query, top_k)

        bm25_results = self.bm25.search(query, top_k=top_k)
        return [(self.chunks[idx], score) for idx, score in bm25_results]

    def _retrieve_hybrid(
        self, query: str, top_k: int
    ) -> list[tuple[str, float]]:
        """
        混合检索：Embedding + BM25 加权融合。

        策略：
        1. 分别用 embedding 和 BM25 对所有文档打分
        2. BM25 分数做 min-max 归一化到 [0,1]（embedding 余弦相似度已在 [0,1]）
        3. 加权求和：final = (1-α) × emb + α × bm25
        4. 按最终分数降序返回 top_k
        """
        alpha = self.bm25_weight

        # --- Embedding 分数（全体 chunks） ---
        if self.chunk_embeddings is None:
            return []
        query_vec = self.embedding_client.encode([query])[0]
        emb_scores = np.array([
            self.embedding_client.cosine_similarity(query_vec, emb)
            for emb in self.chunk_embeddings
        ])

        # --- BM25 分数（全体 chunks） ---
        bm25_raw = self.bm25.search(query, top_k=len(self.chunks))
        # 转换为数组，保持与 chunks 索引对齐
        bm25_array = np.zeros(len(self.chunks))
        for idx, score in bm25_raw:
            bm25_array[idx] = score

        # --- BM25 min-max 归一化 ---
        bm25_norm = np.zeros_like(bm25_array)
        if bm25_array.max() > bm25_array.min():
            bm25_norm = (bm25_array - bm25_array.min()) / (
                bm25_array.max() - bm25_array.min()
            )
        elif bm25_array.max() > 0:
            # 所有非零分数相同
            bm25_norm[bm25_array > 0] = 1.0

        # --- 加权融合 ---
        final_scores = (1.0 - alpha) * emb_scores + alpha * bm25_norm

        # --- 按最终分数排序 ---
        sorted_indices = np.argsort(final_scores)[::-1][:top_k]
        return [(self.chunks[i], float(final_scores[i])) for i in sorted_indices]
