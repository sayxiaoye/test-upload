"""
BM25 关键词检索模块（从零实现 Okapi BM25 算法）

BM25 (Best Matching 25) 是经典的稀疏/关键词检索算法，不依赖向量化模型。
它基于词频（TF）和逆文档频率（IDF）对文档进行打分，擅长精确关键词匹配。

与 Embedding 检索的互补关系：
- Embedding：擅长语义相近的表达（"价格" ≈ "多少钱"）
- BM25：擅长精确术语命中（"BM25"、"PostgreSQL"、"第3条"）

组合使用可实现「混合检索」(hybrid retrieval)，同时兼顾语义和关键词。

公式回顾：
    BM25(q, d) = Σ IDF(qi) * (f(qi,d) * (k1+1)) / (f(qi,d) + k1*(1-b + b*|d|/avgdl))

参考：Robertson & Zaragoza (2009) "The Probabilistic Relevance Framework: BM25 and Beyond"
"""

import math
import re
from collections import defaultdict

# ============================================================================
# 中英文混合分词
# ============================================================================

# 中文 Unicode 范围：基本汉字区 + 扩展区
_CJK_RANGES = [
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
]


def _is_cjk(char: str) -> bool:
    """判断一个字符是否在中文 Unicode 范围内。"""
    cp = ord(char)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def tokenize(text: str) -> list[str]:
    """
    中英文混合分词。

    策略（简单但实用，不依赖 jieba 等外部库）：
    - 英文/数字：按空白和标点分割，转小写，过滤空串
    - 中文：按单字符切分（unigram）——因为 BM25 本身对中文
      分词精度的要求在工程上也经常用 unigram/双字组合替代

    示例：
        >>> tokenize("RAG结合BM25效果更好")
        ['rag', '结', '合', 'bm25', '效', '果', '更', '好']

    Args:
        text: 原始文本

    Returns:
        分词后的 token 列表
    """
    tokens: list[str] = []

    # 第一步：用正则将英文/数字块与中文/标点分开
    # 匹配连续的字母/数字（英文 token）或者单个非空白字符
    parts = re.findall(r"[a-zA-Z0-9]+|[^a-zA-Z0-9\s]", text)

    for part in parts:
        if re.match(r"[a-zA-Z0-9]+", part):
            # 英文/数字：转小写
            tokens.append(part.lower())
        elif _is_cjk(part):
            # 中文字符：单字切分
            tokens.append(part)
        # 其他字符（标点等）直接跳过

    return tokens


# ============================================================================
# BM25 检索器
# ============================================================================

class BM25Retriever:
    """
    从零实现的 Okapi BM25 检索器。

    用法：
        >>> bm25 = BM25Retriever()
        >>> bm25.build_index(["文档一的内容", "第二个文档", "还有更多文档"])
        >>> results = bm25.search("文档 内容", top_k=2)
        >>> for idx, score in results:
        ...     print(f"chunk[{idx}] score={score:.4f}")
        chunk[0] score=1.2345
        chunk[2] score=0.8765

    参数：
        k1: 词频饱和度控制（默认 1.5，值越大词频影响越大）
        b:  文档长度归一化控制（默认 0.75，0=不归一化，1=完全归一化）
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        # 索引数据
        self._corpus: list[list[str]] = []  # 每条是 token 列表
        self._doc_lens: list[int] = []  # 每条文档的 token 数量
        self._avgdl: float = 0.0  # 平均文档长度

        # 倒排索引：term → {doc_idx: term_frequency}
        self._inverted_index: dict[str, dict[int, int]] = defaultdict(dict)

        # 文档频率：term → 包含该 term 的文档数
        self._df: dict[str, int] = defaultdict(int)

        # 文档总数（缓存，避免重复 len(self._corpus)）
        self._N: int = 0

        # 是否已构建索引
        self._built: bool = False

    # ------------------------------------------------------------------
    # 索引构建
    # ------------------------------------------------------------------

    def build_index(self, corpus: list[str]) -> None:
        """
        构建 BM25 索引。

        对每个文档分词 → 统计词频 → 构建倒排索引 → 计算 IDF。

        Args:
            corpus: 文档列表，每个元素是一段文本
        """
        self.reset()

        if not corpus:
            return

        self._corpus = [tokenize(doc) for doc in corpus]
        self._doc_lens = [len(tokens) for tokens in self._corpus]
        self._N = len(self._corpus)
        self._avgdl = sum(self._doc_lens) / self._N if self._N > 0 else 0.0

        # 构建倒排索引 + 文档频率
        for doc_idx, tokens in enumerate(self._corpus):
            # 统计当前文档的词频
            tf_map: dict[str, int] = defaultdict(int)
            for token in tokens:
                tf_map[token] += 1

            # 写入倒排索引
            for token, freq in tf_map.items():
                self._inverted_index[token][doc_idx] = freq
                self._df[token] += 1  # 文档频率 +1

        self._built = True

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """
        对查询进行 BM25 检索。

        步骤：
        1. 对查询分词
        2. 对每个查询 term，从倒排索引中取出命中的文档及词频
        3. 对每个命中文档累加 BM25 得分
        4. 按得分降序返回 top_k

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            [(chunk_index, bm25_score), ...]，按分数降序排列
        """
        if not self._built or self._N == 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # doc_idx → 累加分数
        scores: dict[int, float] = defaultdict(float)

        for token in query_tokens:
            # 倒排索引命中
            postings = self._inverted_index.get(token, {})
            if not postings:
                continue

            # 计算该 term 的 IDF
            df_t = len(postings)  # 也可以用 self._df[token]
            idf = math.log((self._N - df_t + 0.5) / (df_t + 0.5) + 1.0)

            # 对每个包含该 term 的文档累加分数
            for doc_idx, tf in postings.items():
                doc_len = self._doc_lens[doc_idx]
                # BM25 term-doc score
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (
                    1.0 - self.b + self.b * (doc_len / self._avgdl)
                )
                scores[doc_idx] += idf * (numerator / denominator)

        # 按分数降序排列
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]

    # ------------------------------------------------------------------
    # 分数归一化（用于混合检索）
    # ------------------------------------------------------------------

    def normalize_scores(
        self, scored_docs: list[tuple[int, float]]
    ) -> list[tuple[int, float]]:
        """
        对 BM25 原始分数做 min-max 归一化到 [0, 1]。

        由于 BM25 分数范围不定，在混合检索时需要归一化，才能和
        embedding 的 [0,1] 余弦相似度做加权求和。

        如果所有分数相同（min == max），则全部置为 1.0。

        Args:
            scored_docs: [(doc_idx, raw_score), ...]

        Returns:
            [(doc_idx, normalized_score), ...]
        """
        if not scored_docs:
            return []

        scores = [s for _, s in scored_docs]
        min_s, max_s = min(scores), max(scores)

        if max_s == min_s:
            return [(idx, 1.0) for idx, _ in scored_docs]

        return [
            (idx, (score - min_s) / (max_s - min_s))
            for idx, score in scored_docs
        ]

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """清空索引，回到初始状态。"""
        self._corpus.clear()
        self._doc_lens.clear()
        self._avgdl = 0.0
        self._inverted_index.clear()
        self._df.clear()
        self._N = 0
        self._built = False

    @property
    def is_built(self) -> bool:
        """索引是否已构建。"""
        return self._built

    @property
    def doc_count(self) -> int:
        """已索引的文档数。"""
        return self._N
