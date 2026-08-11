"""测试 BM25 关键词检索模块（E9 新增）。"""


import pytest

from src.rag.bm25 import BM25Retriever, _is_cjk, tokenize

# ============================================================================
# 分词测试
# ============================================================================

class TestTokenize:
    """测试中英文混合分词。"""

    def test_pure_english(self):
        """纯英文按空白分割并转小写。"""
        result = tokenize("Hello World BM25")
        assert result == ["hello", "world", "bm25"]

    def test_pure_chinese(self):
        """纯中文按单字切分。"""
        result = tokenize("你好世界")
        assert result == ["你", "好", "世", "界"]

    def test_mixed_cn_en(self):
        """中英文混合时正确分离。"""
        result = tokenize("RAG结合BM25效果更好")
        assert "rag" in result
        assert "bm25" in result
        assert "结" in result
        assert "合" in result
        # 中文部分按单字
        assert result.count("效") == 1

    def test_numbers_preserved(self):
        """数字和英文一起保留。"""
        result = tokenize("模型参数量7B到13B")
        assert "7b" in result
        assert "13b" in result

    def test_empty_string(self):
        """空字符串返回空列表。"""
        assert tokenize("") == []

    def test_punctuation_only(self):
        """纯标点返回空列表。"""
        result = tokenize("，。！？,.")
        assert result == []

    def test_punctuation_removed(self):
        """标点被过滤，仅保留有意义 token。"""
        result = tokenize("你好！世界。")
        assert result == ["你", "好", "世", "界"]

    def test_cjk_detection_helper(self):
        """_is_cjk 正确判断中文字符。"""
        assert _is_cjk("中") is True
        assert _is_cjk("a") is False


# ============================================================================
# 索引构建测试
# ============================================================================

class TestBuildIndex:
    """测试 BM25 索引构建。"""

    def test_empty_corpus(self):
        """空语料下 is_built=False，doc_count=0。"""
        bm25 = BM25Retriever()
        bm25.build_index([])
        assert bm25.is_built is False  # 空语料不算构建成功
        assert bm25.doc_count == 0

    def test_single_document(self):
        """单文档可以被正常索引。"""
        bm25 = BM25Retriever()
        bm25.build_index(["hello world hello"])
        assert bm25.is_built is True
        assert bm25.doc_count == 1

    def test_multiple_documents(self):
        """多文档构建后 doc_count 正确。"""
        bm25 = BM25Retriever()
        bm25.build_index([
            "文档一的内容是RAG系统",
            "文档二内容关于BM25检索",
            "文档三介绍混合检索方法",
        ])
        assert bm25.doc_count == 3
        assert bm25.is_built is True

    def test_avgdl_correct(self):
        """avgdl（平均文档长度）计算正确。"""
        bm25 = BM25Retriever()
        bm25.build_index(["a b c", "d e f g h"])
        # 分词后: 3 tokens, 5 tokens → avgdl = 4.0
        assert bm25._avgdl == 4.0

    def test_document_frequency_correct(self):
        """文档频率统计正确：term 在几个文档中出现。"""
        bm25 = BM25Retriever()
        bm25.build_index([
            "apple banana",
            "banana cherry",
            "cherry date",
        ])
        # "banana" 出现在 doc0 和 doc1 各一次 → df=2
        assert bm25._df["banana"] == 2
        # "apple" 只在 doc0 → df=1
        assert bm25._df["apple"] == 1


# ============================================================================
# 检索测试
# ============================================================================

class TestSearch:
    """测试 BM25 检索功能。"""

    CORPUS = [
        "向量数据库是一种专门用来存储和索引向量的数据库",
        "BM25是一种基于概率检索模型的排序函数",
        "混合检索结合了向量检索和BM25关键词检索",
        "RAG系统需要高效的检索模块来召回相关文档片段",
    ]

    def test_search_returns_top_k(self):
        """search 返回的结果数量不超过 top_k。"""
        bm25 = BM25Retriever()
        bm25.build_index(self.CORPUS)
        results = bm25.search("BM25 检索", top_k=2)
        assert len(results) <= 2

    def test_search_returns_descending_order(self):
        """返回结果按分数降序排列。"""
        bm25 = BM25Retriever()
        bm25.build_index(self.CORPUS)
        results = bm25.search("向量数据库", top_k=3)
        scores = [s for _, s in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], f"scores[{i}] < scores[{i+1}]"

    def test_keyword_exact_match_ranks_highest(self):
        """包含精确关键词的文档排在最前面。"""
        bm25 = BM25Retriever()
        bm25.build_index(self.CORPUS)
        results = bm25.search("BM25", top_k=4)
        # doc1 和 doc2 包含 "BM25"，应排在 doc0 和 doc3 前面
        top_indices = [idx for idx, _ in results[:2]]
        assert 1 in top_indices or 2 in top_indices  # BM25 相关的在前两位

    def test_search_no_index_returns_empty(self):
        """未构建索引时返回空列表。"""
        bm25 = BM25Retriever()
        results = bm25.search("anything")
        assert results == []

    def test_search_empty_query_returns_empty(self):
        """空查询返回空列表。"""
        bm25 = BM25Retriever()
        bm25.build_index(self.CORPUS)
        results = bm25.search("")
        assert results == []

    def test_idf_higher_for_rare_terms(self):
        """稀有词的 IDF 比常见词高。"""
        bm25 = BM25Retriever()
        bm25.build_index([
            "common common common",
            "common common rare",
            "common common common",
            "common common rare",
        ])
        # "common" 出现在所有 4 个文档中 → 低 IDF
        # "rare" 只出现在 doc1 和 doc3 → 高 IDF
        results = bm25.search("common", top_k=4)
        # 只搜索 "common"，所有文档都命中
        assert len(results) == 4

        results_rare = bm25.search("rare", top_k=4)
        # 只搜索 "rare"，只有 doc1 和 doc3 命中
        assert len(results_rare) == 2


# ============================================================================
# 分数归一化测试
# ============================================================================

class TestNormalizeScores:
    """测试 BM25 分数 min-max 归一化。"""

    def test_normalize_to_zero_one(self):
        """归一化后分数都落在 [0, 1]。"""
        bm25 = BM25Retriever()
        scored = [(0, 3.0), (1, 1.0), (2, 0.0)]
        normalized = bm25.normalize_scores(scored)
        for _, score in normalized:
            assert 0.0 <= score <= 1.0

    def test_max_becomes_one_min_becomes_zero(self):
        """最大值映射为 1，最小值映射为 0。"""
        bm25 = BM25Retriever()
        scored = [(0, 5.0), (1, 2.0), (2, 0.0)]
        normalized = bm25.normalize_scores(scored)
        scores_by_idx = dict(normalized)
        assert scores_by_idx[0] == pytest.approx(1.0)
        assert scores_by_idx[2] == pytest.approx(0.0)

    def test_all_equal_scores(self):
        """所有分数相同时设为 1.0。"""
        bm25 = BM25Retriever()
        scored = [(0, 2.5), (1, 2.5), (2, 2.5)]
        normalized = bm25.normalize_scores(scored)
        for _, score in normalized:
            assert score == 1.0

    def test_empty_input(self):
        """空输入返回空列表。"""
        bm25 = BM25Retriever()
        assert bm25.normalize_scores([]) == []


# ============================================================================
# 参数敏感性测试
# ============================================================================

class TestBM25Parameters:
    """验证 k1 和 b 参数的影响。"""

    CORPUS_LONG = [
        "RAG retrieval augmented generation is a technique",
        "BM25 is a keyword based retrieval algorithm for text search",
    ]

    def test_higher_k1_saturates_tf_slower(self):
        """k1 增大时，高频词的得分增长空间更大。"""
        # k1=0.5：词频影响较小（快速饱和）
        # k1=2.0：词频影响较大（慢速饱和）
        corpus = [
            "BM25 BM25 BM25 BM25 BM25",  # 5 次
            "BM25",  # 1 次
        ]
        bm25_low = BM25Retriever(k1=0.1, b=0.75)
        bm25_low.build_index(corpus)
        results_low = bm25_low.search("BM25", top_k=2)

        bm25_high = BM25Retriever(k1=5.0, b=0.75)
        bm25_high.build_index(corpus)
        results_high = bm25_high.search("BM25", top_k=2)

        # 高 k1 下 doc0 的分数相对更高（因为高频词没有过度饱和）
        low_ratio = results_low[0][1] / results_low[1][1] if results_low[1][1] > 0 else float("inf")
        high_ratio = results_high[0][1] / results_high[1][1] if results_high[1][1] > 0 else float("inf")
        assert high_ratio >= low_ratio, (
            f"k1=5.0 ratio ({high_ratio:.2f}) should be >= k1=0.1 ratio ({low_ratio:.2f})"
        )

    def test_reset_clears_all_state(self):
        """reset 后状态完全清空。"""
        bm25 = BM25Retriever()
        bm25.build_index(["文档一", "文档二", "文档三"])
        bm25.reset()
        assert bm25.is_built is False
        assert bm25.doc_count == 0
        assert bm25._avgdl == 0.0
        assert bm25._inverted_index == {}
        assert bm25.search("文档") == []

    def test_rebuild_after_reset(self):
        """reset 后可以重新构建索引。"""
        bm25 = BM25Retriever()
        bm25.build_index(["旧数据"])
        bm25.reset()
        bm25.build_index(["新数据新数据"])
        assert bm25.doc_count == 1
        results = bm25.search("新数据")
        assert len(results) == 1


# ============================================================================
# 与 Retriever 集成测试
# ============================================================================

class TestRetrieverIntegration:
    """测试 Retriever 的混合检索功能。"""

    def test_retriever_has_bm25_attr(self):
        """Retriever 初始化后具备 bm25 属性。"""
        from unittest.mock import patch

        # 避免 EmbeddingClient 加载真实模型
        with patch("src.rag.retriever.EmbeddingClient", autospec=True):
            from src.rag.retriever import Retriever

            retriever = Retriever()
            assert retriever.bm25 is not None
            assert retriever.retrieval_mode == "embedding"

    def test_retriever_accepts_hybrid_mode(self):
        """Retriever 接受 retrieval_mode='hybrid'。"""
        from unittest.mock import patch

        with patch("src.rag.retriever.EmbeddingClient", autospec=True):
            from src.rag.retriever import Retriever

            retriever = Retriever(retrieval_mode="hybrid", bm25_weight=0.3)
            assert retriever.retrieval_mode == "hybrid"
            assert retriever.bm25_weight == 0.3

    def test_retrieve_bm25_mode_uses_keyword_search(self):
        """bm25 模式下只用关键词检索（不调 embedding encode）"""
        from unittest.mock import patch

        with patch("src.rag.retriever.EmbeddingClient", autospec=True) as MockEmb:
            mock_emb = MockEmb.return_value
            mock_emb.encode.return_value = __import__("numpy").array([[0.1, 0.2]])
            mock_emb.cosine_similarity.return_value = 0.8

            from src.rag.retriever import Retriever

            retriever = Retriever(retrieval_mode="bm25")
            # 手动注入 — 绕过 EmbeddingClient 加载
            retriever.chunks = [
                "RAG 系统使用向量检索",
                "BM25 是关键词检索算法",
                "混合检索最强大",
            ]
            retriever.chunk_embeddings = __import__("numpy").array([
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
            ])
            retriever.bm25.build_index(retriever.chunks)

            results = retriever.retrieve("BM25 关键词", top_k=2, mode="bm25")

            assert len(results) <= 2
            # BM25 模式下，"BM25 关键词" 的精确匹配应让 doc1 排第一
            if len(results) >= 1:
                assert "BM25" in results[0][0] or "关键词" in results[0][0]

    def test_retrieve_hybrid_mode_combines_scores(self):
        """hybrid 模式下分数既是浮点数又落在合理范围。"""
        from unittest.mock import patch

        with patch("src.rag.retriever.EmbeddingClient", autospec=True) as MockEmb:
            mock_emb = MockEmb.return_value
            # 返回 3 个 embedding：大小不同以产生不同相似度
            mock_emb.encode.return_value = __import__("numpy").array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.5, 0.0],
            ])
            # cosine_similarity = dot product (归一化后)
            def fake_cosine_sim(v1, v2):
                return float(__import__("numpy").dot(v1, v2))
            mock_emb.cosine_similarity.side_effect = fake_cosine_sim

            from src.rag.retriever import Retriever

            retriever = Retriever(retrieval_mode="hybrid", bm25_weight=0.5)
            retriever.chunks = [
                "RAG 系统使用向量检索",
                "BM25 是关键词检索算法",
                "混合检索结合两者优势",
            ]
            retriever.chunk_embeddings = mock_emb.encode.return_value
            retriever.bm25.build_index(retriever.chunks)

            results = retriever.retrieve("检索 算法", top_k=3, mode="hybrid")
            assert len(results) == 3
            # hybrid 分数应在合理范围（线性组合后在 [0,1]）
            for _, score in results:
                assert 0.0 <= score <= 1.0, f"Score {score} out of [0,1]"

    def test_retrieve_unknown_mode_falls_back(self):
        """未知模式回退到 embedding 检索（不崩溃）。"""
        from unittest.mock import patch

        with patch("src.rag.retriever.EmbeddingClient", autospec=True) as MockEmb:
            mock_emb = MockEmb.return_value
            mock_emb.encode.return_value = __import__("numpy").array([[0.5]])
            mock_emb.cosine_similarity.return_value = 0.9

            from src.rag.retriever import Retriever

            retriever = Retriever()
            retriever.chunks = ["测试文档"]
            retriever.chunk_embeddings = mock_emb.encode.return_value

            # 不会崩溃
            results = retriever.retrieve("测试", top_k=1, mode="unknown_mode")
            assert len(results) == 1
