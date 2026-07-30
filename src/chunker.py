"""兼容层：保留旧的 src.chunker 导入路径。"""

from src.rag.chunker import TextChunker, print_chunks

if __name__ == "__main__":
    # 测试文本
    test_text = """
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

    chunker = TextChunker(chunk_size=150, chunk_overlap=30)
    print("=" * 60)
    print("📄 文本切分演示")
    print("=" * 60)
    print(f"\n原始文本长度: {len(test_text)} 字符\n")
    # 1.固定大小切分
    fixed_chunks = chunker.fixed_size_chunk(test_text)
    print_chunks(fixed_chunks, "固定大小切分 (chunk_size=150, overlap=30)")

    # 2.语义切分
    semantic_chunks = chunker.semantic_chunk(test_text)
    print_chunks(semantic_chunks, "语义切分")
