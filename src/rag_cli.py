"""E 阶段第 3 个示例：从文件读取文档内容的 RAG CLI。"""

import argparse
from pathlib import Path

from src.core.pdf_processor import extract_full_text
from src.rag_pipeline import RAGPipeline

DEFAULT_DOCUMENT = """
Python 是一种通用编程语言，适合脚本、自动化、数据分析和 AI 应用开发。
RAG 是检索增强生成，它通过先检索相关文档，再让大模型基于这些文档做回答。
大模型可以用于聊天、翻译、总结和问答等任务。
"""

SUPPORTED_SUFFIXES = {".txt", ".md", ".json", ".pdf"}


def load_document(document_path: str | None) -> str:
    """读取知识库文档；支持单个文件或目录。"""
    if not document_path:
        return DEFAULT_DOCUMENT

    path = Path(document_path)
    if not path.exists():
        raise FileNotFoundError(f"文档文件不存在: {path}")

    if path.is_dir():
        documents: list[str] = []
        for file_path in sorted(path.rglob("*")):
            if not file_path.is_file():
                continue
            suffix = file_path.suffix.lower()
            if suffix == ".pdf":
                text = extract_full_text(str(file_path))
            elif suffix in {".txt", ".md", ".json"}:
                text = file_path.read_text(encoding="utf-8")
            else:
                continue

            if text.strip():
                documents.append(f"===== 文件: {file_path.name} =====\n{text}")

        if not documents:
            raise ValueError(f"目录中没有可读取的文档: {path}")
        return "\n\n".join(documents)

    if path.is_file():
        if path.suffix.lower() == ".pdf":
            return extract_full_text(str(path))

        if path.suffix.lower() in {".txt", ".md", ".json"}:
            return path.read_text(encoding="utf-8")

    raise ValueError(f"暂不支持的文档格式: {path.suffix}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从文件读取文档的最小 RAG 问答 CLI")
    parser.add_argument("question", help="用户输入的问题")
    parser.add_argument(
        "--doc-file",
        default=None,
        help="包含知识库内容的文件或文件夹路径",
    )
    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=5,
        help="检索召回的候选数量，默认 5",
    )
    parser.add_argument(
        "--rerank-k",
        type=int,
        default=3,
        help="重排后用于生成回答的参考片段数量，默认 3",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    pipeline = RAGPipeline()
    document = load_document(args.doc_file)
    pipeline.index_document(document)
    result = pipeline.query(
        args.question,
        top_k_retrieve=args.retrieve_k,
        top_k_rerank=args.rerank_k,
    )

    print("=" * 20 + " RAG_CLI " + "=" * 20)
    print(f"🧠 问题：{args.question}")
    print(f"📌 使用文档：{args.doc_file or '内置默认文档'}")
    print("📖 回答：")
    print(result["answer"])
    print("\n来源：")
    for source in result["sources"]:
        print(f"【】 {source['content']}")
    print(f"📚 参考片段数: {len(result['sources'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
