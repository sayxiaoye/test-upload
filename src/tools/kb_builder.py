"""
knowledge base builder
E2 阶段：知识库构建工具（目录 -> 切分 -> JSONL 索引）。
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.core.pdf_processor import extract_text_from_pdf
from src.rag.chunker import TextChunker
from src.rag.embedding import EmbeddingClient

SUPPORTED_SUFFIXES = {".txt", ".md", ".json", ".pdf"}


def collect_input_files(input_path: Path) -> list[Path]:
    """收集可处理文件，支持单文件或目录。"""
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"不支持的文件类型: {input_path.suffix}")
        return [input_path]

    files = [
        path
        for path in sorted(input_path.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    if not files:
        raise ValueError(f"目录中没有可处理文件: {input_path}")
    return files


def build_document_units(file_path: Path, root_path: Path) -> list[dict[str, object]]:
    """把文件展开为可切分单元（PDF 按页，文本按整文件）。"""
    source_relpath = file_path.relative_to(root_path).as_posix()
    source_dir = file_path.parent.relative_to(root_path).as_posix()
    document_id = hashlib.sha1(source_relpath.encode("utf-8")).hexdigest()[:16]

    if file_path.suffix.lower() == ".pdf":
        pages = extract_text_from_pdf(str(file_path))
        units: list[dict[str, object]] = []
        for page_item in pages:
            text = str(page_item.get("text", "")).strip()
            if not text:
                continue
            metadata = page_item.get("metadata", {})
            title = ""
            if isinstance(metadata, dict):
                title = str(metadata.get("title", "")).strip()

            units.append(
                {
                    "document_id": document_id,
                    "source_file": file_path.as_posix(),
                    "source_relpath": source_relpath,
                    "source_dir": source_dir,
                    "title": title or file_path.stem,
                    "page": int(page_item.get("page", 0)),
                    "text": text,
                }
            )
        return units

    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [
        {
            "document_id": document_id,
            "source_file": file_path.as_posix(),
            "source_relpath": source_relpath,
            "source_dir": source_dir,
            "title": file_path.stem,
            "page": None,
            "text": text,
        }
    ]


def build_index_records(
    input_path: str,
    chunk_size: int = 200,
    chunk_overlap: int = 50,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """从输入路径构建 chunk 级索引记录。"""
    path = Path(input_path)
    files = collect_input_files(path)
    root_path = path if path.is_dir() else path.parent
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    records: list[dict[str, object]] = []
    processed_file_count = 0
    duplicate_chunks = 0
    seen_hashes: set[str] = set()

    for file_path in files:
        units = build_document_units(file_path, root_path)
        if not units:
            continue

        processed_file_count += 1

        for unit in units:
            text = str(unit["text"])
            chunks = chunker.semantic_chunk(text)
            if not chunks:
                chunks = [text]

            for index, chunk_text in enumerate(chunks, start=1):
                content_hash = hashlib.sha1(
                    chunk_text.strip().encode("utf-8")
                ).hexdigest()[:16]
                if content_hash in seen_hashes:
                    duplicate_chunks += 1
                    continue

                seen_hashes.add(content_hash)
                records.append(
                    {
                        "document_id": unit["document_id"],
                        "source_file": unit["source_file"],
                        "source_relpath": unit["source_relpath"],
                        "source_dir": unit["source_dir"],
                        "title": unit["title"],
                        "page": unit["page"],
                        "chunk_id": index,
                        "chunk_text": chunk_text,
                        "char_count": len(chunk_text),
                        "content_hash": content_hash,
                    }
                )

    stats = {
        "file_count": processed_file_count,
        "chunk_count": len(records),
        "duplicate_chunk_count": duplicate_chunks,
    }
    return records, stats


def save_jsonl(records: list[dict[str, object]], output_file: Path) -> None:
    """把索引记录保存为 JSONL。"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(index_file: Path) -> list[dict[str, object]]:
    """读取 JSONL 索引。"""
    if not index_file.exists():
        raise FileNotFoundError(f"索引文件不存在: {index_file}")

    records: list[dict[str, object]] = []
    with index_file.open("r", encoding="utf-8") as file:
        for raw_line in file:
            stripped_line = raw_line.strip()
            if stripped_line:
                records.append(json.loads(stripped_line))
    return records


def query_index(index_file: str, query_text: str, top_k: int = 5) -> list[dict[str, object]]:
    """基于 embedding 对索引进行检索查询（不依赖 LLM）。"""
    records = load_jsonl(Path(index_file))
    if not records:
        return []

    chunks = [str(record.get("chunk_text", "")) for record in records]
    embedding_client = EmbeddingClient()
    chunk_vectors = embedding_client.encode(chunks)
    query_vec = embedding_client.encode([query_text])[0]

    similarities = np.dot(chunk_vectors, query_vec)
    sorted_indices = np.argsort(similarities)[::-1][:top_k]

    results: list[dict[str, object]] = []
    for rank, idx in enumerate(sorted_indices, start=1):
        record = dict(records[int(idx)])
        record["score"] = float(similarities[int(idx)])
        record["rank"] = rank
        results.append(record)
    return results


def print_query_results(results: list[dict[str, object]], query_text: str) -> None:
    """打印查询结果。"""
    print("=" * 20 + " KB Query " + "=" * 20)
    print(f"查询: {query_text}")
    print(f"命中条数: {len(results)}")
    for item in results:
        score_value = item.get("score", 0.0)
        if isinstance(score_value, (int, float, str)):
            score = float(score_value)
        else:
            score = 0.0
        print(
            f"[{item.get('rank')}] score={score:.4f} "
            f"file={item.get('source_relpath')} page={item.get('page')}"
        )
        print(f"  {str(item.get('chunk_text', ''))[:120]}")
    print("=" * 50)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="知识库构建工具：目录 -> JSONL 索引")
    parser.add_argument("input_path", nargs="?", help="输入文件或目录路径")
    parser.add_argument(
        "--output",
        default="data/kb_index.jsonl",
        help="输出 JSONL 文件路径，默认 data/kb_index.jsonl",
    )
    parser.add_argument("--chunk-size", type=int, default=200, help="chunk 大小")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="chunk 重叠")
    parser.add_argument("--query-text", default=None, help="查询文本（启用查询模式）")
    parser.add_argument(
        "--index-file",
        default="data/kb_index.jsonl",
        help="查询模式使用的 JSONL 索引文件路径",
    )
    parser.add_argument("--top-k", type=int, default=5, help="查询返回条数")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.query_text:
        results = query_index(
            index_file=args.index_file,
            query_text=args.query_text,
            top_k=args.top_k,
        )
        print_query_results(results, query_text=args.query_text)
        return

    if not args.input_path:
        parser.error("构建模式需要提供 input_path，或使用 --query-text 进入查询模式")

    records, stats = build_index_records(
        input_path=args.input_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    output_file = Path(args.output)
    save_jsonl(records, output_file)

    print("=" * 20 + " KB Builder " + "=" * 20)
    print(f"输入路径: {args.input_path}")
    print(f"输出文件: {output_file.as_posix()}")
    print(f"处理文档数: {stats['file_count']}")
    print(f"生成 chunk 数: {stats['chunk_count']}")
    print(f"去重跳过 chunk 数: {stats['duplicate_chunk_count']}")
    print("=" * 52)


if __name__ == "__main__":
    main()
