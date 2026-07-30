from pathlib import Path

from src.tools.kb_builder import build_index_records, query_index, save_jsonl


def test_build_index_records_from_directory_with_pdf_and_txt(
    tmp_path: Path, monkeypatch
):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    (docs_dir / "a.txt").write_text("苹果。香蕉。", encoding="utf-8")
    (docs_dir / "b.md").write_text("橘子。葡萄。", encoding="utf-8")
    (docs_dir / "c.pdf").write_bytes(b"fake")

    monkeypatch.setattr(
        "src.tools.kb_builder.extract_text_from_pdf",
        lambda _: [{"page": 1, "text": "桃子。西瓜。", "metadata": {}}],
    )

    records, stats = build_index_records(str(docs_dir), chunk_size=20, chunk_overlap=5)

    assert stats["file_count"] == 3
    assert stats["chunk_count"] == len(records)
    assert len(records) >= 3
    assert all("source_file" in record for record in records)
    assert all("source_relpath" in record for record in records)
    assert all("source_dir" in record for record in records)
    assert all("document_id" in record for record in records)
    assert all("chunk_id" in record for record in records)
    assert all("chunk_text" in record for record in records)
    assert all("char_count" in record for record in records)
    assert all("content_hash" in record for record in records)


def test_save_jsonl_writes_lines(tmp_path: Path):
    output = tmp_path / "kb_index.jsonl"
    records = [
        {
            "document_id": "doc001",
            "source_file": "docs/a.txt",
            "source_relpath": "a.txt",
            "source_dir": ".",
            "title": "a",
            "page": None,
            "chunk_id": 1,
            "chunk_text": "示例",
            "char_count": 2,
            "content_hash": "hash001",
        }
    ]

    save_jsonl(records, output)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert "docs/a.txt" in lines[0]


def test_query_index_returns_top_k(monkeypatch, tmp_path: Path):
    index_file = tmp_path / "kb_index.jsonl"
    records = [
        {
            "document_id": "d1",
            "source_file": "docs/a.txt",
            "source_relpath": "a.txt",
            "source_dir": ".",
            "title": "a",
            "page": None,
            "chunk_id": 1,
            "chunk_text": "苹果 香蕉",
            "char_count": 5,
            "content_hash": "h1",
        },
        {
            "document_id": "d2",
            "source_file": "docs/b.txt",
            "source_relpath": "b.txt",
            "source_dir": ".",
            "title": "b",
            "page": None,
            "chunk_id": 1,
            "chunk_text": "西瓜 葡萄",
            "char_count": 5,
            "content_hash": "h2",
        },
    ]
    save_jsonl(records, index_file)

    class FakeEmbeddingClient:
        def encode(self, texts):
            if len(texts) == 1:
                return [[1.0, 0.0]]
            return [[1.0, 0.0], [0.0, 1.0]]

    monkeypatch.setattr("src.tools.kb_builder.EmbeddingClient", FakeEmbeddingClient)

    result = query_index(str(index_file), "苹果", top_k=1)
    assert len(result) == 1
    assert result[0]["source_relpath"] == "a.txt"
    assert result[0]["rank"] == 1
