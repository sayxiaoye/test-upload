import json

from src.eval.evaluate_rag import analyze_results, generate_logs


class FakePipeline:
    def __init__(self):
        self.indexed = []

    def index_document(self, text):
        self.indexed.append(text)

    def query(self, question):
        return {
            "question": question,
            "context": "context",
            "answer": f"answer:{question}",
            "sources": [{"id": 1, "content": "doc", "score": 1.0}],
        }


def test_generate_logs_writes_json_files(tmp_path, monkeypatch):
    monkeypatch.setattr("src.eval.evaluate_rag.RAGPipeline", FakePipeline)

    result = generate_logs(
        log_dir=str(tmp_path / "rag_eval"),
        document="demo",
        questions=["what"],
    )

    assert result["generated"] == 1
    assert (tmp_path / "rag_eval").exists()
    assert len(list((tmp_path / "rag_eval").glob("*.json"))) == 1


def test_analyze_results_reports_summary(tmp_path):
    log_dir = tmp_path / "rag_eval"
    log_dir.mkdir()

    scored_log = log_dir / "scored.json"
    scored_log.write_text(
        json.dumps({"question": "hello", "answer": "world", "score": 2}),
        encoding="utf-8",
    )

    unscored_log = log_dir / "pending.json"
    unscored_log.write_text(
        json.dumps({"question": "later", "answer": "world"}),
        encoding="utf-8",
    )

    result = analyze_results(log_dir=str(log_dir))

    assert result["total"] == 2
    assert result["scored"] == 1
    assert result["pending"] == 1
    assert result["low_score_count"] == 1
