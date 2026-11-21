import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.app import api
from src.app.api import RAGQueryRequest


def test_strips_question_and_applies_defaults():
    request = RAGQueryRequest(question="  hello  ")

    assert request.question == "hello"
    assert request.top_k == 3
    assert request.retrieval_mode is None


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "   "},
        {"question": "hello", "top_k": 0},
        {"question": "hello", "top_k": 11},
        {"question": "hello", "retrieval_mode": "keyword"},
    ],
)
def test_rejects_invalid_request(payload):
    with pytest.raises(ValidationError):
        RAGQueryRequest(**payload)


def test_accepts_bounds_and_known_mode():
    request = RAGQueryRequest(question=" hello ", top_k=10, retrieval_mode="hybrid")

    assert request.question == "hello"
    assert request.top_k == 10
    assert request.retrieval_mode == "hybrid"


def test_invalid_query_returns_422_without_calling_pipeline(monkeypatch):
    def fail_get_pipeline():
        raise AssertionError("pipeline should not be called")

    monkeypatch.setattr(api, "_get_pipeline", fail_get_pipeline)
    client = TestClient(api.app)
    payloads = [
        {"question": ""},
        {"question": "   "},
        {"question": "hello", "top_k": 0},
        {"question": "hello", "top_k": 11},
        {"question": "hello", "retrieval_mode": "keyword"},
    ]

    for payload in payloads:
        response = client.post("/rag/query", json=payload)

        assert response.status_code == 422
        assert "answer" not in response.json()
