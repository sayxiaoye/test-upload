"""RAG 领域包。"""

from src.rag.chunker import TextChunker
from src.rag.embedding import EmbeddingClient
from src.rag.llm_client import LLMClient
from src.rag.pipeline import RAGPipeline
from src.rag.prompt_templates import (  # E3 新增: 提示词模板管理
    PromptRegistry,
    PromptTemplate,
    get_registry,
    render_document_summary,
    render_rag_qa,
    render_relevance_eval,
)
from src.rag.reranker import CrossEncoderReranker, Reranker
from src.rag.retriever import Retriever

__all__ = [
    "TextChunker",
    "EmbeddingClient",
    "LLMClient",
    "RAGPipeline",
    "CrossEncoderReranker",
    "Reranker",
    "Retriever",
    # E3 新增导出
    "PromptRegistry",
    "PromptTemplate",
    "get_registry",
    "render_rag_qa",
    "render_document_summary",
    "render_relevance_eval",
]
