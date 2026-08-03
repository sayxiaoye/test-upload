"""
FastAPI 应用（E6 重写：从文件操作 API → RAG 问答 API）

提供 RAG 问答的 HTTP 接口：
- GET  /             欢迎页
- GET  /health       健康检查
- POST /rag/query    RAG 问答（传入问题，返回回答 + 参考来源）
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.rag.pipeline import RAGPipeline  # RAG 完整流程

# 创建 FastAPI 应用
app = FastAPI(
    title="RAG Q&A API",
    description="面向文档的 RAG 智能问答系统",
    version="0.1.0",
)

class _AppState:
    """应用状态容器，避免 global 声明。"""

    pipeline: RAGPipeline | None = None


def _get_pipeline() -> RAGPipeline:
    """懒加载 RAG pipeline（首次请求时初始化，加载默认知识库）。"""
    if _AppState.pipeline is None:
        _AppState.pipeline = RAGPipeline()
        # 加载预构建的知识库索引
        from pathlib import Path

        index_path = Path("data/kb_index.jsonl")
        if index_path.exists():
            _AppState.pipeline.load_index(str(index_path))
    return _AppState.pipeline


# ============ 请求/响应模型 ============
class RAGQueryRequest(BaseModel):
    """RAG 问答请求体"""

    question: str  # 用户问题
    top_k: int = 3  # 返回的参考片段数量


class RAGQueryResponse(BaseModel):
    """RAG 问答响应体"""

    question: str  # 原始问题
    answer: str  # 生成的回答
    sources: list[dict]  # 参考来源 [{content, score}, ...]


# ============ API 端点 ============
@app.get("/")
async def root():
    """根路径，返回 API 信息"""
    return {
        "service": "RAG Q&A API",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "rag_query": "POST /rag/query",
        },
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


@app.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """
    RAG 问答接口

    传入问题，返回基于知识库的智能回答和参考来源。

    示例请求体:
        {"question": "什么是向量数据库？", "top_k": 3}
    """
    try:
        pipeline = _get_pipeline()
        result = pipeline.query(
            request.question,
            top_k_retrieve=5,
            top_k_rerank=request.top_k,
        )

        return RAGQueryResponse(
            question=request.question,
            answer=result["answer"],
            sources=[
                {"content": s.get("content", ""), "score": s.get("score", 0)}
                for s in result["sources"]
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
