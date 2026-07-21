"""
FastAPI应用
提供API接口, 包括健康检查、文档处理等
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.core import fetch_post, read_note, save_note

# 创建 FastAPI 应用
app = FastAPI(
    title="My first Project API",
    description="A simple API service for demonstrating FastAPI",
    version="0.1.0",
)


# ============ 响应模型 ============
class NoteResponse(BaseModel):
    """笔记本响应模型"""

    content: str


class NoteRequest(BaseModel):
    """笔记本请求模型"""

    content: str
    filename: str = "data/note.txt"


class PostResponse(BaseModel):
    """文章响应模型"""

    id: int
    title: str
    body: str
    userId: int


# ============ API 端点 ============
@app.get("/")
async def root():
    """根路径，返回欢迎信息"""
    return {
        "message": "welcome used My Frist Project API",
        "docs": "/docs",
        "redoc": "/redoc",
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


@app.get("/notes/{filename:path}")
async def get_note(filename: str = "data/note.txt") -> NoteResponse:
    """
    读取笔记本文件内容

    Args:
        filename: 文件路径（相对于项目根目录）

    Returns:
        笔记内容
    """
    try:
        content = read_note(filename)
        return NoteResponse(content=content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/notes")
async def create_note(request: NoteRequest) -> dict:
    """
    保存笔记到文件

    Args:
        request: 包含内容和文件名的请求体

    Returns:
        保存结果
    """
    try:
        save_note(request.filename, request.content)
        return {"status": "success", "message": f"已保存到 {request.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/posts/{post_id}")
async def get_post(post_id: int) -> PostResponse:
    try:
        data = fetch_post(post_id)
        if not data:
            raise HTTPException(status_code=404, detail="文章不存在")
        return PostResponse(
            id=data.get("id", 0),
            title=data.get("title", ""),
            body=data.get("body", ""),
            userId=data.get("userId", 0),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
