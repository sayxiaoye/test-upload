# Proposal

## Why

`POST /rag/query` 只检查字段类型。空问题、越界的 `top_k`、非法的 `retrieval_mode` 都会进入检索和生成，浪费一次 pipeline 调用，错误也变成 500 而不是明确的请求错误。

## What Changes

- 拒绝空白的 `question`（去掉首尾空白后为空）
- 把 `top_k` 限制在 1 到 10，默认仍为 3
- `retrieval_mode` 只允许 `embedding`、`bm25`、`hybrid`，省略时仍为 `null`
- 非法请求由 FastAPI / Pydantic 返回 422，不进入 `RAGPipeline`

## Capabilities

### New Capabilities

- `rag-query`：RAG 问答 HTTP 接口对请求体的接受规则

### Modified Capabilities

- 无。`openspec/specs/` 里还没有既有 capability。

## Impact

- `src/app/api.py`：给 `RAGQueryRequest` 加上字段约束
- 调用方：原先能发出去的空问题、`top_k <= 0` 或未知检索模式，会收到 422
- 检索、重排、生成逻辑不变
