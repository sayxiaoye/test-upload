# Tasks

## 1. 请求模型校验

- [x] 1.1 在 `src/app/api.py` 的 `RAGQueryRequest` 上按 design 加上 `question`、`top_k`、`retrieval_mode` 约束 — verify: 空白问题、`top_k` 为 0 或 11、`retrieval_mode` 为 `keyword` 时模型校验失败；省略 `top_k` 时为 3；省略 `retrieval_mode` 时为 `None`；合法 `question` 会去掉首尾空白
- [x] 1.2 用上述用例为请求模型补测试 — verify: 对应测试文件通过

## 2. Integration Verification

- [x] 2.1 确认非法的 `POST /rag/query` 返回 HTTP 422，且不会调用 pipeline — verify: 用测试客户端发送空问题、越界 `top_k`、未知 `retrieval_mode`，状态码为 422，pipeline 的 query 未被调用
