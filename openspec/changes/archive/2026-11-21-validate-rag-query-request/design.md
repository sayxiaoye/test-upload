# Design

## Context

`RAGQueryRequest` 目前只有类型注解：`question: str`、`top_k: int = 3`、`retrieval_mode: str | None = None`。FastAPI 在类型不匹配时才会返回 422。取值是否合法要到 `pipeline.query()` 之后才暴露。约束见 `specs/rag-query/spec.md`。动机见 `proposal.md` 的 Why。

## Goals / Non-Goals

**Goals:**

- 在请求模型上拒绝空白问题、越界 `top_k` 和未知 `retrieval_mode`
- 合法请求仍走现有的 `pipeline.query(question, top_k_retrieve=5, top_k_rerank=request.top_k, retrieval_mode=...)`

**Non-Goals:**

- 不改检索、重排、生成实现
- 不把召回条数从固定的 5 改成可配置
- 不把未知检索模式的「静默回退到 embedding」改成检索器内部的行为；只在 API 边界拦住

## Decisions

### Decision 1: 在 Pydantic 模型上校验，而不是在路由里手写判断

`RAGQueryRequest` 使用 `Field` 和校验器。FastAPI 会把校验失败转成 422，路由函数只有在模型已经合法时才会执行，因此不会调用 pipeline。

备选：在 `rag_query` 里 `raise HTTPException(400)`。这样要自己维护错误结构，也和 FastAPI 对请求体的默认行为不一致。

### Decision 2: `question` 先去掉首尾空白，再要求非空

使用字段校验器对 `question` 做 `strip()`。结果为空则校验失败。这样 `"   "` 和 `""` 都是 422，通过校验的问题不再带首尾空白。

备选：只设 `min_length=1`。纯空白字符串长度大于 0，挡不住。

### Decision 3: `top_k` 用闭区间 1-10，默认 3

`Field(default=3, ge=1, le=10)`。0、负数和 11 及以上都是 422。省略字段时仍是 3，并继续作为 `top_k_rerank` 传入。

备选：越界时夹到 1 或 10。Spec 要求拒绝，不要求静默改写。

### Decision 4: `retrieval_mode` 用字面量联合，省略仍为 `None`

类型改为 `Literal["embedding", "bm25", "hybrid"] | None = None`。其他字符串由 Pydantic 拒绝。不传该字段时保持 `None`，pipeline 继续使用检索器自己的默认模式。

## Risks / Trade-offs

- [已有调用方传入 `top_k` > 10 或自定义 mode] → 开始收到 422。这是这次变更的预期行为，不另做兼容层。
- [`top_k` 最大 10，但召回仍固定为 5] → 重排结果不会多于召回条数。本次不调整召回条数，避免把校验改动扩成检索行为变更。
