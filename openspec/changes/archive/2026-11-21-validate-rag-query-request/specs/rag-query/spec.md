# Spec Delta

## Purpose

规定 RAG 问答 HTTP 接口在检索和生成之前，哪些请求体可以接受，哪些必须直接拒绝。

## ADDED Requirements

### Requirement: Question must be non-blank

系统 MUST 拒绝 `question` 在去掉首尾空白后为空的问答请求。

#### Scenario: Empty question

- **WHEN** 客户端提交 `question` 为空字符串的问答请求
- **THEN** 系统以 HTTP 422 拒绝该请求
- **AND** 不返回生成的回答

#### Scenario: Whitespace-only question

- **WHEN** 客户端提交 `question` 只含空白字符的问答请求
- **THEN** 系统以 HTTP 422 拒绝该请求

### Requirement: Top k stays in range

系统 MUST 只接受 1 到 10（含）的 `top_k`。省略 `top_k` 时，系统 MUST 按 3 处理。

#### Scenario: Top k below range

- **WHEN** 客户端提交 `top_k` 为 0 且 `question` 非空的问答请求
- **THEN** 系统以 HTTP 422 拒绝该请求

#### Scenario: Top k above range

- **WHEN** 客户端提交 `top_k` 为 11 且 `question` 非空的问答请求
- **THEN** 系统以 HTTP 422 拒绝该请求

#### Scenario: Top k omitted

- **WHEN** 客户端提交非空 `question` 且省略 `top_k`
- **THEN** 系统不因 `top_k` 拒绝该请求
- **AND** 按 `top_k` 为 3 继续处理

### Requirement: Retrieval mode is one of the known values

系统 MUST 只接受 `embedding`、`bm25`、`hybrid` 作为 `retrieval_mode`。省略该字段时，系统 MUST 接受请求。

#### Scenario: Unknown retrieval mode

- **WHEN** 客户端提交 `retrieval_mode` 为 `keyword` 且 `question` 非空的问答请求
- **THEN** 系统以 HTTP 422 拒绝该请求

#### Scenario: Retrieval mode omitted

- **WHEN** 客户端提交非空 `question` 且省略 `retrieval_mode`
- **THEN** 系统不因 `retrieval_mode` 拒绝该请求
