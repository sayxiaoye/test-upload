# my-first-project

面向文档的 RAG（检索增强生成）问答系统 — Python 全栈 AI 项目。

支持文档导入 → 切分 → 向量化 → 检索 → 重排 → LLM 生成的完整链路，具备工程化项目结构、CLI 工具和 API 服务。

## 项目目标

- 构建一个可运行的 RAG 问答系统，支持多文档知识库检索
- 展示 Python 工程化能力（模块化、测试、CI、类型检查、容器化）
- 产出可直接用于求职展示的项目成果

## 目录结构

```
my-first-project/
├── src/
│   ├── rag/                  # RAG 核心模块
│   │   ├── chunker.py        #   文本切分（固定大小 + 语义切分）
│   │   ├── embedding.py      #   向量化（sentence-transformers）
│   │   ├── retriever.py      #   召回检索（余弦相似度）
│   │   ├── reranker.py       #   重排精排（Cross-Encoder / LLM）
│   │   ├── llm_client.py     #   大模型客户端（模型别名 + 模板集成）
│   │   ├── pipeline.py       #   RAG 完整流程编排
│   │   ├── prompt_templates.py # 提示词模板管理（4 个预置模板）
│   │   └── model_compare.py  #   多模型对比演示
│   ├── eval/                 # 评估模块
│   │   ├── evaluate_rag.py   #   RAG 评估日志生成 / 分析
│   │   └── batch_score.py    #   人工评分工具
│   ├── tools/
│   │   └── kb_builder.py     # 知识库构建 CLI（目录 → JSONL）
│   ├── app/
│   │   ├── main.py           # 统一入口（rag / build / serve）
│   │   ├── rag_cli.py        # RAG 问答 CLI
│   │   ├── cli.py            # 通用 CLI 工具
│   │   └── api.py            # FastAPI 接口
│   ├── core/                 # 核心基础设施
│   │   ├── config.py         #   配置管理（YAML + 环境变量）
│   │   ├── api_client.py     #   HTTP 请求封装
│   │   ├── pdf_processor.py  #   PDF 解析
│   │   ├── text_processor.py #   文本清洗 / 正则
│   │   ├── data_processor.py #   数据处理（pandas）
│   │   └── file_ops.py       #   文件读写工具
│   ├── utils/
│   │   └── logging_utils.py  # 统一日志模块（配置驱动 + 结构化 JSON）
│   ├── database.py           # SQLAlchemy ORM（文档 & 查询日志）
│   └── agent_demo.py         # Agent / 工具调用 / ReAct 演示
├── tests/                    # 单元测试（67 个）
├── config/
│   └── config.yaml           # 应用配置
├── data/                     # 示例数据与索引文件
├── Dockerfile                # 容器化部署
├── pyproject.toml            # 项目元数据 / 依赖 / 工具配置
├── .env.example              # 环境变量模板
└── absorb_python_SPEC.md     # 学习计划与进度跟踪
```

## 快速开始

### 1. 环境准备

```bash
# Python 3.12+ 虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -e .
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key
```

### 3. 运行

```bash
# 统一入口（推荐）
python -m src.app.main --help          # 查看帮助
python -m src.app.main rag "什么是RAG？" --doc-dir data/   # RAG 问答
python -m src.app.main build data/ --output data/kb_index.jsonl  # 构建知识库
python -m src.app.main serve --port 8000  # 启动 API 服务

# 独立 CLI
python -m src.app.rag_cli "什么是向量数据库？" --doc-file data/sample.pdf
python -m src.tools.kb_builder data/ --output data/kb_index.jsonl

# 演示脚本
python -m src.rag.prompt_templates   # 提示词模板演示
python -m src.rag.model_compare      # 多模型对比演示
python -m src.utils.logging_utils    # 日志模块演示
```

## RAG 问答示例

```bash
# 使用内置默认文档
python -m src.app.main rag "Python 适合做什么？"

# 输出:
# ============================================================
# 问题: Python 适合做什么？
# 文档: 内置默认文档
# ------------------------------------------------------------
# 回答:
# 1. Python 适合脚本编写、自动化任务和数据分析
# 2. Python 在 AI 应用开发领域广泛应用
# 3. ...
# ------------------------------------------------------------
# 参考来源 (3 条):
#   [1] Python 是一种通用编程语言，适合脚本、自动化...
#   [2] ...
```

## API 服务

```bash
# 启动
python -m src.app.main serve --port 8000 --reload

# 健康检查
curl http://127.0.0.1:8000/health

# API 文档
open http://127.0.0.1:8000/docs
```

## Docker 部署

```bash
docker build -t my-rag-app .
docker run -p 8000:8000 --env-file .env my-rag-app python -m src.app.main serve --host 0.0.0.0
```

## 测试与质量

```bash
# 运行所有测试
pytest tests/ -v

# 代码检查
ruff check src/

# 类型检查
mypy src/

# 格式化
ruff format src/
```

## 核心特性

| 能力 | 实现 |
|---|---|
| 文档切分 | 固定大小切分 + 语义切分，支持重叠窗口 |
| 向量检索 | sentence-transformers 多语言模型，余弦相似度 |
| 重排精排 | Cross-Encoder (BGE Reranker) + LLM 评分回退 |
| 模型管理 | 别名系统（fast/pro），配置驱动切换 |
| 提示词管理 | 4 个预置模板 + 注册中心 + 参数化渲染 + 失败样例清单 |
| 评估体系 | 评估日志生成 / 人工评分 / 低分分析 |
| 日志系统 | 配置驱动 + 控制台/文件双输出 + 结构化 JSON |
| 工程化 | pytest (67 tests) / ruff / mypy / pre-commit / CI / Docker |

## 技术栈

- **语言**: Python 3.12+
- **LLM**: DeepSeek API（OpenAI 兼容）
- **Embedding**: sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2)
- **Reranker**: BGE Reranker v2-m3
- **API 框架**: FastAPI + uvicorn
- **数据库**: SQLite + SQLAlchemy ORM
- **数据处理**: pandas, pdfplumber
- **测试**: pytest, pytest-cov
- **代码质量**: ruff, mypy, pre-commit

## License

MIT
