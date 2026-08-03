"""
my-first-project 统一起动入口（E4 整合）

整合了三大能力：
1. RAG 问答 — 从文档目录构建知识库，支持自然语言检索+LLM生成
2. KB 构建 — 扫描目录，切分/向量化/生成 JSONL 知识库索引
3. API 服务 — FastAPI 提供 HTTP 问答接口 + 健康检查

用法：
    # RAG 问答（从文件/目录读取文档）
    python -m src.app.main rag "什么是向量数据库？" --doc-dir data/

    # KB 构建（目录 → JSONL 知识库）
    python -m src.app.main build data/ --output data/kb_index.jsonl

    # 启动 API 服务
    python -m src.app.main serve --port 8000

    # 查看帮助
    python -m src.app.main --help
"""

import argparse
import sys
from pathlib import Path

from src.utils.logging_utils import get_logger, setup_logging  # E3 统一日志

# 项目启动时初始化日志（INFO 级别，控制台输出）
setup_logging(level="INFO")
logger = get_logger(__name__)


# ============================================================================
# 子命令：RAG 问答
# ============================================================================
def _cmd_rag(args: argparse.Namespace) -> None:
    """对指定文档/目录进行 RAG 问答。"""
    from src.app.rag_cli import load_document  # 复用现有文档加载逻辑

    # 延迟导入：只在用到 RAG 时才加载重量级依赖（Embedding 模型等）
    from src.rag.pipeline import RAGPipeline

    logger.info("初始化 RAG pipeline...")
    pipeline = RAGPipeline()

    # 加载文档：--doc-dir > 预构建知识库 > 内置默认文档
    doc_source = "内置默认文档"
    if args.doc_dir:
        # 用户明确指定了原始文档 → 实时处理（临时探索）
        document = load_document(args.doc_dir)
        pipeline.index_document(document)
        doc_source = args.doc_dir
        logger.info("文档索引完成（实时处理），开始查询")
    else:
        # 从知识库加载（--index 覆盖默认路径）
        index_file = getattr(args, "index", None) or "data/kb_index.jsonl"
        index_path = Path(index_file)
        if index_path.exists():
            pipeline.load_index(str(index_path))
            doc_source = f"知识库索引: {index_path.as_posix()}"
            logger.info("从索引加载: %s", index_path)
        else:
            if getattr(args, "index", None):
                print(f"错误: 知识库索引文件不存在: {args.index}")
                print("请先运行: python -m src.app.main build <目录> --output data/kb_index.jsonl")
                return
            document = load_document(None)
            pipeline.index_document(document)
            logger.info("使用内置默认文档")

    # 执行问答
    result = pipeline.query(
        args.question,
        top_k_retrieve=args.retrieve_k,
        top_k_rerank=args.rerank_k,
    )

    # 输出结果
    print()
    print("=" * 60)
    print(f"问题: {args.question}")
    print(f"文档: {doc_source}")
    print("-" * 60)
    print(f"回答:\n{result['answer']}")
    print("-" * 60)
    print(f"参考来源 ({len(result['sources'])} 条):")
    for src in result["sources"]:
        content_preview = str(src.get("content", ""))[:100]
        print(f"  [{src.get('id')}] {content_preview}...")
    print("=" * 60)


# ============================================================================
# 子命令：知识库构建
# ============================================================================
def _cmd_build(args: argparse.Namespace) -> None:
    """扫描目录，构建 JSONL 知识库索引。"""
    from src.tools.kb_builder import build_index_records, save_jsonl

    logger.info("开始构建知识库: %s", args.input_dir)
    records, stats = build_index_records(
        input_path=args.input_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    output_path = Path(args.output)
    save_jsonl(records, output_path)

    print()
    print("=" * 60)
    print("知识库构建完成")
    print("-" * 60)
    print(f"  输入目录:   {args.input_dir}")
    print(f"  输出文件:   {output_path.as_posix()}")
    print(f"  处理文档:   {stats['file_count']} 个")
    print(f"  生成 chunk:  {stats['chunk_count']} 个")
    print(f"  去重跳过:   {stats['duplicate_chunk_count']} 个")
    print("=" * 60)


# ============================================================================
# 子命令：启动 API 服务
# ============================================================================
def _cmd_serve(args: argparse.Namespace) -> None:
    """启动 FastAPI 服务。"""
    import uvicorn

    logger.info("启动 API 服务: http://%s:%s", args.host, args.port)
    print(f"API 文档: http://{args.host}:{args.port}/docs")
    print(f"健康检查: http://{args.host}:{args.port}/health")
    uvicorn.run(
        "src.app.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


# ============================================================================
# 参数解析
# ============================================================================
def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="my-first-project: 面向文档的 RAG 问答系统",
        prog="my-project",
    )

    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="可用子命令")

    # ---- rag 子命令 ----
    rag_parser = subparsers.add_parser("rag", help="RAG 问答")
    rag_parser.add_argument("question", help="要查询的问题")
    rag_parser.add_argument(
        "--doc-dir",
        default=None,
        help="文档文件或目录路径（支持 txt/md/json/pdf）",
    )
    rag_parser.add_argument(
        "--index",
        default=None,
        help="知识库 JSONL 索引文件路径（默认 data/kb_index.jsonl）",
    )
    rag_parser.add_argument(
        "--retrieve-k", type=int, default=5, help="召回候选数（默认 5）"
    )
    rag_parser.add_argument(
        "--rerank-k", type=int, default=3, help="重排后用于回答的片段数（默认 3）"
    )

    # ---- build 子命令 ----
    build_parser = subparsers.add_parser("build", help="构建 JSONL 知识库")
    build_parser.add_argument("input_dir", help="输入文件或目录路径")
    build_parser.add_argument(
        "--output", default="data/kb_index.jsonl", help="输出 JSONL 文件路径"
    )
    build_parser.add_argument(
        "--chunk-size", type=int, default=200, help="chunk 大小（默认 200 字符）"
    )
    build_parser.add_argument(
        "--chunk-overlap", type=int, default=50, help="chunk 重叠（默认 50 字符）"
    )

    # ---- serve 子命令 ----
    serve_parser = subparsers.add_parser("serve", help="启动 FastAPI 服务")
    serve_parser.add_argument(
        "--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）"
    )
    serve_parser.add_argument(
        "--port", type=int, default=8000, help="监听端口（默认 8000）"
    )
    serve_parser.add_argument(
        "--reload", action="store_true", help="开发模式热重载"
    )

    return parser


# ============================================================================
# 主函数
# ============================================================================
def main(argv: list[str] | None = None) -> int:
    """项目统一入口。根据子命令路由到对应功能。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    # 子命令路由表
    command_map = {
        "rag": _cmd_rag,
        "build": _cmd_build,
        "serve": _cmd_serve,
    }

    handler = command_map.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    try:
        handler(args)
        return 0
    except Exception:
        logger.exception("执行子命令 [%s] 时出错", args.command)
        return 1


if __name__ == "__main__":
    sys.exit(main())
