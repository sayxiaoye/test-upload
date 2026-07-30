"""RAG 评估模块。

支持生成日志、人工评分和分析报告三步流程。
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.rag.pipeline import RAGPipeline

DEFAULT_LOG_DIR = "logs/rag_eval"
DEFAULT_DOCUMENT = """
机器学习是人工智能的一个分支，它让计算机能够从数据中学习。
传统的编程方式需要程序员明确写出规则，而机器学习则通过算法自动发现数据中的模式。

深度学习是机器学习的一个子集，它使用多层神经网络来学习数据的表示。
深度学习的核心是神经网络，它由多个层组成，每一层都从前一层学习特征。

RAG（检索增强生成）是一种结合检索和生成的技术。
它先从知识库中检索相关信息，然后让大语言模型基于这些信息生成回答。
RAG 能够显著提高回答的准确性和可解释性。
"""
DEFAULT_QUESTIONS = [
    "什么是机器学习？",
    "深度学习和机器学习是什么关系？",
    "RAG 的核心思想是什么？",
    "这个文档里没有的内容会怎样？",
]


class RAGEvaluator:
    """RAG 系统评估器。"""

    def __init__(self, log_dir: str = DEFAULT_LOG_DIR):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("rag_evaluator")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if not self.logger.handlers:
            log_file = self.log_dir / "evaluate.log"
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        self.pipeline = RAGPipeline()

    def evaluate_qa(
        self, question: str, expected_answer: str | None = None
    ) -> dict[str, Any]:
        """执行一次 RAG 问答并记录结果。"""
        result = self.pipeline.query(question)

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "context": result["context"],
            "num_sources": len(result["sources"]),
        }

        if expected_answer:
            log_entry["expected_answer"] = expected_answer

        log_file = (
            self.log_dir
            / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(question)}.json"
        )
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)

        self.logger.info("✅ 记录评估: %s...", question[:30])
        return log_entry


def generate_logs(
    log_dir: str = DEFAULT_LOG_DIR,
    document: str | None = None,
    questions: list[str] | None = None,
) -> dict[str, Any]:
    """生成一批评估日志。"""
    evaluator = RAGEvaluator(log_dir=log_dir)
    evaluator.pipeline.index_document(document or DEFAULT_DOCUMENT)

    qa_questions = questions or DEFAULT_QUESTIONS
    generated = []
    for question in qa_questions:
        generated.append(evaluator.evaluate_qa(question))

    return {
        "generated": len(generated),
        "log_dir": str(evaluator.log_dir),
        "questions": qa_questions,
    }


def analyze_results(log_dir: str = DEFAULT_LOG_DIR) -> dict[str, Any]:
    """分析已评分和待评分的日志。"""
    log_path = Path(log_dir)
    logs = sorted(log_path.glob("*.json"), key=lambda p: p.name)

    if not logs:
        return {
            "total": 0,
            "scored": 0,
            "pending": 0,
            "low_score_count": 0,
            "low_score_ratio": 0.0,
            "failures": [],
        }

    scored = 0
    pending = 0
    low_score_logs: list[dict[str, Any]] = []

    for log_file in logs:
        with open(log_file, encoding="utf-8") as f:
            data = json.load(f)

        score = data.get("score")
        if score is None:
            pending += 1
        else:
            scored += 1

        if isinstance(score, (int, float)) and int(score) <= 3:
            low_score_logs.append(
                {
                    "question": data.get("question"),
                    "answer": data.get("answer"),
                    "score": score,
                    "sources": len(data.get("sources", [])),
                }
            )

    return {
        "total": len(logs),
        "scored": scored,
        "pending": pending,
        "low_score_count": len(low_score_logs),
        "low_score_ratio": round(len(low_score_logs) / len(logs), 2) if logs else 0.0,
        "failures": low_score_logs[:10],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG 评估三步流程")
    parser.add_argument(
        "--generate",
        action="store_true",
        help="生成评估日志，随后运行 python -m src.batch_score 进行评分",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="查看已评分日志的分析报告",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.generate:
        summary = generate_logs()
        print("=" * 60)
        print("✅ 已生成评估日志")
        print(f"保存目录: {summary['log_dir']}")
        print(f"共生成: {summary['generated']} 条")
        print("下一步: 运行 python -m src.batch_score 进行人工评分")
        return 0

    if args.analyze:
        summary = analyze_results()
        print("=" * 60)
        print("📊 RAG 评估分析报告")
        print("=" * 60)
        print(f"总记录数: {summary['total']}")
        print(f"已评分: {summary['scored']}")
        print(f"待评分: {summary['pending']}")
        print(f"低分样例数: {summary['low_score_count']}")
        print(f"低分占比: {summary['low_score_ratio']:.2%}")

        if summary["failures"]:
            print("\n低分样例:")
            for item in summary["failures"]:
                print(f"- 问题: {item['question']}")
                print(f"  回答: {item['answer'][:80]}...")
                print(f"  评分: {item['score']}，来源数: {item['sources']}")
        else:
            print("\n当前没有低分样例")
        return 0

    summary = generate_logs()
    print("=" * 60)
    print("📊 RAG 评估演示")
    print("=" * 60)
    print(f"已生成 {summary['generated']} 条评估日志，默认保存到 {summary['log_dir']}")
    print("提示：使用 --generate 生成日志，使用 --analyze 查看分析报告")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
