"""评估子包：集中放置评估日志生成、打分与分析能力。"""

from src.eval.batch_score import batch_score
from src.eval.evaluate_rag import analyze_results, generate_logs

__all__ = ["generate_logs", "analyze_results", "batch_score"]
