"""
重排模块
对找回结果进行精排优化
# 由于本地无法运行大型重排模型，我们用一个规则重排器来演示重排逻辑，方便你理解概念。
"""

import json
import math
from pathlib import PurePath

from sentence_transformers import CrossEncoder

from src.rag.llm_client import LLMClient

DEFAULT_CROSS_ENCODER_MODEL = (
    r"D:/AI_Models/huggingface/hub/"
    r"models--BAAI--bge-reranker-v2-m3/"
    r"snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
)


def get_model_display_name(model_name: str) -> str:
    """从模型路径中提取适合终端显示的目录名。"""
    for part in PurePath(model_name).parts:
        if part.startswith("models--"):
            return part
    return PurePath(model_name).name or model_name


class SimpleReranker:
    """
    简单重排器
    用LLM对召回结果重新打分排序
    """

    def __init__(self):
        print(f"\033[1;32m{'=' * 20} 【SimpleReranker】 {'=' * 20}\033[0m")

        self.llm = LLMClient()

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """
        用LLM对候选结果重排

        Args:
            query: 用户查询
            candidates: [(chunk, embedding_score), ...]
            top_k: 返回数量

        Returns:
            重排后的[(chunk, rerank_sorce), ...]
        """
        if not candidates:
            return []

        # 构建重排 Prompt
        docs_text = "\n\n---\n\n".join(
            [f"文档 {i + 1}: {chunk[:200]}" for i, (chunk, _) in enumerate(candidates)]
        )

        messages = [
            {
                "role": "system",
                "content": """
                        你是一个相关性评估专家。根据用户查询，对以下文档进行相关性评分（0-10分）。

                        评分标准：
                        - 10分：完全回答了查询的核心问题
                        - 7-9分：回答了一部分，但不够完整
                        - 4-6分：相关但不直接
                        - 0-3分：不相关或无关

                        输出格式（JSON）：
                        {"scores": [{"doc_id": 1, "score": 9, "reason": "..."}, ...]}
                        """,
            },
            {
                "role": "user",
                "content": f"""
                        用户查询: {query}

                        文档列表:
                        {docs_text}

                        请对每个文档打分，并说明理由。
                        """,
            },
        ]

        try:
            respones = self.llm.chat(messages, temperature=0.1)
            # 尝试解析 JSON
            start = respones.find("{")
            end = respones.rfind("}") + 1

            if start >= 0 and end > start:
                json_str = respones[start:end]
                data = json.loads(json_str)
                scores = data.get("scores", [])
                # 按分数重新排序
                score_map: dict = {item["doc_id"]: item["score"] for item in scores}
                reranked: list = []
                for i, (chunk, emb_score) in enumerate(candidates):
                    rerank_score = score_map.get(i + 1, emb_score / 2)
                    reranked.append((chunk, rerank_score))
                reranked.sort(key=lambda x: x[1], reverse=True)
                return reranked[:top_k]
        except Exception as e:
            print(f"\033[1;32m⚠️ 重排失败，返回原始结果: {e}\033[0m")

        # 降级：按原顺序返回
        return candidates[:top_k]


class CrossEncoderReranker:
    """基于 Cross-Encoder 的精排器。"""

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER_MODEL):
        print(f"\033[1;32m{'=' * 20} 【CrossEncoder】 {'=' * 20}\033[0m")

        self.model = CrossEncoder(model_name)
        print(f"\033[1;32m📂 找到本地模型: {get_model_display_name(model_name)}\033[0m")

    @staticmethod
    def _sigmoid(score: float) -> float:
        return 1.0 / (1.0 + math.exp(-score))

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        print(f"\033[1;32m{'=' * 20} 【CrossEncoder】-rerank{'=' * 20}\033[0m")
        if not candidates:
            return []

        pairs = [[query, chunk] for chunk, _ in candidates]
        raw_scores = self.model.predict(pairs)
        reranked = [
            (chunk, self._sigmoid(float(score)))
            for (chunk, _), score in zip(candidates, raw_scores, strict=False)
        ]
        reranked.sort(key=lambda item: item[1], reverse=True)
        return reranked[:top_k]


class Reranker:
    """正式重排器，优先使用 Cross-Encoder，失败时回退到简单实现。"""

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER_MODEL):
        print(f"\033[1;32m{'=' * 20} 【Reranker】 {'=' * 20}\033[0m")

        self.backend: SimpleReranker | CrossEncoderReranker

        try:
            self.backend = CrossEncoderReranker(model_name=model_name)
            self.backend_name = "cross-encoder"
        except Exception as exc:
            print(f"\033[1;32m⚠️ Cross-Encoder 不可用，回退到 SimpleReranker: {exc}\033[0m")
            self.backend = SimpleReranker()
            self.backend_name = "simple"

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        print(f"\033[1;32m{'=' * 20} 【Reranker】-rerank{'=' * 20}\033[0m")
        return self.backend.rerank(query, candidates, top_k=top_k)
