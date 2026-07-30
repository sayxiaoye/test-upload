"""批量评分工具。

对未评分的 RAG 评估日志进行人工评分。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

DEFAULT_LOG_DIR = "logs/rag_eval"


def batch_score(log_dir: str = DEFAULT_LOG_DIR) -> None:
    """批量对未评分的日志进行人工评分。"""
    log_path = Path(log_dir)
    logs = sorted(log_path.glob("*.json"), key=lambda p: p.name)

    if not logs:
        print("📂 没有找到评估日志文件")
        return

    unscored = []
    for log_file in logs:
        try:
            with open(log_file, encoding="utf-8") as f:
                data = json.load(f)
                if "score" not in data:
                    unscored.append((log_file, data))
        except (json.JSONDecodeError, KeyError):
            continue

    if not unscored:
        print("✅ 所有日志已评分！")
        return

    print(f"📋 找到 {len(unscored)} 条未评分记录\\n")

    for i, (log_file, data) in enumerate(unscored, 1):
        print(f"[{i}/{len(unscored)}]")
        print(f"📌 问题: {data['question']}")
        print(f"📖 回答: {data['answer'][:200]}...")
        print(f"📚 来源数: {data.get('num_sources', 0)}")
        print("-" * 40)

        try:
            score_input = input("评分 (1-5, 0=跳过, q=退出): ").strip()
            if score_input.lower() == "q":
                print("👋 退出评分")
                break
            score = int(score_input)
            if score == 0:
                print("⏭️ 跳过此条\\n")
                continue
            if 1 <= score <= 5:
                data["score"] = score
                data["score_at"] = datetime.now().isoformat()
                with open(log_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"✅ 已评分: {score}分\\n")
            else:
                print("⚠️ 分数必须在 1-5 之间，或 0 跳过\\n")
        except ValueError:
            print("⚠️ 请输入数字\\n")

    all_logs = list(log_path.glob("*.json"))
    scored = 0
    for log_file in all_logs:
        with open(log_file, encoding="utf-8") as f:
            data = json.load(f)
            if "score" in data:
                scored += 1

    print("=" * 60)
    print(f"📊 统计: 共 {len(all_logs)} 条记录，已评分 {scored} 条")
    print("=" * 60)


def main() -> int:
    """CLI 入口。"""
    batch_score()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
