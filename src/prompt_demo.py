"""
Prompt 设计演示
对比不同 Prompt 的输出效果
"""

from src.llm_client import LLMClient

client = LLMClient()

# ============ 示例 1：无角色 vs 有角色 ============
print("=" * 60)
print("📌 示例 1：无角色 vs 有角色")
print("=" * 60)

# 无角色
message_no_role = [{"role": "user", "content": "请解释什么是 RAG"}]
try:
    response_no_role = client.chat(message_no_role)
    print("【无角色】")
    print(response_no_role[:100] + "...")
except ValueError as e:
    print(f"❌ 消息格式错误: {e}")
except Exception as e:
    print(f"❌ 其他错误: {e}")
print()

# 有角色
messages_with_role = [
    {
        "role": "system",
        "content": "你是一位 AI 架构师，善于用通俗易懂的方式解释技术概念",
    },
    {"role": "user", "content": "请解释什么是 RAG"},
]
try:
    response_with_role = client.chat(messages_with_role)
    print("【有角色（AI 架构师）】")
    print(response_with_role[:100] + "...")
except ValueError as e:
    print(f"❌ 消息格式错误: {e}")
except Exception as e:
    print(f"❌ 其他错误: {e}")
print()

# ============ 示例 2：有约束 ============
print("=" * 60)
print("📌 示例 2：有约束")
print("=" * 60)

messages_with_constraint = [
    {"role": "system", "content": "你是 AI 助手，回答要简洁、控制在 3 句话以内"},
    {"role": "user", "content": "请解释什么是 embedding"},
]
try:
    response = client.chat(messages_with_constraint)
    print(response)
except ValueError as e:
    print(f"❌ 消息格式错误: {e}")
except Exception as e:
    print(f"❌ 其他错误: {e}")
print()

# ============ 示例 3：指定输出格式 ============
print("=" * 60)
print("📌 示例 3：指定输出格式（JSON）")
print("=" * 60)

messages_with_format = [
    {
        "role": "system",
        "content": """
你是一个信息提取助手。根据用户输入，提取关键信息，并以 JSON 格式返回。

JSON 格式：
{
    "name": "名称",
    "description": "描述",
    "tags": ["标签1", "标签2"]
}
""",
    },
    {
        "role": "user",
        "content": "RAG 是检索增强生成，它结合了检索和生成技术，常用于问答系统。关键词：RAG、检索、生成",
    },
]
try:
    response = client.chat(messages_with_format)
except ValueError as e:
    print(f"❌ 消息格式错误: {e}")
except Exception as e:
    print(f"❌ 其他错误: {e}")
print(response)
