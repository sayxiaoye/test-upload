"""兼容层：保留旧的 src.llm_client 导入路径。"""

from src.rag.llm_client import LLMClient

if __name__ == "__main__":
    client = LLMClient()

    # 测试调用
    message = [{"role": "user", "content": "请用一句话介绍你自己"}]

    response = client.chat(message)
    print("🤖 模型回复:")
    print(response)
