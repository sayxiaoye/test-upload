"""
大模型客户端模块
支持 DeepSeek API 调用
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env
load_dotenv()


class LLMClient:
    """大模型客户端"""

    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        )
        self.default_model = "deepseek-chat"

    def chat(
        self,
        message: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """
        调用大模型

        Args:
            message: 消息列表[{"role":"user","content":"..."}]
            model: 模型名称
            temperature: 温度(0-1)
            max_tokens: 最大输出 token

        Returns:
            模型返回的内容
        """
        model = model or self.default_model

        response = self.client.chat.completions.create(
            model=model,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content
        return str(content)


if __name__ == "__main__":
    client = LLMClient()

    # 测试调用
    message = [{"role": "user", "content": "请用一句话介绍你自己"}]

    response = client.chat(message)
    print("🤖 模型回复:")
    print(response)
