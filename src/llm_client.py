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

    # 允许的角色类型
    VALID_ROLES = {"system", "user", "assistant"}

    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        )
        self.default_model = "deepseek-chat"

    def _validate_messages(self, messages: list[dict[str, str]]) -> None:
        """
        验证消息格式是否正确

        Args:
            messages: 消息列表

        Raises:
            ValueError: 消息格式不正确时抛出
        """
        if not messages:
            raise ValueError("消息列表不能为空")

        for i, msg in enumerate(messages):
            # 检查是否是字典
            if not isinstance(msg, dict):
                raise ValueError(f"消息 {i} 必须是字典类型，实际类型: {type(msg)}")

            # 检查 role 字段
            if "role" not in msg:
                raise ValueError(f"消息 {i} 缺少 'role' 字段")

            # 检查 content 字段
            if "content" not in msg:
                raise ValueError(f"消息 {i} 缺少 'content' 字段")

            # 检查 role 值是否合法
            if msg["role"] not in self.VALID_ROLES:
                raise ValueError(
                    f"消息 {i} 的 role 必须是 {', '.join(self.VALID_ROLES)}，"
                    f"当前值: {msg['role']}"
                )

            # 检查 content 类型
            if not isinstance(msg["content"], str):
                raise ValueError(
                    f"消息 {i} 的 content 必须是字符串，"
                    f"实际类型: {type(msg['content'])}"
                )

            # 检查 content 是否为空（非必须，但建议）
            if not msg["content"].strip():
                # 发出警告，但不中断执行
                print(f"⚠️ 警告: 消息 {i} 的 content 为空或只有空白字符")

    def chat(
        self,
        messages: list[dict[str, str]],
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

        Raises:
            ValueError: 消息格式不正确时抛出
            openai.APIError: API 调用失败时抛出
        """
        # ✅ 验证消息格式
        self._validate_messages(messages)
        model = model or self.default_model

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            content = response.choices[0].message.content
            return str(content) if content is not None else ""
        except Exception as e:
            # 可以添加跟详细的错误值
            print(f"❌ API 调用失败: {e}")
            raise


if __name__ == "__main__":
    client = LLMClient()

    # 测试调用
    message = [{"role": "user", "content": "请用一句话介绍你自己"}]

    response = client.chat(message)
    print("🤖 模型回复:")
    print(response)
