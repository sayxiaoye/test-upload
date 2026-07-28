"""
Agent 与工具调用演示
使用Deep Seek 实现 Tool Calling
"""

import json
import os
from collections.abc import Callable

from dotenv import load_dotenv

from src.llm_client import LLMClient

# 加载 .env
load_dotenv()


# ============ 定义工具 ============
def get_weather(city: str) -> str:
    """获取城市天气（模拟）"""
    # 真实场景可调用天气 API
    weather_data = {
        "北京": "🌤️ 25°C，晴",
        "上海": "🌧️ 22°C，小雨",
        "深圳": "☀️ 30°C，炎热",
        "杭州": "⛅ 24°C，多云",
    }
    return weather_data.get(city, f"❌ 未找到 {city} 的天气信息")


def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        # 安全起见， 只允许简单运算
        allowed = ["+", "-", "*", "/", "(", ")"]
        if not all(c.isdigit() or c.isspace() or c in allowed for c in expression):
            return "❌ 只支持基本数学运算"
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"❌ 计算错误: {e}"


# ============ 工具定义（给 LLM 看的描述） ============
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获得指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如：北京、上海、深圳",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式的结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如：12+34 或 (5+3)*2",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]

# 工具名称 → 实际函数的映射
FUNCTION_MAP = {
    "get_weather": get_weather,
    "calculate": calculate,
}


class Agent:
    """Agent 智能体"""

    def __init__(self):
        self.llm = LLMClient()
        self.tools = TOOLS
        self.function_map: dict[str, Callable[..., str]] = FUNCTION_MAP
        self.messages = [
            {
                "role": "system",
                "content": "你是一个智能助手，当用户需要插叙天气或者计算时，调用响应的工具。",
            }
        ]

    def chat(self, user_input: str) -> str:
        """
        处理用户输入，自动决定是否调用工具
        """
        self.messages.append({"role": "user", "content": user_input})

        # 第一次调用 LLM
        response = self.llm.client.chat.completions.create(  # 直接使用 OpenAI 客户端
            model=os.getenv("DEEPSEEK_MODEL_FLASH", "deepseek-v4-flash"),
            messages=self.messages,  # type: ignore
            tools=self.tools,  # type: ignore
            tool_choice="auto",
        )
        """
        模型会返回一个响应，其中可能包含 response.choices[0].message.tool_calls 字段
        这是一个列表，每个元素包含：
        function.name：要调用的函数名。
        function.arguments：一个 JSON 字符串，包含调用该函数所需的实参数。
        """
        message = response.choices[0].message

        # 检查是否有工具调用
        if message.tool_calls:
            # 执行工具
            tool_call = message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"🔧 调用工具: {tool_name}({tool_args})")

            # 执行对应的函数
            func = self.function_map.get(tool_name)
            result = func(**tool_args) if func else f"❌ 未知工具: {tool_name}"

            # 将工具结果返回给 LLM
            self.messages.append(message)  # 添加 LLM 的响应   # type: ignore
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                }
            )

            # 第二次调用LLM 让 LLM 根据工具结果生成最终回答
            final_response = self.llm.client.chat.completions.create(
                model=os.getenv("DEEPSEEK_MODEL_FLASH", "deepseek-v4-flash"),
                messages=self.messages,  # type: ignore
            )

            answer = final_response.choices[0].message.content or ""
            self.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )
            return answer

        # 不需要工具，直接返回
        answer = message.content or ""
        self.messages.append({"role": "assistant", "content": answer})
        return answer


if __name__ == "__main__":
    agent = Agent()

    test_inputs = [
        "北京今天天气怎么样？",
        "帮我算一下 123 + 456 等于多少？",
        "深圳天气如何？",
        "计算 (25 + 17) * 3 的结果",
        "你好，今天有什么新闻？",  # 不需要工具
    ]

    print("=" * 60)
    print("🤖 Agent 智能体演示")
    print("=" * 60)

    for query in test_inputs:
        print(f"\n📌 用户: {query}")
        response = agent.chat(query)
        print(f"🤖 Agent: {response}")
        print("-" * 40)
