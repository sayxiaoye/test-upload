"""
Agent 与工具调用演示
使用Deep Seek 实现 Tool Calling
"""

import json
import os
from collections.abc import Callable

from dotenv import load_dotenv

from src.rag.llm_client import LLMClient

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


# ReAct = Reasoning + Acting（推理 + 行动）
class ReActAgent:
    """支持多步推理的(ReAct) 的 Agent"""

    def __init__(self):
        self.llm = LLMClient()
        self.tools = TOOLS
        self.function_map: dict[str, Callable[..., str]] = FUNCTION_MAP
        # ReAct的系统提示词，引导模型进行“思考-行动-观察”循环
        self.system_prompt = """你是一个能使用工具的智能助手。请严格按照以下格式进行推理和行动：

问题: 用户的问题
思考: 你需要思考下一步该做什么，是否需要使用工具。
行动: 如果需要使用工具，请输出: 行动: 工具名称({"参数名": "参数值"})
观察: 系统会返回工具执行的结果。
... (你可以重复“思考-行动-观察”多次) ...
思考: 我现在知道最终答案了。
关键字【最终回答:】后面加上，你的最终回答

注意:
- 关键字【最终回答:】只出现在最后一步。
- 如果你不需要使用工具，可以直接给出“最终回答”。
"""

    def chat(self, user_input: str, max_steps: int = 5) -> str:
        """
        执行ReAct的工作流
        """
        # 初始化消息史
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]
        observations = []
        final_answer = "抱歉，我无法完成这个任务"

        for step in range(max_steps):
            # 1 推理reasoning: 调用LLM获取下一步思考
            response = self.llm.client.chat.completions.create(
                model=os.getenv("DEEPSEEK_MODEL_FLASH", "deepseek-v4-flash"),
                messages=messages,  # type: ignore
                temperature=0.3,
            )
            thought = response.choices[0].message.content or ""

            # 2 判断是否已完成，如果LLM输出包含“最终完成”，则提取并结束循环
            if "最终回答" in thought:
                # 找到“最终回答:”后面的内容
                parts = thought.split("最终回答:")
                final_answer = parts[-1].strip() if len(parts) > 1 else thought
                break

            # 3. 行动acting：解析并执行工具
            action = self._parse_action(thought)
            if action:
                tool_name, tool_args = action
                print(f"🔧 ReAct 步骤 {step + 1}: 调用工具 {tool_name}({tool_args})")
                result = self._execute_tool(tool_name, tool_args)
                observation = f"观察: {result}"
            else:
                # 如果LLM没有输出有效的“Acting”， 则手动生成一个“observation”，让对话继续
                observation = "观察：无法解析行动，请重新思考并输出正确的行动。"

            # 将reasoning acting observation追加到消息历史，供下一步推理使用
            messages.append({"role": "assistant", "content": thought})
            messages.append({"role": "user", "content": observation})
            observations.append(observation)

        # 4. 生成最终回答
        return final_answer

    def _parse_action(self, thought: str) -> tuple[str, dict] | None:
        """
        从 LLM 的输出中解析出工具名称和参数
        期望的格式： 行动: 工具名称（参数）
        例如： 行动: get_weather({"city": "北京"})
        """
        if "行动:" not in thought:
            return None

        # 提取acting之后的部分
        action_line = thought.rsplit("行动:", maxsplit=1)[-1].strip()
        # 寻找括号来分割工具名和参数
        if "(" not in action_line or ")" not in action_line:
            return None

        tool_name = action_line.split("(")[0].strip()
        args_str = action_line.split("(")[1].split(")")[0].strip()

        try:
            # 尝试将参数解析为JSON
            tool_args = json.loads(args_str)
        except json.JSONDecodeError:
            # 如果解析失败，可能是参数格式不标准，尝试构建一个空字典或返回 None
            print(f"⚠️ 参数解析失败: {args_str}")
            return None

        return tool_name, tool_args

    def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """执行具体的工具 函数"""
        func = self.function_map.get(tool_name)
        if func:
            try:
                return func(**tool_args)
            except Exception as e:
                return f"❌ 工具执行错误: {e}"
        else:
            return f"❌ 未知工具: {tool_name}"


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

    # for query in test_inputs:
    #     print(f"\n📌 用户: {query}")
    #     response = agent.chat(query)
    #     print(f"🤖 Agent: {response}")
    #     print("-" * 40)

    print("\n" + "=" * 60)
    print("🧠 ReAct Agent 多步推理演示")
    print("=" * 60)

    react_agent = ReActAgent()
    complex_query = "北京今天天气怎么样？"
    print(f"\n📌 用户: {complex_query}")
    response = react_agent.chat(complex_query)
    print(f"🧠 ReAct Agent: {response}")
