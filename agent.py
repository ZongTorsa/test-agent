"""创建具备工具调用能力的 LangChain 单 Agent。"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from config import get_settings
from logging_config import get_logger
from tools import AGENT_TOOLS


SYSTEM_PROMPT = """你是一个中文助手，可以自主决定是否调用工具。
需要精确进行四则运算时使用 calculator；需要读取当前项目中的 .md 或 .txt 文档时使用 file_reader。
不要臆造工具输出。获得工具结果后，用简洁中文整合并回答用户。"""
logger = get_logger(__name__)


def create_chat_model() -> ChatOpenAI:
    """使用环境变量中的 DeepSeek 凭据创建可复用的对话模型。"""
    settings = get_settings()
    api_key = settings.deepseek_api_key
    if not api_key:
        raise RuntimeError("未检测到 DEEPSEEK_API_KEY 环境变量。")

    logger.debug("创建 DeepSeek 对话模型：%s", settings.deepseek_model)
    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=api_key,
        base_url=settings.deepseek_base_url,
        temperature=0,
        # DeepSeek 文档将工具调用示例置于非思考模式，避免工具消息兼容问题。
        extra_body={"thinking": {"type": "disabled"}},
    )


def create_local_agent():
    """创建带计算器和文件读取工具的单 Agent。"""
    return create_agent(
        model=create_chat_model(), tools=AGENT_TOOLS, system_prompt=SYSTEM_PROMPT
    )


def ask_agent(question: str) -> str:
    """向 Agent 提问，并返回整合了工具结果的最终文本。"""
    if not question.strip():
        raise ValueError("问题不能为空。")
    agent = create_local_agent()
    logger.info("单 Agent 开始处理问题")
    result = agent.invoke({"messages": [{"role": "user", "content": question.strip()}]})
    return str(result["messages"][-1].content)
