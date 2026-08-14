"""单 Agent 的命令行入口。"""

from __future__ import annotations

import argparse

from openai import OpenAIError

from agent import ask_agent
from config import get_settings
from logging_config import get_logger


logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行具备工具调用能力的 LangChain 单 Agent")
    parser.add_argument("question", nargs="*", help="要交给 Agent 的问题；省略后进入交互模式")
    args = parser.parse_args()

    if not get_settings().deepseek_api_key:
        print("未检测到 DEEPSEEK_API_KEY；请在已注入该环境变量的终端运行本程序。")
        return

    if args.question:
        try:
            print(ask_agent(" ".join(args.question)))
        except (OpenAIError, RuntimeError, ValueError) as error:
            logger.error("单 Agent 执行失败：%s", error)
            print(f"执行失败：{error}")
        return

    print("单 Agent 已启动。输入 exit 或 quit 结束。")
    while True:
        question = input("你：").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            try:
                print(f"助手：{ask_agent(question)}")
            except (OpenAIError, RuntimeError, ValueError) as error:
                logger.error("单 Agent 执行失败：%s", error)
                print(f"执行失败：{error}")


if __name__ == "__main__":
    main()
