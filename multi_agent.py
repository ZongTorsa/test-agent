"""多智能体工作流统一入口：Planner → Executor → 汇总。"""

from __future__ import annotations

import argparse
import sys

from openai import OpenAIError

from executor_agent import ExecutionResultMessage, execute_task, summarize_results
from logging_config import get_logger
from memory import MultiAgentMemory
from planner_agent import PlanMessage, plan_request


logger = get_logger(__name__)


def _safe_print(value: str) -> None:
    """兼容 Windows 本地代码页，避免个别模型符号导致整个流程中断。"""
    try:
        print(value)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        print(value.encode(encoding, errors="replace").decode(encoding))


def run_multi_agent_workflow(
    user_request: str, memory: MultiAgentMemory | None = None
) -> tuple[PlanMessage, list[ExecutionResultMessage], str]:
    """通过消息传递执行完整协同工作流，并自动维护双层记忆。"""
    workflow_memory = memory or MultiAgentMemory()
    memory_context = workflow_memory.context_for(user_request)
    plan = plan_request(user_request, memory_context=memory_context)
    results: list[ExecutionResultMessage] = []
    for task in plan.tasks:
        # PlanMessage.tasks 是 Planner 到 Executor 的消息通道；结果列表是反向通道。
        results.append(execute_task(task, memory_context=memory_context))
    summary = summarize_results(plan.user_request, results, memory_context=memory_context)
    workflow_memory.complete_turn(plan.user_request, summary)
    logger.info("多智能体工作流完成：%d 个任务", len(results))
    return plan, results, summary


def main() -> None:
    """执行单次多智能体协同任务。"""
    parser = argparse.ArgumentParser(description="运行 Planner 与 Executor 协同工作流")
    parser.add_argument("request", nargs="+", help="需要拆解并执行的复杂需求")
    args = parser.parse_args()

    memory = MultiAgentMemory()
    try:
        plan, results, summary = run_multi_agent_workflow(" ".join(args.request), memory)
    except (OpenAIError, RuntimeError, ValueError) as error:
        logger.error("多智能体工作流失败：%s", error)
        _safe_print(f"执行失败：{error}")
        return

    _safe_print("执行计划：")
    for task in plan.tasks:
        _safe_print(f"{task.task_id}. {task.instruction}")
    _safe_print("\n任务结果：")
    for item in results:
        _safe_print(f"\n[{item.task_id}] {item.result}")
    _safe_print(f"\n汇总：\n{summary}")


if __name__ == "__main__":
    main()
