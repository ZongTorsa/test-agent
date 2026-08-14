"""提供给单 Agent 调用的本地工具。"""

from __future__ import annotations

import ast
import operator
from pathlib import Path

from langchain_core.tools import tool

from config import get_settings
from logging_config import get_logger
from vector_store import search_similar_documents


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_ALLOWED_SUFFIXES = {".md", ".txt"}
logger = get_logger(__name__)


def _evaluate_expression(node: ast.AST) -> int | float:
    """递归计算只含四则运算的 AST，禁止执行任意 Python 代码。"""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate_expression(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_expression(node.left)
        right = _evaluate_expression(node.right)
        return _BINARY_OPERATORS[type(node.op)](left, right)
    raise ValueError("只支持数字、括号和 + - * / // % ** 运算符。")


@tool
def calculator(expression: str) -> str:
    """计算数学表达式。适用于加、减、乘、除、整除、取模、乘方和括号。"""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate_expression(tree.body)
        logger.info("计算器执行成功")
        return f"{expression} = {result}"
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError) as error:
        return f"计算失败：{error}"


@tool
def file_reader(file_path: str) -> str:
    """读取当前工作目录内指定的 .md 或 .txt 文档。file_path 必须是相对路径。"""
    root = Path.cwd().resolve()
    candidate = (root / file_path).resolve()

    try:
        candidate.relative_to(root)
    except ValueError:
        return "读取失败：只能读取当前工作目录及其子目录内的文件。"

    if candidate.suffix.lower() not in _ALLOWED_SUFFIXES:
        return "读取失败：只允许读取 .md 或 .txt 文件。"
    if not candidate.is_file():
        return f"读取失败：找不到文件 {file_path}。"

    try:
        content = candidate.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return "读取失败：该文件不是 UTF-8 文本。"
    except OSError as error:
        return f"读取失败：{error}"

    max_chars = get_settings().file_read_max_chars
    if len(content) > max_chars:
        content = content[:max_chars] + "\n\n[文件内容已截断]"
    logger.info("文件读取成功：%s", candidate.relative_to(root))
    return content


@tool
def rag_retriever(query: str) -> str:
    """从本地 RAG 知识库检索与问题相关的资料。只在问题涉及项目知识库时调用。"""
    try:
        documents = search_similar_documents(query)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        logger.warning("知识库检索不可用：%s", error)
        return f"知识库检索不可用：{error}"

    if not documents:
        logger.info("知识库未命中相关资料")
        return "知识库未找到与该问题相关的信息。"

    passages = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "未知来源")
        passages.append(f"[片段 {index}，来源：{source}]\n{document.page_content}")
    return "\n\n".join(passages)


AGENT_TOOLS = [calculator, file_reader]
RAG_AGENT_TOOLS = [calculator, file_reader, rag_retriever]
