"""项目运行参数的集中配置，全部支持环境变量覆盖。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

def _read_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError as error:
        raise ValueError(f"环境变量 {name} 必须是整数。") from error


def _read_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError as error:
        raise ValueError(f"环境变量 {name} 必须是数字。") from error


@dataclass(frozen=True)
class Settings:
    """Agent 和 RAG 模块共享的只读运行参数。"""

    project_root: Path
    knowledge_dir: Path
    chroma_dir: Path
    deepseek_api_key: str | None
    deepseek_model: str
    deepseek_base_url: str
    dashscope_api_key: str | None
    dashscope_base_url: str
    embedding_model: str
    embedding_dimensions: int
    embedding_batch_size: int
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    max_distance: float
    file_read_max_chars: int
    max_reflection_rounds: int
    planner_max_tasks: int
    short_term_memory_turns: int
    memory_retrieval_k: int
    memory_max_distance: float


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """从环境变量构建配置，避免在各业务模块重复读取环境。"""
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env", override=False)
    settings = Settings(
        project_root=project_root,
        knowledge_dir=project_root / os.getenv("KNOWLEDGE_DIR", "knowledge"),
        chroma_dir=project_root / os.getenv("CHROMA_DIR", "chroma_db"),
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY") or os.getenv("dashscope"),
        dashscope_base_url=os.getenv(
            "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-v4"),
        embedding_dimensions=_read_int("EMBEDDING_DIMENSIONS", 1024),
        embedding_batch_size=_read_int("EMBEDDING_BATCH_SIZE", 10),
        chunk_size=_read_int("RAG_CHUNK_SIZE", 500),
        chunk_overlap=_read_int("RAG_CHUNK_OVERLAP", 50),
        retrieval_k=_read_int("RAG_RETRIEVAL_K", 3),
        max_distance=_read_float("RAG_MAX_DISTANCE", 0.9),
        file_read_max_chars=_read_int("FILE_READ_MAX_CHARS", 20_000),
        max_reflection_rounds=_read_int("RAG_MAX_REFLECTION_ROUNDS", 3),
        planner_max_tasks=_read_int("PLANNER_MAX_TASKS", 5),
        short_term_memory_turns=_read_int("SHORT_TERM_MEMORY_TURNS", 5),
        memory_retrieval_k=_read_int("MEMORY_RETRIEVAL_K", 3),
        # 长期事实通常很短，嵌入距离会高于长文档检索，使用独立的宽松阈值。
        memory_max_distance=_read_float("MEMORY_MAX_DISTANCE", 1.8),
    )
    if settings.chunk_size <= 0 or not 0 <= settings.chunk_overlap < settings.chunk_size:
        raise ValueError("RAG_CHUNK_SIZE 和 RAG_CHUNK_OVERLAP 配置无效。")
    if settings.retrieval_k <= 0 or settings.max_distance < 0:
        raise ValueError("RAG_RETRIEVAL_K 或 RAG_MAX_DISTANCE 配置无效。")
    if settings.max_reflection_rounds <= 0:
        raise ValueError("RAG_MAX_REFLECTION_ROUNDS 必须大于 0。")
    if settings.planner_max_tasks <= 0:
        raise ValueError("PLANNER_MAX_TASKS 必须大于 0。")
    if settings.short_term_memory_turns <= 0 or settings.memory_retrieval_k <= 0:
        raise ValueError("短期或长期记忆数量配置必须大于 0。")
    if settings.memory_max_distance < 0:
        raise ValueError("MEMORY_MAX_DISTANCE 不能小于 0。")
    return settings
