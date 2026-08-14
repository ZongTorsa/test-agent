"""知识库文档的加载与切分功能。"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings
from logging_config import get_logger

SUPPORTED_SUFFIXES = {".md", ".txt"}
logger = get_logger(__name__)


def load_documents(knowledge_dir: str | Path | None = None) -> list[Document]:
    """递归读取知识库中的 UTF-8 Markdown 和文本文件。"""
    root = Path(knowledge_dir or get_settings().knowledge_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"知识库目录不存在：{root}")

    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            logger.warning("跳过非 UTF-8 文档：%s（%s）", path, error)
            continue
        except OSError as error:
            logger.warning("跳过无法读取的文档：%s（%s）", path, error)
            continue

        documents.append(
            Document(
                page_content=content,
                metadata={"source": path.relative_to(root).as_posix()},
            )
        )
    logger.info("已加载 %d 篇知识库文档", len(documents))
    return documents


def split_documents(
    documents: list[Document], chunk_size: int | None = None, chunk_overlap: int | None = None
) -> list[Document]:
    """按自然段和字符边界切分文档，保留来源元数据。"""
    settings = get_settings()
    chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size 必须大于 0，且 chunk_overlap 应在 [0, chunk_size) 内。")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", " ", ""],
    )
    return splitter.split_documents(documents)
