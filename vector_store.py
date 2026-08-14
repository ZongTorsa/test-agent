"""基于 Chroma 的本地持久化向量库与语义检索接口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from openai import OpenAIError

from config import get_settings
from logging_config import get_logger
from rag_loader import load_documents, split_documents


COLLECTION_NAME = "knowledge"
logger = get_logger(__name__)
DEFAULT_KNOWLEDGE_DIR = get_settings().knowledge_dir
DEFAULT_PERSIST_DIR = get_settings().chroma_dir


def get_embeddings() -> OpenAIEmbeddings:
    """创建 DashScope text-embedding-v4 嵌入器，不在代码中保存密钥。"""
    settings = get_settings()
    api_key = settings.dashscope_api_key
    if not api_key:
        raise RuntimeError("未检测到 DASHSCOPE_API_KEY（或 dashscope）环境变量。")

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=api_key,
        base_url=settings.dashscope_base_url,
        dimensions=settings.embedding_dimensions,
        chunk_size=settings.embedding_batch_size,
        # DashScope 兼容接口只接受字符串；不要让 LangChain 预先转为 token ID 列表。
        check_embedding_ctx_length=False,
    )


def build_vector_store(
    knowledge_dir: str | Path | None = None,
    persist_dir: str | Path | None = None,
    embeddings: Embeddings | None = None,
    reset: bool = True,
) -> Chroma:
    """加载、切分并写入 Chroma；默认重建指定的本地向量库。"""
    settings = get_settings()
    documents = load_documents(knowledge_dir)
    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("knowledge 目录中没有可入库的 .md 或 .txt 文档。")

    target_dir = Path(persist_dir or settings.chroma_dir)
    embedding_model = embeddings or get_embeddings()
    if reset and target_dir.exists():
        # 只删除本模块自己的 Chroma 集合，不递归删除调用方传入的目录。
        existing_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(target_dir),
            embedding_function=embedding_model,
        )
        existing_store.delete_collection()

    logger.info("写入 %d 个文本片段到 Chroma", len(chunks))

    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=str(target_dir),
    )


def search_similar_documents(
    query: str,
    k: int = 3,
    persist_dir: str | Path | None = None,
    embeddings: Embeddings | None = None,
    max_distance: float | None = None,
) -> list[Document]:
    """查询本地向量库，仅返回距离不超过阈值的文档片段。"""
    settings = get_settings()
    if not query.strip():
        raise ValueError("query 不能为空。")
    if k <= 0:
        raise ValueError("k 必须大于 0。")
    max_distance = max_distance if max_distance is not None else settings.max_distance
    if max_distance < 0:
        raise ValueError("max_distance 不能小于 0。")

    target_dir = Path(persist_dir or settings.chroma_dir)
    if not target_dir.exists():
        raise FileNotFoundError("向量库不存在，请先执行 build_vector_store()。")

    store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(target_dir),
        embedding_function=embeddings or get_embeddings(),
    )
    matches = store.similarity_search_with_score(query, k=k)
    documents = [document for document, distance in matches if distance <= max_distance]
    logger.info("知识库检索完成：命中 %d 个片段", len(documents))
    return documents


def _print_results(results: list[Document]) -> None:
    for index, document in enumerate(results, start=1):
        print(f"\n[{index}] 来源：{document.metadata.get('source', '未知')}")
        print(document.page_content)


def main() -> None:
    """提供手动构建和检索的命令行入口。"""
    parser = argparse.ArgumentParser(description="构建并查询本地 Chroma 知识库")
    parser.add_argument("--rebuild", action="store_true", help="重新加载 knowledge 并构建向量库")
    parser.add_argument("--query", help="待检索的问题")
    parser.add_argument("--k", type=int, default=get_settings().retrieval_k, help="返回片段数量")
    args = parser.parse_args()

    try:
        if args.rebuild:
            store = build_vector_store()
            print(f"向量库构建完成，共 {store._collection.count()} 个文本片段。")
        if args.query:
            _print_results(search_similar_documents(args.query, k=args.k))
    except (OpenAIError, FileNotFoundError, RuntimeError, ValueError) as error:
        logger.error("向量库操作失败：%s", error)
        print(f"向量库操作失败：{error}")
    if not args.rebuild and not args.query:
        parser.print_help()


if __name__ == "__main__":
    main()
