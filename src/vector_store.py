"""向量库：基于 Chroma 的持久化存储与写入。"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import settings
from src.embeddings import DashScopeEmbeddings

logger = logging.getLogger(__name__)

_embedding_singleton: DashScopeEmbeddings | None = None


def get_embedding() -> DashScopeEmbeddings:
    """全局复用同一个 Embedding 实例。"""
    global _embedding_singleton
    if _embedding_singleton is None:
        _embedding_singleton = DashScopeEmbeddings()
    return _embedding_singleton


def get_vector_store(embedding: DashScopeEmbeddings | None = None) -> Chroma:
    """打开（不存在则创建）本地 Chroma 集合。"""
    settings.ensure_dirs()
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embedding or get_embedding(),
        persist_directory=str(settings.persist_dir),
    )


def add_documents(documents: list[Document], batch_size: int = 64) -> int:
    """分批写入向量库，返回写入的分片数量。

    Chroma 默认对同一 id 去重；这里按 内容hash 生成稳定 id，重复入库不会产生脏数据。
    """
    if not documents:
        return 0

    store = get_vector_store()
    ids = [_stable_id(doc) for doc in documents]

    written = 0
    for start in range(0, len(documents), batch_size):
        store.add_documents(
            documents=documents[start : start + batch_size],
            ids=ids[start : start + batch_size],
        )
        written += len(documents[start : start + batch_size])
        logger.info("已写入 %d/%d 个分片", written, len(documents))

    return written


def _stable_id(doc: Document) -> str:
    import hashlib

    meta = doc.metadata or {}
    raw = f"{meta.get('source', '')}|{meta.get('page', '')}|{meta.get('block', '')}|{doc.page_content}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def count_documents() -> int:
    """当前集合内的分片总数。"""
    try:
        return get_vector_store()._collection.count()
    except Exception:
        return 0


def reset_collection() -> None:
    """清空集合（危险操作，会删除全部已入库内容）。"""
    store = get_vector_store()
    try:
        store.reset_collection()
    except AttributeError:
        store.delete_collection()
    logger.warning("已清空集合 %s", settings.collection_name)


def collection_info() -> dict:
    return {
        "collection": settings.collection_name,
        "persist_dir": str(Path(settings.persist_dir)),
        "chunks": count_documents(),
        "embedding_model": settings.embedding_model,
    }
