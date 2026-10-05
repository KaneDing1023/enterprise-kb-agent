"""检索器：把用户问题转成向量，从 Chroma 召回 Top-K 相关片段。"""

from __future__ import annotations

import logging

from langchain_core.documents import Document

from config import settings
from src.vector_store import get_vector_store

logger = logging.getLogger(__name__)


def search(query: str, k: int | None = None, score_threshold: float | None = None) -> list[Document]:
    """相似度检索。

    Args:
        query: 用户问题。
        k: 召回数量，默认取 .env 中的 TOP_K。
        score_threshold: 相似度下限（0~1，越高越严格）；None 表示不过滤。
    """
    query = (query or "").strip()
    if not query:
        return []

    k = k or settings.top_k
    store = get_vector_store()

    if score_threshold is None:
        return store.similarity_search(query, k=k)

    pairs = store.similarity_search_with_relevance_scores(query, k=k)
    return [doc for doc, score in pairs if score >= score_threshold]


def search_with_scores(query: str, k: int | None = None) -> list[tuple[Document, float]]:
    """返回 (片段, 相似度分数) 列表，用于调试与可视化。"""
    query = (query or "").strip()
    if not query:
        return []
    return get_vector_store().similarity_search_with_relevance_scores(query, k=k or settings.top_k)


def build_context(documents: list[Document]) -> tuple[str, list[dict]]:
    """把召回片段拼成带编号的上下文，并输出引用来源列表。"""
    context_parts: list[str] = []
    sources: list[dict] = []

    for index, doc in enumerate(documents, start=1):
        meta = doc.metadata or {}
        name = meta.get("file_name", "未知文件")
        page = meta.get("page")
        label = f"{name}（第 {page} 页）" if page else name

        context_parts.append(f"[{index}] 来源：{label}\n{doc.page_content}")
        sources.append(
            {
                "index": index,
                "file_name": name,
                "page": page,
                "source": meta.get("source", ""),
                "preview": doc.page_content[:120],
            }
        )

    return "\n\n---\n\n".join(context_parts), sources
