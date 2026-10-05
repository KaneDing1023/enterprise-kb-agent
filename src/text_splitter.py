"""文本切分：针对中文语料优化分隔符，并记录切片序号便于溯源。"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

# 中文优先的分隔符序列：段落 -> 换行 -> 句号 -> 感叹/问号 -> 分号 -> 逗号 -> 空格
CHINESE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


def build_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """构建切分器，参数默认取自 .env。"""
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 5)

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CHINESE_SEPARATORS,
        length_function=len,
    )


def split_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """切分文档并补充 chunk_index / chunk_total 元数据。"""
    if not documents:
        return []

    chunks = build_splitter(chunk_size, chunk_overlap).split_documents(documents)
    total = len(chunks)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_total"] = total
    return chunks
