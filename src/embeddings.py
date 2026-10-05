"""Embedding 封装：直接调用阿里云百炼 dashscope SDK。

之所以不直接用 langchain-community 的 DashScopeEmbeddings，是为了让依赖列表
保持在用户给定的最小集合内（只需 dashscope）。

实现 langchain_core 的 Embeddings 接口，可直接传给 langchain-chroma 的 Chroma。
"""

from __future__ import annotations

import logging

import dashscope
from dashscope import TextEmbedding
from langchain_core.embeddings import Embeddings

from config import settings

logger = logging.getLogger(__name__)

# DashScope 文本向量单次请求的文本数量上限。
# 官方规格：text-embedding-v4 / v3 批次大小 = 10，v2 = 25，v1 = 1；
# 这里统一按 10 处理，对全系模型都安全。
MAX_BATCH_SIZE = 10


class DashScopeEmbeddings(Embeddings):
    """基于 DashScope TextEmbedding 的向量化实现。

    Args:
        model: 向量模型名，默认取 ``settings.embedding_model``（text-embedding-v4）。
        api_key: 百炼 API Key，默认取 ``settings.dashscope_api_key``。
        batch_size: 单次请求的文本条数，默认 10（v3/v4 上限）。
        dimension: 输出向量维度；``None`` 表示用模型默认值
            （text-embedding-v4 默认 1024，可选 2048/1536/768/512/256/128/64）。
        use_text_type: 是否开启非对称检索优化。为 ``True`` 时，
            文档走 ``text_type="document"``、查询走 ``text_type="query"``，
            这是 v3/v4 在检索场景下的官方推荐用法（聚类/分类等对称任务应保持 ``False``）。
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        batch_size: int = MAX_BATCH_SIZE,
        dimension: int | None = None,
        use_text_type: bool = False,
    ) -> None:
        self.model = model or settings.embedding_model
        self.api_key = api_key or settings.dashscope_api_key
        self.batch_size = max(1, batch_size)
        self.dimension = dimension
        self.use_text_type = use_text_type

        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY 为空，请先配置 .env 文件。")
        dashscope.api_key = self.api_key

    # ------------------------------------------------------------------
    def _embed(self, texts: list[str], text_type: str | None = None) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = TextEmbedding.call(
                model=self.model,
                input=batch,
                dimension=self.dimension,
                text_type=text_type,
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"DashScope Embedding 调用失败 [{response.status_code}] {response.message}"
                )

            items = sorted(response.output["embeddings"], key=lambda item: item["text_index"])
            vectors.extend(item["embedding"] for item in items)

        return vectors

    # ------------------------------------------------------------------
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化（入库用）。"""
        cleaned = [t.replace("\n", " ").strip() or " " for t in texts]
        return self._embed(cleaned, text_type="document" if self.use_text_type else None)

    def embed_query(self, text: str) -> list[float]:
        """单条查询向量化。"""
        cleaned = [text.replace("\n", " ").strip() or " "]
        return self._embed(cleaned, text_type="query" if self.use_text_type else None)[0]
