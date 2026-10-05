"""RAG 问答链：检索 -> 拼上下文 -> 调用通义千问生成回答。

LLM 同样直接走 dashscope SDK（Generation 接口），无需额外依赖。
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from dashscope import Generation

from config import settings
from src.retriever import build_context, search

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是一名严谨的企业知识库助手。你只能依据用户提供的「参考资料」作答，"
    "不得使用参考资料之外的任何知识，也不得编造事实。"
)

USER_PROMPT_TEMPLATE = """请依据下面的「参考资料」回答「用户问题」。

作答要求：
1. 只使用参考资料中的信息，不得编造、不得外推；
2. 如果参考资料不足以回答，请直接回复：根据现有资料无法回答该问题；
3. 回答要条理清晰，可使用列表；关键结论后标注引用编号，如 [1]、[2]；
4. 不要在回答中复述本段要求。

【参考资料】
{context}

【用户问题】
{question}

【回答】
"""


def _build_messages(question: str, context: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(context=context, question=question)},
    ]


def _prepare(question: str, k: int | None = None, score_threshold: float | None = None):
    documents = search(question, k=k, score_threshold=score_threshold)
    if not documents:
        return None, None, []
    context, sources = build_context(documents)
    return context, sources, documents


NO_RESULT_ANSWER = "知识库中暂未检索到与问题相关的内容。请先在左侧上传文档并入库，或换一种问法。"


def answer(
    question: str,
    k: int | None = None,
    score_threshold: float | None = None,
    temperature: float = 0.2,
) -> dict:
    """同步问答，返回 {"answer", "sources", "documents"}。"""
    question = (question or "").strip()
    if not question:
        return {"answer": "请输入问题。", "sources": [], "documents": []}

    context, sources, documents = _prepare(question, k, score_threshold)
    if context is None:
        return {"answer": NO_RESULT_ANSWER, "sources": [], "documents": []}

    response = Generation.call(
        model=settings.llm_model,
        messages=_build_messages(question, context),
        result_format="message",
        temperature=temperature,
    )
    if response.status_code != 200:
        raise RuntimeError(f"通义千问调用失败 [{response.status_code}] {response.message}")

    text = response.output.choices[0].message.content
    return {"answer": text, "sources": sources, "documents": documents}


def stream_answer(
    question: str,
    k: int | None = None,
    score_threshold: float | None = None,
    temperature: float = 0.2,
) -> Iterator[tuple[str, object]]:
    """流式问答，依次 yield ("sources", list) / ("delta", str) / ("end", str)。"""
    question = (question or "").strip()
    if not question:
        yield "delta", "请输入问题。"
        return

    context, sources, _ = _prepare(question, k, score_threshold)
    if context is None:
        yield "sources", []
        yield "delta", NO_RESULT_ANSWER
        return

    yield "sources", sources

    responses = Generation.call(
        model=settings.llm_model,
        messages=_build_messages(question, context),
        result_format="message",
        temperature=temperature,
        stream=True,
        incremental_output=True,
    )

    full_text: list[str] = []
    for response in responses:
        if response.status_code != 200:
            raise RuntimeError(f"通义千问调用失败 [{response.status_code}] {response.message}")
        delta = response.output.choices[0].message.content or ""
        if delta:
            full_text.append(delta)
            yield "delta", delta

    yield "end", "".join(full_text)
