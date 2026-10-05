"""`src/kb.py` 单元测试：入库 / 检索 / 去重 / 元数据 / 重排序 / 逐段溯源，
**不联网、不消耗百炼额度**。

做法是注入一个确定性的伪 Embedding（字符袋向量）+ 伪 Reranker，
把 Chroma 的读写链路与「粗排 → 精排 → 溯源」完整跑一遍。

约定：本文件里的 ``search()`` / ``chat()`` / ``chat_stream()`` 三个测试入口
**默认关闭重排序**，避免单元测试真的去调百炼；需要验证重排的用例显式传
``rerank=True`` 并把伪 Reranker 打进 ``kb.get_reranker``。

运行：
    python tests/test_kb.py
或（若已安装 pytest）：
    pytest tests -q
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402

import src.kb as kb  # noqa: E402
import src.reranker as reranker_mod  # noqa: E402
from src.citations import annotate_answer, annotate_partial, build_segments  # noqa: E402
from src.kb import add_docs_to_db, count_docs, db_info  # noqa: E402

DIM = 128


@contextmanager
def temp_db():
    """提供一个独立的临时向量库目录。

    Windows 上 Chroma 进程内仍持有 ``data_level0.bin`` / sqlite 的文件句柄，
    ``TemporaryDirectory`` 在退出时会因为删不掉而抛 WinError 32，
    所以这里用 ``mkdtemp`` + 忽略错误的清理：就算残留也只是系统临时目录里的垃圾。
    """
    root = Path(tempfile.mkdtemp(prefix="kb_test_"))
    try:
        yield root / "chroma_kb"
    finally:
        shutil.rmtree(root, ignore_errors=True)


class FakeEmbeddings(Embeddings):
    """确定性伪向量：按字符做 bag-of-characters，再 L2 归一化。

    不用内置 ``hash()``（受 PYTHONHASHSEED 影响，跨进程不稳定），
    改用 ``ord()``，保证同一段文本永远得到同一个向量。
    """

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * DIM
        for char, count in Counter(text).items():
            vector[ord(char) % DIM] += float(count)
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeReranker:
    """确定性伪重排器：按「查询字符在片段中的命中率」打分。

    用来验证「粗排顺序 → 精排顺序」的整条链路（字段回填、重新编号、截断），
    不产生任何网络调用。分数恒为 0~1，且与查询字符集直接相关，便于断言。
    传入 ``score_fn`` 可以自定义打分，用来构造「精排把粗排靠后的片段捞到第一」的场景。
    """

    def __init__(self, model: str | None = None, score_fn=None) -> None:
        self.model = model or "fake-rerank"
        self.score_fn = score_fn
        self.calls: list[dict] = []

    def _score(self, query: str, text: str) -> float:
        if self.score_fn is not None:
            return float(self.score_fn(query, text))
        chars = set(query or "")
        if not chars:
            return 0.0
        return sum(1 for ch in chars if ch in (text or "")) / len(chars)

    def rerank_hits(self, query: str, hits: list[dict], top_n: int | None = None) -> list[dict]:
        self.calls.append({"query": query, "candidates": len(hits), "top_n": top_n})
        scored = [
            (self._score(query, hit.get("content") or ""), index, hit)
            for index, hit in enumerate(hits)
        ]
        # 分数降序，同分时保留粗排原序，保证结果稳定可断言
        scored.sort(key=lambda item: (-item[0], item[1]))
        limit = len(scored) if top_n is None else max(1, min(int(top_n), len(scored)))

        results: list[dict] = []
        for rank, (score, _, hit) in enumerate(scored[:limit], 1):
            item = dict(hit)
            item["recall_rank"] = hit.get("rank")
            item["recall_total"] = len(hits)
            item["vector_score"] = hit.get("score")
            item["rerank_score"] = round(score, 4)
            item["rerank_model"] = self.model
            item["score"] = round(score, 4)
            item["rank"] = rank
            results.append(item)
        return results


class BoomReranker:
    """一调用就炸的伪重排器，用于验证「重排失败必须降级、不能打断问答」。"""

    def rerank_hits(self, query, hits, top_n=None):  # pragma: no cover —— 必然抛错
        raise RuntimeError("模拟重排服务不可用")


@contextmanager
def patched_reranker(reranker):
    """把伪重排器打进 ``kb.get_reranker``（kb 内部按模块属性查找，便于打桩）。"""
    original = kb.get_reranker
    kb.get_reranker = lambda model=None: reranker  # noqa: ARG005 —— 打桩忽略模型名
    try:
        yield reranker
    finally:
        kb.get_reranker = original


# ----------------------------------------------------------------------
# 测试入口：默认关闭重排序（单元测试不碰网络），需要时显式传 rerank=True
# ----------------------------------------------------------------------
def search(query: str, **kwargs):
    kwargs.setdefault("rerank", False)
    return kb.search_from_db(query, **kwargs)


def chat(query: str, **kwargs):
    kwargs.setdefault("rerank", False)
    return kb.kb_chat(query, **kwargs)


def chat_stream(query: str, **kwargs):
    kwargs.setdefault("rerank", False)
    return kb.kb_chat_stream(query, **kwargs)


def _docs() -> list[Document]:
    """构造一批带「文件名 + 页码」元数据的切分结果。"""
    return [
        Document(
            page_content="员工累计工作满一年不满十年的，年假为五天；满十年不满二十年的，年假为十天。",
            metadata={"file_name": "员工手册.pdf", "source": "data/raw/员工手册.pdf",
                      "page": 2, "chunk_index": 0},
        ),
        Document(
            page_content="年假申请需要在系统提交，经直属主管审批后生效，跨年度最多结转三天。",
            metadata={"file_name": "员工手册.pdf", "source": "data/raw/员工手册.pdf",
                      "page": 3, "chunk_index": 1},
        ),
        Document(
            page_content="差旅报销需在行程结束后五个工作日内提交发票，逾期需额外说明原因。",
            metadata={"file_name": "财务制度.docx", "source": "data/raw/财务制度.docx",
                      "page": None, "chunk_index": 0},
        ),
    ]


def test_add_and_search_roundtrip() -> None:
    embedding = FakeEmbeddings()
    with temp_db() as db:

        written = add_docs_to_db(_docs(), persist_dir=db, embedding=embedding)
        assert written == 3, written
        assert count_docs(persist_dir=db, embedding=embedding) == 3

        hits = search("年假有多少天？", top_k=3, persist_dir=db, embedding=embedding)
        assert len(hits) == 3, hits

        # 返回结构：内容 + 元数据（文件名 / 页码）
        top = hits[0]
        for key in ("rank", "content", "score", "file_name", "page", "source", "citation"):
            assert key in top, f"缺少字段 {key}"
        assert top["rank"] == 1
        assert top["file_name"] == "员工手册.pdf"
        assert top["page"] in (2, 3)
        assert top["citation"] == f"员工手册.pdf（第 {top['page']} 页）"
        assert "年假" in top["content"]

        # 分数降序
        scores = [h["score"] for h in hits]
        assert scores == sorted(scores, reverse=True), scores


def test_top_k_limit() -> None:
    embedding = FakeEmbeddings()
    with temp_db() as db:
        add_docs_to_db(_docs(), persist_dir=db, embedding=embedding)

        assert len(search("报销流程", top_k=1, persist_dir=db, embedding=embedding)) == 1

        # top_k 超过库内总数时应自动收敛，不报错
        hits = search("报销流程", top_k=99, persist_dir=db, embedding=embedding)
        assert len(hits) == 3

        # 默认 top_k = 3
        assert len(search("年假", persist_dir=db, embedding=embedding)) == 3


def test_metadata_filter_with_threshold() -> None:
    embedding = FakeEmbeddings()
    with temp_db() as db:
        add_docs_to_db(_docs(), persist_dir=db, embedding=embedding)

        # 不带阈值时，最相关的一条应命中财务制度（该类文档没有页码）
        baseline = search("差旅报销发票", top_k=3, persist_dir=db, embedding=embedding)
        assert baseline and baseline[0]["file_name"] == "财务制度.docx", baseline
        assert baseline[0]["page"] is None
        assert baseline[0]["citation"] == "财务制度.docx"

        top_score = baseline[0]["score"]

        # 阈值低于最高分：保留该条
        kept = search(
            "差旅报销发票", top_k=3, score_threshold=top_score - 0.001,
            persist_dir=db, embedding=embedding,
        )
        assert kept and kept[0]["file_name"] == "财务制度.docx", kept

        # 阈值高于最高分：全部被过滤掉
        assert (
            search(
                "差旅报销发票", top_k=3, score_threshold=top_score + 0.001,
                persist_dir=db, embedding=embedding,
            )
            == []
        )


def test_reingest_is_idempotent() -> None:
    embedding = FakeEmbeddings()
    with temp_db() as db:
        add_docs_to_db(_docs(), persist_dir=db, embedding=embedding)
        add_docs_to_db(_docs(), persist_dir=db, embedding=embedding)
        assert count_docs(persist_dir=db, embedding=embedding) == 3, "重复入库不应产生脏数据"


def test_persistence_across_clients() -> None:
    """换一个客户端重新打开同一目录，数据应当仍在（验证持久化生效）。"""
    embedding = FakeEmbeddings()
    with temp_db() as db:
        add_docs_to_db(_docs(), persist_dir=db, embedding=embedding)

        from src.kb import get_vector_store

        store = get_vector_store(persist_dir=db, embedding=embedding)
        assert store._collection.count() == 3


def test_edge_cases() -> None:
    embedding = FakeEmbeddings()
    with temp_db() as db:

        assert add_docs_to_db([], persist_dir=db, embedding=embedding) == 0
        assert add_docs_to_db(None, persist_dir=db, embedding=embedding) == 0
        assert (
            add_docs_to_db(
                [Document(page_content="   ", metadata={"file_name": "空.txt"})],
                persist_dir=db,
                embedding=embedding,
            )
            == 0
        )

        # 空库 / 空问题：返回空列表而不是抛异常
        assert search("年假", persist_dir=db, embedding=embedding) == []
        add_docs_to_db(_docs(), persist_dir=db, embedding=embedding)
        assert search("", persist_dir=db, embedding=embedding) == []
        assert search("   ", persist_dir=db, embedding=embedding) == []


def test_default_persist_dir_is_chroma_kb() -> None:
    """默认持久化目录应当就是项目根目录下的 ./chroma_kb。"""
    info = db_info()
    assert info["persist_dir"] == str(PROJECT_ROOT / "chroma_kb"), info
    assert info["embedding_model"] == "text-embedding-v4", info
    assert info["distance"] == "cosine"


def test_kb_chat_empty_library_returns_fixed_answer() -> None:
    """知识库为空时：不调用大模型，直接返回固定话术。"""
    embedding = FakeEmbeddings()

    def boom(*args, **kwargs):  # pragma: no cover —— 不应被触发
        raise AssertionError("知识库为空时不应调用大模型")

    original = kb._call_llm
    kb._call_llm = boom
    try:
        with temp_db() as db:
            result = chat("年假有多少天？", persist_dir=db, embedding=embedding)
    finally:
        kb._call_llm = original

    assert result["found"] is False
    assert result["sources"] == []
    assert result["hits"] == []
    assert "无法从知识库中获取相关信息" in result["answer"]


def test_kb_chat_returns_answer_and_sources() -> None:
    """有命中时：走「检索 -> 拼上下文 -> 调模型」链路，返回答案与引用来源。"""
    embedding = FakeEmbeddings()
    captured: dict = {}

    def fake_llm(messages, temperature=0.2):
        captured["messages"] = messages
        captured["temperature"] = temperature
        return "根据资料，年假为五天。[1]"

    original = kb._call_llm
    kb._call_llm = fake_llm
    try:
        with temp_db() as db:
            add_docs_to_db(_docs(), persist_dir=db, embedding=embedding)
            result = chat("年假有多少天？", top_k=3, persist_dir=db, embedding=embedding)
    finally:
        kb._call_llm = original

    # 1) 返回结构
    assert result["found"] is True
    assert result["answer"] == "根据资料，年假为五天。[1]"
    assert result["query"] == "年假有多少天？"
    assert len(result["hits"]) == 3
    assert len(result["sources"]) == 3

    top = result["sources"][0]
    for key in ("index", "file_name", "page", "source", "citation", "score", "preview"):
        assert key in top, f"引用来源缺少字段 {key}"
    assert top["index"] == 1
    assert top["citation"] == result["hits"][0]["citation"]
    assert top["file_name"] == result["hits"][0]["file_name"]

    # 2) 提示词：system 为严格防幻觉提示，user 里同时带上了参考资料与问题
    messages = captured["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == kb.KB_SYSTEM_PROMPT
    assert "禁止编造" in messages[0]["content"]
    assert "无法从知识库中获取相关信息" in messages[0]["content"]

    user_msg = messages[1]["content"]
    assert "年假有多少天？" in user_msg
    assert "员工手册.pdf" in user_msg        # 参考资料里带来源标签
    assert "[1]" in user_msg                 # 参考资料按编号标注
    assert messages[0]["content"] != user_msg


def test_kb_chat_empty_query() -> None:
    embedding = FakeEmbeddings()
    with temp_db() as db:
        result = chat("   ", persist_dir=db, embedding=embedding)
    assert result["found"] is False
    assert "请输入问题" in result["answer"]


def test_live_dashscope_roundtrip() -> None:
    """真实调用百炼 text-embedding-v4 + gte-rerank 的联调测试；未配置 Key 时自动跳过。"""
    from config import settings

    if not settings.is_configured():
        print("    (跳过：.env 中未配置有效的 DASHSCOPE_API_KEY)")
        return

    with temp_db() as db:
        written = add_docs_to_db(_docs(), persist_dir=db)
        assert written == 3, written

        # 不重排：纯向量检索
        baseline = search("年假几天？", top_k=2, persist_dir=db)
        assert baseline and baseline[0]["file_name"] == "员工手册.pdf", baseline
        assert baseline[0]["rerank_score"] is None
        assert baseline[0]["score"] is not None

        # 真实走一遍 gte-rerank（候选 3 条 > top_k 2 条，会触发重排）
        reranked = search("年假几天？", top_k=2, rerank=True, recall_k=3, persist_dir=db)
        assert len(reranked) == 2, reranked
        assert all(hit["rerank_score"] is not None for hit in reranked), reranked
        assert [hit["rank"] for hit in reranked] == [1, 2]
        assert reranked[0]["file_name"] == "员工手册.pdf", reranked


def test_kb_chat_stream_events_and_answer() -> None:
    """流式问答：先抛 sources，再逐段抛 delta，最后抛完整结果。"""
    embedding = FakeEmbeddings()
    captured: dict = {}

    def fake_stream(messages, temperature=0.2):
        captured["messages"] = messages
        for piece in ("根据资料，", "年假为五天。", "[1]"):
            yield piece

    original = kb._stream_llm
    kb._stream_llm = fake_stream
    try:
        with temp_db() as db:
            add_docs_to_db(_docs(), persist_dir=db, embedding=embedding)
            events = list(
                chat_stream("年假有多少天？", top_k=3, persist_dir=db, embedding=embedding)
            )
    finally:
        kb._stream_llm = original

    # 1) 事件顺序：sources 在最前，end 在最后
    kinds = [kind for kind, _ in events]
    assert kinds[0] == "sources", kinds
    assert kinds[-1] == "end", kinds
    assert kinds.count("sources") == 1, kinds

    # 2) 来源先于回答给出
    sources = events[0][1]
    assert len(sources) == 3
    assert sources[0]["file_name"] == "员工手册.pdf"

    # 3) delta 拼接后与最终结果一致
    text = "".join(payload for kind, payload in events if kind == "delta")
    assert text == "根据资料，年假为五天。[1]"

    result = events[-1][1]
    assert result["found"] is True
    assert result["answer"] == text
    assert result["sources"] == sources
    assert len(result["hits"]) == 3

    # 4) 仍走严格防幻觉提示词
    assert captured["messages"][0]["content"] == kb.KB_SYSTEM_PROMPT
    assert "禁止编造" in captured["messages"][0]["content"]


def test_kb_chat_stream_empty_library() -> None:
    """知识库为空时：不调用大模型，照样走完整事件流。"""
    embedding = FakeEmbeddings()

    def boom(*args, **kwargs):  # pragma: no cover —— 不应被触发
        raise AssertionError("知识库为空时不应调用大模型")

    original = kb._stream_llm
    kb._stream_llm = boom
    try:
        with temp_db() as db:
            events = list(chat_stream("年假有多少天？", persist_dir=db, embedding=embedding))
    finally:
        kb._stream_llm = original

    assert [kind for kind, _ in events] == ["sources", "delta", "end"], events
    assert events[0][1] == []
    assert "无法从知识库中获取相关信息" in events[1][1]
    assert events[2][1]["found"] is False
    assert events[2][1]["answer"] == events[1][1]


def test_kb_chat_stream_empty_query() -> None:
    """空问题：直接给出 end 事件，不检索、不调用大模型。"""
    events = list(chat_stream("   "))
    assert len(events) == 1, events
    assert events[0][0] == "end"
    assert "请输入问题" in events[0][1]["answer"]
    assert events[0][1]["found"] is False


# ======================================================================
# 重排序（gte-rerank 两阶段检索）
# ======================================================================
def test_config_rerank_defaults() -> None:
    """重排序相关配置项默认值。"""
    from config import settings

    assert settings.rerank_enabled is True
    assert settings.rerank_model == "gte-rerank"
    assert "gte-rerank-v2" in settings.rerank_fallback_models
    assert settings.rerank_recall_k == 0  # 0 = 自动
    assert settings.rerank_recall_multiplier >= 1


def test_recall_k_resolution() -> None:
    """粗排候选数：自动取 Top-K × 倍率且不低于 10 条，且不超过库内总数。"""
    from src.kb import MIN_RECALL_K, _resolve_recall_k

    assert _resolve_recall_k(3, 100) == max(3 * 3, MIN_RECALL_K) == 10
    assert _resolve_recall_k(8, 100) == 24
    assert _resolve_recall_k(3, 4) == 4        # 库很小：最多只能召回 4 条
    assert _resolve_recall_k(3, 1) == 3        # 下限不小于 top_k


def test_search_with_rerank_reorders_and_limits() -> None:
    """开启重排：候选多于 Top-K 时精排生效，能把粗排靠后的片段捞到第一。"""
    embedding = FakeEmbeddings()
    # 刻意让打分与向量相似度「唱反调」：只认财务制度，用来证明排序确实被重排改写了
    reranker = FakeReranker(score_fn=lambda _q, text: 0.9 if "报销" in text else 0.1)
    with temp_db() as db:
        add_docs_to_db(_docs(), persist_dir=db, embedding=embedding)

        baseline = search("年假有多少天？", top_k=2, persist_dir=db, embedding=embedding)
        assert baseline[0]["file_name"] == "员工手册.pdf", baseline  # 粗排第一名是员工手册

        with patched_reranker(reranker):
            hits = search(
                "年假有多少天？", top_k=2, rerank=True, recall_k=3,
                persist_dir=db, embedding=embedding,
            )

        assert len(hits) == 2, hits
        # 1) 确实把 3 条候选交给了重排，并只保留 2 条
        assert reranker.calls == [{"query": "年假有多少天？", "candidates": 3, "top_n": 2}]

        # 2) 精排把财务制度（含「报销」）顶到了第一位 —— 排序依据被改写
        assert hits[0]["file_name"] == "财务制度.docx", hits
        assert hits[0]["recall_rank"] != 1
        assert [h["rank"] for h in hits] == [1, 2]
        assert hits[0]["score"] == hits[0]["rerank_score"] == 0.9

        # 3) 粗排信息被完整保留，便于解释「重排把哪条捞上来了」
        for hit in hits:
            assert 1 <= hit["recall_rank"] <= 3
            assert hit["recall_total"] == 3
            assert hit["vector_score"] is not None
            assert hit["rerank_model"] == "fake-rerank"


def test_search_rerank_disabled_keeps_vector_order() -> None:
    """关闭重排：只截取 Top-K，不带任何重排字段。"""
    embedding = FakeEmbeddings()
    with temp_db() as db:
        add_docs_to_db(_docs(), persist_dir=db, embedding=embedding)
        hits = search("年假有多少天？", top_k=2, rerank=False, persist_dir=db, embedding=embedding)

        assert len(hits) == 2
        assert [h["rank"] for h in hits] == [1, 2]
        assert all(h["rerank_score"] is None for h in hits)
        assert all(h["rerank_model"] is None for h in hits)
        assert all(h["recall_rank"] is None for h in hits)


def test_search_skips_rerank_when_pool_is_small() -> None:
    """库内条数不超过 Top-K 时没有择优空间，应跳过重排（省掉一次调用）。"""
    embedding = FakeEmbeddings()
    with temp_db() as db:
        add_docs_to_db(_docs(), persist_dir=db, embedding=embedding)

        with patched_reranker(BoomReranker()):
            hits = search("年假有多少天？", top_k=3, rerank=True, persist_dir=db, embedding=embedding)

        assert len(hits) == 3
        assert all(h["rerank_score"] is None for h in hits)


def test_search_rerank_failure_degrades_to_vector_order() -> None:
    """重排服务不可用时：降级为向量粗排结果，不能把问答打断。"""
    embedding = FakeEmbeddings()
    with temp_db() as db:
        add_docs_to_db(_docs(), persist_dir=db, embedding=embedding)

        baseline = search("年假有多少天？", top_k=2, rerank=False, persist_dir=db, embedding=embedding)
        with patched_reranker(BoomReranker()):
            degraded = search("年假有多少天？", top_k=2, rerank=True, recall_k=3, persist_dir=db, embedding=embedding)

        assert [h["content"] for h in degraded] == [h["content"] for h in baseline]
        assert all(h["rerank_score"] is None for h in degraded)


def test_reranker_itself_does_not_raise_on_api_error() -> None:
    """DashScopeReranker 内部也要兜住异常：接口报错时原样返回候选顺序。"""
    hits = [
        {"rank": 1, "content": "甲", "score": 0.9},
        {"rank": 2, "content": "乙", "score": 0.8},
    ]

    class BrokenResponse:
        status_code = 500
        message = "模拟服务异常"

    class FakeTextReRank:
        @staticmethod
        def call(**kwargs):  # noqa: ARG004 —— 只验证失败分支
            return BrokenResponse()

    original = reranker_mod.TextReRank
    reranker_mod.TextReRank = FakeTextReRank
    try:
        reranker = reranker_mod.DashScopeReranker(model="gte-rerank", api_key="sk-test")
        result = reranker.rerank_hits("年假", hits, top_n=2)
    finally:
        reranker_mod.TextReRank = original

    assert [h["content"] for h in result] == ["甲", "乙"]
    assert [h["rank"] for h in result] == [1, 2]
    assert all(h["rerank_score"] is None for h in result)


def test_reranker_parses_scores_and_skips_blank_docs() -> None:
    """重排返回值的解析：下标回填正确，空白文本不参与打分但不错位。"""
    captured: dict = {}

    class Item(dict):
        __getattr__ = dict.__getitem__

    class OkResponse:
        status_code = 200

        class output:  # noqa: N801 —— 模拟 dashscope 的响应结构
            # 接口返回的下标是「本次传过去的 documents 列表」的下标（空白文本已被剔除）
            results = [
                Item(index=1, relevance_score=0.91),
                Item(index=0, relevance_score=0.42),
            ]

    class FakeTextReRank:
        @staticmethod
        def call(**kwargs):
            captured.update(kwargs)
            return OkResponse()

    original = reranker_mod.TextReRank
    reranker_mod.TextReRank = FakeTextReRank
    try:
        reranker = reranker_mod.DashScopeReranker(model="gte-rerank", api_key="sk-test")
        result = reranker.rerank("年假", ["甲", "   ", "丙"], top_n=2)
    finally:
        reranker_mod.TextReRank = original

    # 空白文本被剔除，但返回的是原始下标（丙 -> 2）
    assert captured["documents"] == ["甲", "丙"], captured
    assert captured["model"] == "gte-rerank"
    assert [item["index"] for item in result] == [2, 0]
    assert result[0]["relevance_score"] == 0.91


def test_reranker_falls_back_to_available_model() -> None:
    """配置的模型在当前账号下不可用（403）时：自动回退到备用模型并记住它。

    现实背景：百炼上 ``gte-rerank`` 需要先在控制台开通，未开通会返回
    ``403 Access denied``；此时应自动改用 ``gte-rerank-v2``，而不是让问答失败。
    """
    calls: list[str] = []

    class Resp:
        def __init__(self, status, message="", results=None):
            self.status_code = status
            self.message = message
            self.output = {"results": results or []}

    class FakeTextReRank:
        @staticmethod
        def call(**kwargs):
            model = kwargs["model"]
            calls.append(model)
            if model == "gte-rerank":
                return Resp(403, "Access denied")
            return Resp(200, results=[
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.3},
            ])

    original = reranker_mod.TextReRank
    reranker_mod.TextReRank = FakeTextReRank
    try:
        reranker = reranker_mod.DashScopeReranker(
            model="gte-rerank", api_key="sk-test", fallback_models=["gte-rerank-v2"]
        )
        hits = [
            {"rank": 1, "content": "甲", "score": 0.8},
            {"rank": 2, "content": "乙", "score": 0.7},
        ]
        result = reranker.rerank_hits("问题", hits, top_n=2)
        again = reranker.rerank_hits("问题", hits, top_n=2)
    finally:
        reranker_mod.TextReRank = original

    # 第一次：先试配置模型（失败）再试备用模型；第二次：直接走已生效的模型
    assert calls == ["gte-rerank", "gte-rerank-v2", "gte-rerank-v2"], calls
    assert reranker.active_model == "gte-rerank-v2"

    assert [hit["content"] for hit in result] == ["乙", "甲"]
    assert all(hit["rerank_model"] == "gte-rerank-v2" for hit in result)
    assert again[0]["rerank_model"] == "gte-rerank-v2"


def test_reranker_all_models_unavailable_degrades() -> None:
    """所有候选模型都不可用：rerank_hits 仍不抛错，返回粗排顺序。"""

    class Resp:
        status_code = 403
        message = "Access denied"
        output = {"results": []}

    class FakeTextReRank:
        @staticmethod
        def call(**kwargs):  # noqa: ARG004
            return Resp()

    original = reranker_mod.TextReRank
    reranker_mod.TextReRank = FakeTextReRank
    try:
        reranker = reranker_mod.DashScopeReranker(
            model="gte-rerank", api_key="sk-test", fallback_models=["gte-rerank-v2"]
        )
        hits = [{"rank": 1, "content": "甲", "score": 0.8}, {"rank": 2, "content": "乙", "score": 0.7}]
        result = reranker.rerank_hits("问题", hits, top_n=2)
    finally:
        reranker_mod.TextReRank = original

    assert [hit["content"] for hit in result] == ["甲", "乙"]
    assert all(hit["rerank_score"] is None for hit in result)


def test_kb_chat_reports_rerank_in_retrieval() -> None:
    """问答结果里的 retrieval 概况能反映「粗排 N 条 → 保留 M 条」。"""
    embedding = FakeEmbeddings()

    def fake_llm(messages, temperature=0.2):  # noqa: ARG001
        return "年假为五天。[1]"

    original = kb._call_llm
    kb._call_llm = fake_llm
    try:
        with temp_db() as db:
            add_docs_to_db(_docs(), persist_dir=db, embedding=embedding)
            with patched_reranker(FakeReranker()):
                result = chat(
                    "年假有多少天？", top_k=2, rerank=True, recall_k=3,
                    persist_dir=db, embedding=embedding,
                )
    finally:
        kb._call_llm = original

    info = result["retrieval"]
    assert info["rerank"] is True
    assert info["rerank_model"] == "fake-rerank"  # 实际生效的模型名
    assert info["candidates"] == 3
    assert info["returned"] == 2
    assert all(src["rerank_score"] is not None for src in result["sources"])


def test_retrieval_info_without_rerank() -> None:
    """未重排时 retrieval 也要有值，界面才能统一渲染。"""
    embedding = FakeEmbeddings()

    def fake_llm(messages, temperature=0.2):  # noqa: ARG001
        return "年假为五天。[1]"

    original = kb._call_llm
    kb._call_llm = fake_llm
    try:
        with temp_db() as db:
            add_docs_to_db(_docs(), persist_dir=db, embedding=embedding)
            result = chat("年假有多少天？", top_k=2, persist_dir=db, embedding=embedding)
    finally:
        kb._call_llm = original

    assert result["retrieval"]["rerank"] is False
    assert result["retrieval"]["rerank_model"] is None
    assert result["retrieval"]["returned"] == 2


# ======================================================================
# 逐段溯源（src/citations.py）
# ======================================================================
def _sources() -> list[dict]:
    """两条来源（模拟 _to_source 的输出）。"""
    return [
        {"index": 1, "file_name": "员工手册.pdf", "page": 2,
         "citation": "员工手册.pdf（第 2 页）", "snippet": "年假为五天", "score": 0.8,
         "rerank_score": 0.9},
        {"index": 2, "file_name": "员工手册.pdf", "page": 3,
         "citation": "员工手册.pdf（第 3 页）", "snippet": "申请需主管审批", "score": 0.7,
         "rerank_score": 0.6},
    ]


def test_strip_markers_cleans_residual_spaces() -> None:
    """删掉编号后不留多余空格（模型常把标记写成「十五天 [2]。」）。"""
    from src.citations import strip_markers

    assert strip_markers("年假为十五天 [2]。") == "年假为十五天。"
    assert strip_markers("结论。[1][3]") == "结论。"
    assert strip_markers("列表项  ") == "列表项"
    assert strip_markers("英文 A and B [1]") == "英文 A and B"


def test_split_paragraphs() -> None:
    from src.citations import split_paragraphs

    assert split_paragraphs("甲。\n\n乙。\n\n丙。") == ["甲。", "乙。", "丙。"]
    # 没有空行时退化为按单换行切
    assert split_paragraphs("甲。\n乙。") == ["甲。", "乙。"]
    assert split_paragraphs("   ") == []
    assert split_paragraphs("") == []


def test_annotate_answer_marks_every_paragraph() -> None:
    """核心诉求：每段末尾标注它引用了哪个文件的哪一页。"""
    answer = "员工满一年年假为五天。[1]\n\n申请需在系统提交并经主管审批。[2]\n\n跨年度最多结转三天。[2][1]"
    annotated = annotate_answer(answer, _sources())

    # 1) 三段各带一条来源标注，编号本身从正文里被移除
    assert annotated.count("【来源：") == 3
    assert "[1]" not in annotated and "[2]" not in annotated
    assert "员工满一年年假为五天。" in annotated
    assert "【来源：员工手册.pdf（第 2 页）】" in annotated
    assert "【来源：员工手册.pdf（第 3 页）】" in annotated
    # 2) 同一段引用多个来源时按编号顺序合并，且去重
    assert "【来源：员工手册.pdf（第 3 页）；员工手册.pdf（第 2 页）】" in annotated


def test_build_segments_maps_paragraph_to_source() -> None:
    segments = build_segments("第一段。[1]\n\n第二段。[2]\n\n第三段没有编号。", _sources())

    assert [seg["paragraph"] for seg in segments] == [1, 2, 3]
    assert segments[0]["indexes"] == [1]
    assert segments[0]["citations"][0]["citation"] == "员工手册.pdf（第 2 页）"
    assert segments[0]["citations"][0]["snippet"] == "年假为五天"
    assert segments[1]["indexes"] == [2]
    assert segments[2]["indexes"] == [] and segments[2]["tag"] == ""
    # text 去掉编号，raw 保留原始文本便于排查
    assert segments[0]["text"] == "第一段。"
    assert segments[0]["raw"] == "第一段。[1]"


def test_build_segments_ignores_unknown_index() -> None:
    """模型编造了不存在的编号：不猜测、不标注，只保留能对上的。"""
    segments = build_segments("结论。[1][9]", _sources())
    assert segments[0]["indexes"] == [1]
    assert "【来源：员工手册.pdf（第 2 页）】" in segments[0]["tag"]


def test_annotate_answer_without_markers_or_sources() -> None:
    """没有编号或没有来源时，答案原样返回（宁可不标，也不猜）。"""
    plain = "这是一段没有任何编号的答案。"
    assert annotate_answer(plain, _sources()) == plain
    assert annotate_answer("有编号。[1]", []) == "有编号。"
    assert annotate_answer("", _sources()) == ""


def test_annotate_partial_only_marks_finished_paragraphs() -> None:
    """流式：只标注已写完的段落，正在生成的那段保持原样。"""
    # 还没有空行 -> 一段都没写完，不加标注
    assert annotate_partial("员工年假为五", _sources()) == "员工年假为五"

    # 出现空行 -> 上一段已结束，可以标注；末段仍在生成
    partial = annotate_partial("年假为五天。[1]\n\n申请需在", _sources())
    assert "【来源：员工手册.pdf（第 2 页）】" in partial
    assert partial.endswith("申请需在")
    assert partial.count("【来源：") == 1


def test_annotate_answer_is_idempotent_on_annotated_text() -> None:
    """重复标注不会叠加：第二次调用时标注行本身不含编号，不会再次生成来源。"""
    once = annotate_answer("年假为五天。[1]", _sources())
    twice = annotate_answer(once, _sources())
    assert once.count("【来源：") == 1
    assert twice.count("【来源：") == 1


def test_kb_chat_returns_annotated_answer_and_segments() -> None:
    """问答链路端到端：答案、逐段标注、结构化 segments 一起返回。"""
    embedding = FakeEmbeddings()

    def fake_llm(messages, temperature=0.2):  # noqa: ARG001
        return "年假为五天。[1]\n\n申请需经主管审批。[2]"

    original = kb._call_llm
    kb._call_llm = fake_llm
    try:
        with temp_db() as db:
            add_docs_to_db(_docs(), persist_dir=db, embedding=embedding)
            result = chat("年假有多少天？", top_k=3, persist_dir=db, embedding=embedding)
    finally:
        kb._call_llm = original

    assert result["answer"] == "年假为五天。[1]\n\n申请需经主管审批。[2]"
    assert result["answer_annotated"].count("【来源：") == 2
    assert "员工手册.pdf（第" in result["answer_annotated"]

    # 编号到文件的映射必须与检索结果一致（不能是模型编的）
    first = result["answer_annotated"].split("【来源：")[1].split("】")[0]
    assert first in [src["citation"] for src in result["sources"]]

    segments = result["segments"]
    assert len(segments) == 2
    assert segments[0]["indexes"] == [1]
    assert segments[0]["citations"][0]["citation"] == result["sources"][0]["citation"]
    assert segments[0]["citations"][0]["snippet"]


def test_kb_chat_fixed_answer_has_annotation_fields() -> None:
    """检索不到内容时，兜底返回也要带上溯源字段，界面无需分支处理。"""
    embedding = FakeEmbeddings()
    with temp_db() as db:
        result = chat("年假有多少天？", persist_dir=db, embedding=embedding)

    assert result["found"] is False
    assert result["answer_annotated"] == result["answer"]
    assert result["segments"] == []
    assert result["retrieval"]["returned"] == 0


def test_kb_chat_stream_end_event_carries_citations() -> None:
    """流式问答的 end 事件里带着带标注的完整答案与 segments。"""
    embedding = FakeEmbeddings()

    def fake_stream(messages, temperature=0.2):  # noqa: ARG001
        for piece in ("年假为五天。", "[1]", "\n\n", "申请需经主管审批。", "[2]"):
            yield piece

    original = kb._stream_llm
    kb._stream_llm = fake_stream
    try:
        with temp_db() as db:
            add_docs_to_db(_docs(), persist_dir=db, embedding=embedding)
            events = list(chat_stream("年假有多少天？", top_k=3, persist_dir=db, embedding=embedding))
    finally:
        kb._stream_llm = original

    result = events[-1][1]
    kinds = [kind for kind, _ in events]
    assert kinds[0] == "sources" and kinds[-1] == "end"
    assert result["answer_annotated"].count("【来源：") == 2
    assert len(result["segments"]) == 2
    assert result["retrieval"]["returned"] == 3
    # delta 拼接结果与 end 里的 answer 完全一致（界面按原文做流式展示）
    text = "".join(payload for kind, payload in events if kind == "delta")
    assert text == result["answer"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    raise SystemExit(1 if failed else 0)
