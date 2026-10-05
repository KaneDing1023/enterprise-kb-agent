"""Agent 知识库服务：阿里云百炼 ``text-embedding-v4`` + Chroma 本地持久化。

对外暴露四个核心函数：

- :func:`add_docs_to_db` —— 把切分后的 ``Document`` 批量写入向量数据库
- :func:`search_from_db` —— 根据用户问题检索最相关的 Top-K 片段，
  返回「文档内容 + 元数据（文件名、页码）」；支持**两阶段检索**（向量粗排 → gte-rerank 精排）
- :func:`kb_chat` —— 完整 RAG 问答链路：检索 → 拼上下文 → 调用通义千问
  ``qwen-plus`` 生成回答，并返回答案引用的文档来源 + **逐段溯源**
- :func:`kb_chat_stream` —— 与 :func:`kb_chat` 同一链路的**流式**版本，
  边生成边吐字，供 Web 界面做打字机效果与加载状态提示

设计要点
--------
* **向量模型**：百炼 ``text-embedding-v4``（Qwen3-Embedding 系列，默认 1024 维，
  单次最多 10 条文本）。通过 LangChain 的 ``Embeddings`` 接口接入
  （复用 :class:`src.embeddings.DashScopeEmbeddings`，底层直接调用 dashscope SDK），
  并开启 ``text_type`` 非对称检索优化：文档侧用 ``document``、查询侧用 ``query``。
* **两阶段检索（召回 → 精排）**：向量相似度高 ≠ 对回答问题有用。开启重排序后，
  先用向量召回 ``recall_k`` 条候选（默认 ``TOP_K × 3``、不低于 10 条），
  再交给百炼 ``gte-rerank`` 交叉编码器逐对打分重排，最后截取 Top-K。
  这样能显著提升 Top-1 命中率。重排调用失败会自动降级为向量粗排顺序，
  不打断问答（详见 :mod:`src.reranker`）。
* **向量库**：Chroma，本地持久化到 ``./chroma_kb``（可用 ``CHROMA_KB_DIR`` 覆盖）。
  进程退出后数据仍在磁盘上，下次启动直接复用，无需重新入库。
* **相似度空间**：集合以 ``cosine`` 空间创建，因此 ``score`` 近似等于余弦相似度，
  取值区间 ``[-1, 1]``，越大越相关，便于设定阈值。
* **逐段溯源**：提示词要求模型在**每一段**末尾标注片段编号 ``[n]``，
  再由 :mod:`src.citations` 把编号映射成真实的文件名与页码，
  在每段末尾追加 ``【来源：员工手册.pdf（第 2 页）】``，并返回逐段的结构化 ``segments``。
* **去重**：以「来源 + 页码 + 内容」的 MD5 作为稳定 id（Chroma 写入是 upsert），
  同一份文档重复入库不会产生脏数据。

典型用法::

    from src.doc_processor import load_and_split
    from src.kb import add_docs_to_db, search_from_db

    docs = load_and_split("data/raw/员工手册.pdf")   # 解析 + 切分
    add_docs_to_db(docs)                            # 批量入库

    # 两阶段检索：向量粗排 + gte-rerank 精排
    for hit in search_from_db("年假有多少天？", top_k=3, rerank=True):
        print(hit["score"], hit["rerank_score"], hit["citation"])
        print(hit["content"])

    # 完整问答：检索 + 通义千问生成 + 逐段引用来源
    from src.kb import kb_chat

    result = kb_chat("年假有多少天？")
    print(result["answer_annotated"])                # 每段末尾已标注来源
    for seg in result["segments"]:
        print(seg["paragraph"], seg["indexes"], seg["tag"])

    # 流式问答（Web 界面用）：边生成边打印
    from src.kb import kb_chat_stream

    for kind, payload in kb_chat_stream("年假有多少天？"):
        if kind == "sources":
            print("来源：", [s["citation"] for s in payload])
        elif kind == "delta":
            print(payload, end="", flush=True)

命令行试跑::

    python src/kb.py --ingest data/raw/员工手册.pdf     # 入库
    python src/kb.py "年假有多少天？"                   # 检索（默认已开重排）
    python src/kb.py "年假有多少天？" --no-rerank        # 只用向量粗排
    python src/kb.py --chat "年假有多少天？"             # 问答（qwen-plus）
"""

from __future__ import annotations

import hashlib
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

# 允许 `python src/kb.py ...` 直接运行：脚本模式下 sys.path[0] 是 src/，
# 看不到项目根目录里的 config.py，这里先把它补进去。
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dashscope import Generation  # noqa: E402
from langchain_chroma import Chroma  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402

from config import settings  # noqa: E402
from src.citations import annotate_answer, build_segments  # noqa: E402
from src.embeddings import DashScopeEmbeddings  # noqa: E402
from src.reranker import get_reranker  # noqa: E402

__all__ = [
    "add_docs_to_db",
    "search_from_db",
    "kb_chat",
    "kb_chat_stream",
    "get_vector_store",
    "get_embedding",
    "count_docs",
    "reset_db",
    "db_info",
]

logger = logging.getLogger(__name__)

# 默认召回条数（场景约定：Top-3）
DEFAULT_TOP_K = 3

# 粗排候选数下限：候选太少时重排序几乎没有择优空间
MIN_RECALL_K = 10

# 单次写入 Chroma 的切片数量。Chroma 内部会自己再做一次向量化请求分批，
# 这里只控制「一批提交多少条」，64 是个经验值，兼顾内存与写入速度。
DEFAULT_BATCH_SIZE = 64


# ----------------------------------------------------------------------
# 问答（RAG）用到的提示词常量
# ----------------------------------------------------------------------
# 系统提示词是「防幻觉」与「可溯源」的关键：明确只允许依据检索到的知识库原文作答，
# 找不到就直说，不允许用模型自身的常识去补全；同时强制每段末尾标注片段编号，
# 由 src/citations.py 机械映射成真实的文件名与页码（模型无法伪造来源）。
KB_SYSTEM_PROMPT = (
    "你是一名严谨的企业知识库问答助手。你只能依据用户提供的「参考资料」作答，"
    "「参考资料」是从企业知识库中检索出来的原文片段。\n"
    "必须严格遵守以下规则：\n"
    "1. 只使用「参考资料」中的信息回答，禁止使用参考资料之外的任何知识，"
    "禁止编造、猜测、推测或自行补充。\n"
    "2. 如果「参考资料」中找不到回答问题所需的信息，必须直接回复"
    "「无法从知识库中获取相关信息」，不要给出近似、推测性或凭常识的答案。\n"
    "3. 【溯源要求】每一个段落的末尾都必须标注该段结论所依据的片段编号，"
    "格式为 [1]、[2]，引用多个片段时写成 [1][3]（标注放在段落最末尾，"
    "不要在段落中间穿插）。编号必须与「参考资料」中的片段编号严格一致，"
    "禁止使用参考资料里不存在的编号；没有依据可标的段落就不要编造编号。\n"
    "4. 段落之间用一个空行分隔，一段只讲一个要点，便于逐段核对来源。\n"
    "5. 使用中文作答，条理清晰，可以使用列表（列表整体视为一个段落，"
    "每个列表项末尾同样要标注编号）。"
)

# 用户消息模板：把检索到的上下文与用户问题一起交给模型。
KB_USER_TEMPLATE = """请严格依据下面的「参考资料」回答「用户问题」。

【参考资料】
{context}

【用户问题】
{question}

【回答】
"""

# 检索不到任何片段时返回的固定话术（不调用大模型，直接返回）。
KB_NO_ANSWER = "无法从知识库中获取相关信息。请确认相关文档已上传入库，或换一种问法再试。"


# ----------------------------------------------------------------------
# 单例缓存：同一个「持久化目录 + 集合」只创建一个客户端
# ----------------------------------------------------------------------
_embedding_cache: dict[str, Embeddings] = {}
_store_singleton: Chroma | None = None


def get_embedding(model: str | None = None) -> Embeddings:
    """获取（并复用）百炼向量化实例。

    默认使用 ``text-embedding-v4``（可用 ``KB_EMBEDDING_MODEL`` 覆盖），
    并开启 ``text_type`` 非对称检索优化。
    """
    model = model or settings.kb_embedding_model
    if model not in _embedding_cache:
        _embedding_cache[model] = DashScopeEmbeddings(model=model, use_text_type=True)
        logger.info("已加载向量模型：%s", model)
    return _embedding_cache[model]


def _build_store(
    persist_dir: str | Path | None,
    collection_name: str | None,
    embedding: Embeddings | None,
) -> Chroma:
    """构造 Chroma 客户端（不做缓存）。"""
    directory = (
        Path(persist_dir).expanduser().resolve()
        if persist_dir is not None
        else settings.kb_persist_dir
    )
    directory.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=collection_name or settings.kb_collection_name,
        embedding_function=embedding or get_embedding(),
        persist_directory=str(directory),
        # 文本向量场景统一用余弦空间，score 直接可比、便于卡阈值
        collection_metadata={"hnsw:space": "cosine"},
    )


def get_vector_store(
    persist_dir: str | Path | None = None,
    collection_name: str | None = None,
    embedding: Embeddings | None = None,
) -> Chroma:
    """打开（不存在则创建）``./chroma_kb`` 下的 Chroma 集合。

    三个参数一般不用传；显式传入时（测试、多库并存等场景）不会复用单例，
    避免不同目录 / 集合之间互相污染。
    """
    global _store_singleton

    if persist_dir is not None or collection_name is not None or embedding is not None:
        return _build_store(persist_dir, collection_name, embedding)

    if _store_singleton is None:
        _store_singleton = _build_store(None, None, None)
        logger.info("已打开向量库：%s", settings.kb_persist_dir)
    return _store_singleton


# ----------------------------------------------------------------------
# 核心函数 1：入库
# ----------------------------------------------------------------------
def add_docs_to_db(
    docs: list[Document] | tuple[Document, ...] | None,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    persist_dir: str | Path | None = None,
    collection_name: str | None = None,
    embedding: Embeddings | None = None,
) -> int:
    """把切分后的文档批量存入向量数据库。

    Args:
        docs: 已切分的 ``Document`` 列表（通常来自 ``load_and_split()``）。
            内容为空白的切片会被自动跳过。
        batch_size: 单次提交给 Chroma 的切片数量，默认 64。
        persist_dir: 覆盖持久化目录，默认 ``./chroma_kb``。
        collection_name: 覆盖集合名，默认 ``enterprise_kb``。
        embedding: 注入自定义向量化实现（测试用），默认走百炼 text-embedding-v4。

    Returns:
        实际写入（或按 id 覆盖）的切片数量；``docs`` 为空时返回 0，不抛异常。
    """
    if not docs:
        logger.warning("传入的 docs 为空，本次没有内容入库。")
        return 0

    # 1) 丢掉空白切片，并去掉重复 id（Chroma 要求同一次调用内 id 唯一）
    documents: list[Document] = []
    ids: list[str] = []
    seen: set[str] = set()
    for doc in docs:
        if not isinstance(doc, Document) or not doc.page_content.strip():
            continue
        doc_id = _stable_id(doc)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        documents.append(doc)
        ids.append(doc_id)

    duplicates = len(docs) - len(documents)
    if not documents:
        logger.warning("docs 中没有任何非空文本，本次没有内容入库。")
        return 0

    # 2) 分批写入。Chroma 的 add_documents 是 upsert 语义，
    #    相同 id 会覆盖而不是新增，所以重复入库是幂等的。
    store = get_vector_store(persist_dir, collection_name, embedding)
    written = 0
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        store.add_documents(documents=batch, ids=ids[start : start + batch_size])
        written += len(batch)
        logger.info("已写入 %d/%d 个切片", written, len(documents))

    logger.info(
        "入库完成：写入 %d 个切片（跳过重复 %d 个）-> %s",
        written,
        duplicates,
        _describe(persist_dir, collection_name),
    )
    return written


# ----------------------------------------------------------------------
# 核心函数 2：检索
# ----------------------------------------------------------------------
def search_from_db(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    *,
    score_threshold: float | None = None,
    rerank: bool | None = None,
    recall_k: int | None = None,
    persist_dir: str | Path | None = None,
    collection_name: str | None = None,
    embedding: Embeddings | None = None,
) -> list[dict]:
    """根据用户问题检索最相关的文档片段（两阶段：向量粗排 → gte-rerank 精排）。

    处理流程::

        用户问题
          -> 向量粗排    Chroma 召回 recall_k 条候选（余弦相似度）
          -> 阈值过滤    按「相似度下限」丢掉明显不相关的候选
          -> gte-rerank 候选数多于 Top-K 时，逐对打分重新排序（失败自动降级）
          -> 取 Top-K    重新编号 rank = 1..K

    Args:
        query: 用户问题。
        top_k: 最终返回条数，默认 3。
        score_threshold: 相似度下限，作用在**向量粗排的余弦相似度**上
            （``None`` 表示不过滤）；重排序发生在过滤之后，不会用阈值卡掉高精排分的片段。
        rerank: 是否启用重排序；``None`` 表示跟随全局配置 ``RERANK_ENABLED``。
        recall_k: 向量粗排的候选条数；``None`` 表示自动
            （``TOP_K × RERANK_RECALL_MULTIPLIER``，且不低于 10 条，不超过库内总数）。
            候选数不超过 ``top_k`` 时会跳过重排序，省掉一次无意义的调用。
        persist_dir: 覆盖持久化目录，默认 ``./chroma_kb``。
        collection_name: 覆盖集合名，默认 ``enterprise_kb``。
        embedding: 注入自定义向量化实现（测试用），默认走百炼 text-embedding-v4。

    Returns:
        命中列表，按最终相关性从高到低排序；每条包含：

        ==============  ==========================================================
        字段             含义
        ==============  ==========================================================
        ``rank``        最终排名，从 1 开始（重排后重新编号）
        ``content``     文档片段正文
        ``score``       最终排序依据：重排时为 gte-rerank 相关性分，否则为余弦相似度
        ``vector_score`` 向量粗排的余弦相似度（排除「重排把哪条捞上来了」的疑问）
        ``rerank_score`` gte-rerank 相关性分数；未重排或重排失败时为 ``None``
        ``recall_rank`` 该片段在向量粗排阶段的排名；未重排时为 ``None``
        ``file_name``   来源文件名
        ``page``        来源页码，取元数据原值；TXT / DOCX 等无页码时为 ``None``
        ``source``      来源文件路径
        ``citation``    可直接展示的引用标签，如 ``员工手册.pdf（第 3 页）``
        ==============  ==========================================================

        知识库为空或问题为空时返回 ``[]``。
    """
    query = (query or "").strip()
    if not query:
        logger.warning("问题为空，跳过检索。")
        return []

    total = count_docs(persist_dir, collection_name, embedding)
    if total == 0:
        logger.warning("向量库中还没有任何内容，请先调用 add_docs_to_db()。")
        return []

    wanted = max(1, int(top_k))

    # 1) 是否需要重排序：显式参数优先，否则跟随全局配置
    use_rerank = settings.rerank_enabled if rerank is None else bool(rerank)

    # 2) 第一阶段（粗排）：要重排就多召回一些候选，交给 gte-rerank 择优
    want = _resolve_recall_k(wanted, total) if use_rerank else wanted
    k = max(1, min(want, total))  # Chroma 的 n_results 不能超过集合总数

    store = get_vector_store(persist_dir, collection_name, embedding)
    pairs = store.similarity_search_with_relevance_scores(query, k=k)

    hits: list[dict] = []
    for doc, score in pairs:
        if score_threshold is not None and score < score_threshold:
            continue
        hits.append(_to_hit(len(hits) + 1, doc, score))

    if not hits:
        logger.info("检索「%s」未命中（top_k=%d，粗排 %d 条，阈值 %s）", query, wanted, k, score_threshold)
        return []

    # 3) 第二阶段（精排）：候选比所需条数多时才有择优空间
    if use_rerank and len(hits) > wanted:
        reranked = _rerank_hits(query, hits, wanted)
        logger.info(
            "检索「%s」粗排 %d 条 → %s 重排保留 %d 条",
            query,
            len(hits),
            reranked[0].get("rerank_model") or settings.rerank_model,
            len(reranked),
        )
        return reranked

    hits = hits[:wanted]
    logger.info("检索「%s」命中 %d 条（top_k=%d，未重排）", query, len(hits), wanted)
    return hits


def _resolve_recall_k(top_k: int, total: int) -> int:
    """计算向量粗排的候选条数：显式配置优先，否则按倍率自动，且不超过库内总数。"""
    if settings.rerank_recall_k > 0:
        want = settings.rerank_recall_k
    else:
        want = max(top_k * settings.rerank_recall_multiplier, MIN_RECALL_K)
    return max(top_k, min(want, total))


def _rerank_hits(query: str, hits: list[dict], top_k: int) -> list[dict]:
    """调用 gte-rerank 精排；失败时降级为粗排顺序（不打断问答链路）。"""
    try:
        return get_reranker().rerank_hits(query, hits, top_n=top_k)
    except Exception as exc:  # noqa: BLE001 —— 排序只是优化项，不该让问答失败
        logger.warning("重排序不可用，已降级为向量粗排结果：%s", exc)
        return hits[:top_k]


# ----------------------------------------------------------------------
# 核心函数 3：RAG 问答（检索 + 通义千问生成 + 引用来源）
# ----------------------------------------------------------------------
def kb_chat(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    *,
    score_threshold: float | None = None,
    temperature: float = 0.2,
    rerank: bool | None = None,
    recall_k: int | None = None,
    persist_dir: str | Path | None = None,
    collection_name: str | None = None,
    embedding: Embeddings | None = None,
) -> dict:
    """完整 RAG 问答链路：检索知识库 -> 拼上下文 -> 调用通义千问生成回答。

    处理流程::

        用户问题
          -> search_from_db()        两阶段检索：向量粗排 -> gte-rerank 精排 -> Top-K
          -> _build_context()        把片段拼成带编号 [1][2] 的参考资料
          -> Generation.call()       用 qwen-plus 依据参考资料作答（严格提示词，禁止编造）
          -> build_segments()        把回答按段落切分，解析每段引用的片段编号
          -> annotate_answer()       每段末尾追加【来源：文件名（第 X 页）】
          -> 返回 {answer, answer_annotated, segments, sources, ...}

    系统提示词强制模型「只依据检索到的知识库内容作答，禁止编造」，
    并且要求**每一段末尾**标注片段编号；编号→真实文件名/页码的映射由系统完成，
    模型无法伪造来源。若参考资料不足以回答，模型会回复「无法从知识库中获取相关信息」。
    检索阶段若一条都没命中，则不调用大模型，直接返回 :data:`KB_NO_ANSWER`。

    Args:
        query: 用户问题。
        top_k: 最终召回的片段条数，默认 3。
        score_threshold: 相似度下限，作用在向量粗排的余弦相似度上；``None`` 表示不过滤。
        temperature: 采样温度，默认 0.2，越低越稳定、越少发散。
        rerank: 是否用 gte-rerank 精排；``None`` 跟随全局配置 ``RERANK_ENABLED``。
        recall_k: 向量粗排候选数；``None`` 自动（``TOP_K × 3``，不低于 10 条）。
        persist_dir: 覆盖向量库目录，默认 ``./chroma_kb``。
        collection_name: 覆盖集合名，默认 ``enterprise_kb``。
        embedding: 注入自定义向量化实现（测试用），默认走百炼 text-embedding-v4。

    Returns:
        问答结果字典：

        ====================  ==========================================================
        字段                   含义
        ====================  ==========================================================
        ``query``             原始问题（已去空白）
        ``answer``            模型生成的回答原文（找不到时是固定话术）
        ``answer_annotated``  逐段溯源版回答：每段末尾追加【来源：文件（第 X 页）】
        ``segments``          逐段溯源结构，每项含
                              ``paragraph`` / ``text`` / ``raw`` / ``indexes`` /
                              ``citations`` / ``tag``
        ``sources``           答案引用的文档来源列表，每项含
                              ``index`` / ``file_name`` / ``page`` / ``source`` /
                              ``citation`` / ``score`` / ``vector_score`` /
                              ``rerank_score`` / ``recall_rank`` / ``preview`` / ``snippet``
        ``hits``              原始检索命中（结构同 :func:`search_from_db`），便于二次加工
        ``retrieval``         本轮检索概况：``rerank`` / ``rerank_model`` /
                              ``candidates``（进入精排的候选数）/ ``returned``
        ``found``             是否基于知识库给出了回答（检索到片段且未走固定话术为 ``True``）
        ====================  ==========================================================

    Example:
        >>> result = kb_chat("年假有多少天？")
        >>> print(result["answer_annotated"])
        员工累计工作满一年不满十年的，年假为五天。[1]
        【来源：员工手册.pdf（第 2 页）】
        >>> [src["citation"] for src in result["sources"]]
        ['员工手册.pdf（第 2 页）', '员工手册.pdf（第 3 页）']
    """
    query = (query or "").strip()
    if not query:
        return _empty_result("", "请输入问题。")

    # 1) 检索：拿不到任何片段说明知识库里没有相关内容，直接返回固定话术
    hits = search_from_db(
        query,
        top_k=top_k,
        score_threshold=score_threshold,
        rerank=rerank,
        recall_k=recall_k,
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding=embedding,
    )
    if not hits:
        logger.warning("「%s」未检索到任何片段，返回固定话术。", query)
        return _empty_result(query, KB_NO_ANSWER)

    # 2) 拼上下文并调用通义千问（qwen-plus）生成回答
    context = _build_context(hits)
    answer = _call_llm(_build_chat_messages(query, context), temperature=temperature)

    # 3) 逐段溯源：编号 -> 真实文件名 / 页码 / 对应片段
    sources = [_to_source(hit) for hit in hits]
    segments = build_segments(answer, sources)
    annotated = annotate_answer(answer, sources)

    logger.info(
        "「%s」生成回答 %d 字，引用 %d 处来源（%d 段已标注来源）",
        query,
        len(answer),
        len(sources),
        sum(1 for seg in segments if seg["tag"]),
    )
    return {
        "query": query,
        "answer": answer,
        "answer_annotated": annotated,
        "segments": segments,
        "sources": sources,
        "hits": hits,
        "retrieval": _retrieval_info(hits),
        "found": True,
    }


def _empty_result(query: str, answer: str) -> dict:
    """构造「不调用大模型」的兜底返回，字段与正常返回完全一致，方便界面统一处理。"""
    return {
        "query": query,
        "answer": answer,
        "answer_annotated": answer,
        "segments": [],
        "sources": [],
        "hits": [],
        "retrieval": {"rerank": False, "rerank_model": None, "candidates": 0, "returned": 0},
        "found": False,
    }


def _retrieval_info(hits: list[dict]) -> dict:
    """汇总本轮检索概况，供界面展示「粗排 N 条 → 精排保留 M 条」。"""
    reranked = any(hit.get("rerank_score") is not None for hit in hits)
    candidates = 0
    used_model = None
    for hit in hits:
        if hit.get("recall_total") and not candidates:
            candidates = int(hit["recall_total"])
        if hit.get("rerank_model") and not used_model:
            # 实际生效的模型（配置模型不可用时会自动回退到备用模型）
            used_model = hit["rerank_model"]
    return {
        "rerank": reranked,
        "rerank_model": used_model or (settings.rerank_model if reranked else None),
        # 未重排时没有候选总数，界面统一按「召回即候选」展示
        "candidates": candidates or len(hits),
        "returned": len(hits),
    }


def kb_chat_stream(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    *,
    score_threshold: float | None = None,
    temperature: float = 0.2,
    rerank: bool | None = None,
    recall_k: int | None = None,
    persist_dir: str | Path | None = None,
    collection_name: str | None = None,
    embedding: Embeddings | None = None,
) -> Iterator[tuple[str, object]]:
    """流式版 RAG 问答：链路与 :func:`kb_chat` 完全一致，只是把回答边生成边吐出来。

    供 Web 界面（``app_agent.py``）做打字机效果与分阶段加载提示。

    依次 yield 三种事件，调用方按第一个元素分发：

    ============  ======================================================
    ``kind``      ``payload``
    ============  ======================================================
    ``sources``   引用来源列表（结构同 :func:`kb_chat` 的 ``sources``），
                  在开始生成回答**之前**就会先抛出一次，便于界面先显示来源
    ``delta``     本次新增的回答片段（字符串），需要自行拼接
    ``end``       完整结果字典（结构同 :func:`kb_chat` 的返回值，
                  含 ``answer_annotated`` / ``segments`` / ``retrieval``），
                  恒为最后一个事件；界面可用它落库到会话历史
    ============  ======================================================

    检索为空时：先 yield ``("sources", [])``，再 yield 一次固定话术，
    最后 ``("end", ...)`` 且 ``found=False``，全程**不调用大模型**。

    Args:
        query: 用户问题。
        top_k: 最终召回的片段条数，默认 3。
        score_threshold: 相似度下限，作用在向量粗排的余弦相似度上。
        temperature: 采样温度，默认 0.2。
        rerank: 是否用 gte-rerank 精排；``None`` 跟随全局配置。
        recall_k: 向量粗排候选数；``None`` 自动。
        persist_dir: 覆盖向量库目录，默认 ``./chroma_kb``。
        collection_name: 覆盖集合名，默认 ``enterprise_kb``。
        embedding: 注入自定义向量化实现（测试用）。

    流式场景的溯源提示：``delta`` 逐段吐字时，界面可以用
    :func:`src.citations.annotate_partial` 为「已经写完的段落」实时补来源标注，
    等 ``end`` 事件到达后再用 ``answer_annotated`` 换成完整版本。

    Example:
        >>> for kind, payload in kb_chat_stream("年假有多少天？"):
        ...     if kind == "delta":
        ...         print(payload, end="")
    """
    query = (query or "").strip()

    # 空问题：不检索、不调用大模型
    if not query:
        yield "end", _empty_result("", "请输入问题。")
        return

    # 1) 检索：一条都没命中就直接走固定话术
    hits = search_from_db(
        query,
        top_k=top_k,
        score_threshold=score_threshold,
        rerank=rerank,
        recall_k=recall_k,
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding=embedding,
    )
    if not hits:
        logger.warning("「%s」未检索到任何片段，返回固定话术。", query)
        yield "sources", []
        yield "delta", KB_NO_ANSWER
        yield "end", _empty_result(query, KB_NO_ANSWER)
        return

    # 2) 先把引用来源抛给界面，再开始生成回答
    sources = [_to_source(hit) for hit in hits]
    yield "sources", sources

    context = _build_context(hits)
    chunks: list[str] = []
    for delta in _stream_llm(_build_chat_messages(query, context), temperature=temperature):
        chunks.append(delta)
        yield "delta", delta

    answer = "".join(chunks)

    # 3) 逐段溯源（与 kb_chat 一致，只是放在 end 事件里一次性给出）
    segments = build_segments(answer, sources)
    annotated = annotate_answer(answer, sources)

    logger.info(
        "「%s」流式生成回答 %d 字，引用 %d 处来源（%d 段已标注来源）",
        query,
        len(answer),
        len(sources),
        sum(1 for seg in segments if seg["tag"]),
    )
    yield "end", {
        "query": query,
        "answer": answer,
        "answer_annotated": annotated,
        "segments": segments,
        "sources": sources,
        "hits": hits,
        "retrieval": _retrieval_info(hits),
        "found": True,
    }


def _build_context(hits: list[dict]) -> str:
    """把检索命中拼成带编号的参考资料，编号与引用标签一致。"""
    blocks = [
        f"[{hit['rank']}] 来源：{hit['citation']}\n{(hit.get('content') or '').strip()}"
        for hit in hits
    ]
    return "\n\n".join(blocks)


def _build_chat_messages(query: str, context: str) -> list[dict]:
    """构造交给大模型的消息列表（system + user）。"""
    return [
        {"role": "system", "content": KB_SYSTEM_PROMPT},
        {"role": "user", "content": KB_USER_TEMPLATE.format(context=context, question=query)},
    ]


def _call_llm(messages: list[dict], temperature: float = 0.2) -> str:
    """调用通义千问生成回答（独立成函数，便于测试时打桩替换）。"""
    response = Generation.call(
        model=settings.llm_model,
        messages=messages,
        result_format="message",
        temperature=temperature,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"通义千问（{settings.llm_model}）调用失败 "
            f"[{response.status_code}] {getattr(response, 'message', '')}"
        )
    return response.output.choices[0].message.content or ""


def _stream_llm(messages: list[dict], temperature: float = 0.2) -> Iterator[str]:
    """流式调用通义千问，逐段 yield 新增文本（独立成函数，便于测试时打桩替换）。"""
    responses = Generation.call(
        model=settings.llm_model,
        messages=messages,
        result_format="message",
        temperature=temperature,
        stream=True,
        incremental_output=True,  # 只返回本次新增内容，前端直接拼接即可
    )

    for response in responses:
        if response.status_code != 200:
            raise RuntimeError(
                f"通义千问（{settings.llm_model}）调用失败 "
                f"[{response.status_code}] {getattr(response, 'message', '')}"
            )
        delta = response.output.choices[0].message.content or ""
        if delta:
            yield delta


def _to_source(hit: dict) -> dict:
    """把检索命中压缩成「引用来源」条目，供界面展示与逐段溯源使用。"""
    content = (hit.get("content") or "").strip()
    preview = content if len(content) <= 80 else content[:80] + "……"
    # snippet 比 preview 长，用于「这段答案具体出自原文哪一句」
    snippet = content if len(content) <= 160 else content[:160] + "……"
    return {
        "index": hit.get("rank"),
        "file_name": hit.get("file_name"),
        "page": hit.get("page"),
        "source": hit.get("source"),
        "citation": hit.get("citation"),
        "score": hit.get("score"),
        "vector_score": hit.get("vector_score"),
        "rerank_score": hit.get("rerank_score"),
        "rerank_model": hit.get("rerank_model"),
        "recall_rank": hit.get("recall_rank"),
        "recall_total": hit.get("recall_total"),
        "preview": preview,
        "snippet": snippet,
    }


# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------
def _stable_id(doc: Document) -> str:
    """用「来源 + 页码 + 块号 + 内容」生成稳定 id，保证重复入库不产生脏数据。"""
    meta = doc.metadata or {}
    raw = "|".join(
        str(meta.get(key, "")) for key in ("source", "page", "block", "chunk_index")
    )
    return hashlib.md5(f"{raw}|{doc.page_content}".encode("utf-8")).hexdigest()


def _to_hit(rank: int, doc: Document, score: float | None) -> dict:
    """把 LangChain 的 Document 转成对外返回的命中结构。"""
    meta = doc.metadata or {}
    source = str(meta.get("source") or "")
    file_name = meta.get("file_name") or Path(source).name or "未知文件"

    page = meta.get("page")
    if isinstance(page, str) and page.strip().isdigit():
        page = int(page.strip())
    if not isinstance(page, int):
        page = None

    return {
        "rank": rank,
        "content": doc.page_content,
        "score": round(float(score), 4) if score is not None else None,
        # 以下三个字段在重排阶段被填充；未重排时恒为 None，
        # 保证命中结构对所有调用方一致（界面/CLI 不需要分支处理）
        "vector_score": round(float(score), 4) if score is not None else None,
        "rerank_score": None,
        "rerank_model": None,
        "recall_rank": None,
        "recall_total": None,
        "file_name": file_name,
        "page": page,
        "source": source,
        "citation": f"{file_name}（第 {page} 页）" if page is not None else str(file_name),
    }


def count_docs(
    persist_dir: str | Path | None = None,
    collection_name: str | None = None,
    embedding: Embeddings | None = None,
) -> int:
    """向量库中当前的切片总数（出错时返回 0，不中断主流程）。"""
    try:
        return get_vector_store(persist_dir, collection_name, embedding)._collection.count()
    except Exception as exc:  # noqa: BLE001 —— 计数失败不应影响业务
        logger.warning("读取向量库计数失败：%s", exc)
        return 0


def reset_db(
    persist_dir: str | Path | None = None,
    collection_name: str | None = None,
) -> None:
    """清空集合（危险操作，会删除全部已入库内容，且不可恢复）。"""
    global _store_singleton
    store = get_vector_store(persist_dir, collection_name, None)
    try:
        store.reset_collection()
    except AttributeError:
        store.delete_collection()
    if persist_dir is None and collection_name is None:
        _store_singleton = None
    logger.warning("已清空集合 %s", collection_name or settings.kb_collection_name)


def db_info() -> dict:
    """当前知识库的概况，便于界面展示与排错。"""
    rerank_model = None
    if settings.rerank_enabled and settings.is_configured():
        try:
            # active_model 是实际生效的模型（首次调用后可能已自动回退到备用模型）
            rerank_model = get_reranker().active_model
        except Exception as exc:  # noqa: BLE001 —— 概况信息不该影响页面渲染
            logger.warning("读取排序模型信息失败：%s", exc)
            rerank_model = settings.rerank_model
    return {
        "collection": settings.kb_collection_name,
        "persist_dir": str(settings.kb_persist_dir),
        "chunks": count_docs(),
        "embedding_model": settings.kb_embedding_model,
        "rerank_model": rerank_model,
        "distance": "cosine",
    }


def _describe(persist_dir: str | Path | None, collection_name: str | None) -> str:
    directory = Path(persist_dir).expanduser().resolve() if persist_dir else settings.kb_persist_dir
    return f"{directory} / {collection_name or settings.kb_collection_name}"


# ----------------------------------------------------------------------
# 命令行入口
#     python src/kb.py --ingest <文件或目录>     # 入库
#     python src/kb.py [--top-k 3] "问题"        # 检索
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Chroma 知识库：入库 / 检索 / 问答")
    parser.add_argument("query", nargs="?", default=None, help="要检索的问题")
    parser.add_argument("--ingest", metavar="PATH", default=None, help="要入库的文件或目录")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="召回条数，默认 3")
    parser.add_argument("--chat", action="store_true", help="问答模式：检索 + qwen-plus 生成回答")
    parser.add_argument("--reset", action="store_true", help="入库前清空向量库")
    parser.add_argument(
        "--no-rerank", action="store_true", help="关闭 gte-rerank 重排序，只用向量粗排"
    )
    parser.add_argument(
        "--recall-k", type=int, default=None, help="向量粗排候选数，默认按 Top-K × 3 自动计算"
    )
    args = parser.parse_args()

    use_rerank = not args.no_rerank

    if args.ingest:
        from src.doc_processor import load_and_split

        settings.validate()
        if args.reset:
            reset_db()

        target = Path(args.ingest).expanduser().resolve()
        files = sorted(target.rglob("*")) if target.is_dir() else [target]
        total = 0
        for file in files:
            if not file.is_file():
                continue
            try:
                total += add_docs_to_db(load_and_split(file))
            except Exception as exc:  # noqa: BLE001 —— 单个文件失败不影响整体
                logger.warning("跳过 %s：%s", file.name, exc)
        print(f"\n本次共写入 {total} 个切片；{db_info()}\n")

    if args.query and not args.chat:
        hits = search_from_db(
            args.query,
            top_k=args.top_k,
            rerank=use_rerank,
            recall_k=args.recall_k,
        )
        if not hits:
            print("未检索到相关内容。")
        for hit in hits:
            line = f"\n[{hit['rank']}] {hit['citation']}  score={hit['score']}"
            if hit.get("rerank_score") is not None:
                line += f"（重排分 {hit['rerank_score']} / 召回第 {hit['recall_rank']} 位，余弦 {hit['vector_score']}）"
            print(line)
            print(hit["content"])

    if args.chat:
        if not args.query:
            print("用法：python src/kb.py --chat \"问题\"")
        else:
            settings.validate()
            result = kb_chat(
                args.query, top_k=args.top_k, rerank=use_rerank, recall_k=args.recall_k
            )
            print(f"\n【回答】\n{result['answer_annotated']}\n")
            info = result["retrieval"]
            if info["rerank"]:
                print(f"（已用 {info['rerank_model']} 重排：{info['candidates']} 条候选 → 保留 {info['returned']} 条）")
            if result["sources"]:
                print("【引用来源】")
                for src in result["sources"]:
                    line = f"  [{src['index']}] {src['citation']}  score={src['score']}"
                    if src.get("rerank_score") is not None:
                        line += f"（召回第 {src['recall_rank']} 位，余弦 {src['vector_score']}）"
                    print(line)

    if not args.ingest and not args.query:
        parser.print_help()
