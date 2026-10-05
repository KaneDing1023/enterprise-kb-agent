"""企业知识库问答（Agent 应用）—— Streamlit Web 界面。

后端链路见 :mod:`src.kb`：百炼 ``text-embedding-v4`` + Chroma（``./chroma_kb``）
+ ``gte-rerank`` 重排序 + 通义千问 ``qwen-plus``。

检索采用**两阶段**：向量粗排多召回候选 → ``gte-rerank`` 交叉编码器精排 → Top-K，
提升 Top-1 命中率；回答采用**逐段溯源**：每段末尾标注该段依据的文件与页码。

界面布局::

    左侧边栏                      主区
    ─────────────                ───────────────
    上传 PDF / TXT / DOCX    →    多轮对话窗口
      （上传即自动解析入库）          · 流式回答 + 逐段来源标注
    知识库状态（分片数）             · 加载状态（粗排条数 → 精排保留条数）
    检索参数（TOP_K / 阈值 / 重排序） · 引用来源卡片（文件名 / 页码 / 重排分 / 对应片段）
    清空对话 / 清空知识库

启动::

    streamlit run app_agent.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402
from src.citations import annotate_answer, annotate_partial  # noqa: E402
from src.doc_processor import (  # noqa: E402
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    load_and_split,
)
from src.kb import DEFAULT_TOP_K, add_docs_to_db, db_info, kb_chat_stream, reset_db  # noqa: E402

# 上传后要自动入库的文件类型（与 src/doc_processor.py 的 Loader 注册表一致）
ACCEPTED_TYPES = ["pdf", "txt", "docx", "md"]

EXAMPLE_QUESTIONS = [
    "年假有多少天？怎么申请？",
    "差旅报销需要哪些材料？",
    "试用期是多久？",
]

st.set_page_config(
    page_title="企业知识库问答",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# 页面样式（浅色主题下的少量修饰，不改变 Streamlit 默认控件观感）
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 2.6rem; padding-bottom: 5rem; max-width: 1120px; }
      [data-testid="stSidebar"] .block-container { padding-top: 1.8rem; }
      #MainMenu, footer { visibility: hidden; }
      .kb-hero {
        border: 1px solid #E5E7EB; border-radius: 14px; padding: 18px 22px;
        background: linear-gradient(135deg, #F8FAFF 0%, #F5F7FA 100%);
      }
      .kb-hero h3 { margin: 0 0 8px 0; font-size: 1.05rem; color: #111827; }
      .kb-hero p { margin: 0; color: #4B5563; font-size: 0.9rem; line-height: 1.75; }
      .kb-cite {
        margin: 4px 0 12px 0; padding: 4px 10px; border-left: 3px solid #BFDBFE;
        background: #F8FAFF; border-radius: 0 6px 6px 0;
        color: #1D4ED8; font-size: 0.84rem; line-height: 1.6;
      }
      .kb-ring { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
                 background: #16A34A; margin-right: 6px; vertical-align: middle; }
      .kb-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
                background: #DC2626; margin-right: 6px; vertical-align: middle; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def flash(message: str, kind: str = "success") -> None:
    """记一条一次性提示，在下一次页面渲染时显示。"""
    st.session_state.flash = (kind, message)


def render_flash() -> None:
    """显示并清除一次性提示（st.rerun() 后仍能看到结果）。"""
    item = st.session_state.pop("flash", None)
    if not item:
        return
    kind, message = item
    {"success": st.success, "warning": st.warning, "error": st.error, "info": st.info}[kind](message)


def fingerprint(file, chunk_size: int, chunk_overlap: int) -> str:
    """上传文件的去重指纹：文件名 + 大小 + 切分参数。

    同一个文件重复上传（Streamlit 每次 rerun 都会重新给出文件对象）不会被重复处理；
    改了切分参数则视为新任务，会重新切分入库（Chroma 为 upsert，不会产生脏数据）。
    """
    raw = f"{file.name}|{getattr(file, 'size', 0)}|{chunk_size}|{chunk_overlap}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def save_upload(file) -> Path:
    """把上传文件落到 ``data/raw/``，返回本地路径。"""
    settings.ensure_dirs()
    target = settings.raw_dir / Path(file.name).name
    target.write_bytes(file.getbuffer())
    return target


def auto_ingest(files, chunk_size: int, chunk_overlap: int) -> None:
    """上传即入库：解析 → 切分 → 写入向量库，并逐文件汇报进度。"""
    ingested: dict = st.session_state.ingested
    todo = [
        (fp, file)
        for file in files
        if (fp := fingerprint(file, chunk_size, chunk_overlap)) not in ingested
    ]
    if not todo:
        return

    with st.status(f"正在解析 {len(todo)} 个文件并写入向量库……", expanded=True) as status:
        ok = 0
        failed = 0
        total_chunks = 0

        for fp, file in todo:
            st.markdown(f"📄 `{file.name}`")
            try:
                path = save_upload(file)
                docs = load_and_split(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                if not docs:
                    raise ValueError("未解析出文本（可能是扫描件或空文件）")

                written = add_docs_to_db(docs)
                ingested[fp] = {"name": file.name, "chunks": written}
                ok += 1
                total_chunks += written
                st.markdown(f"　　✅ 已入库 {written} 个知识分片")
            except Exception as exc:  # noqa: BLE001 —— 单个文件失败不影响其它文件
                failed += 1
                st.markdown(f"　　❌ 入库失败：{exc}")

        if ok:
            label = f"入库完成：{ok} 个文件 / {total_chunks} 个知识分片"
            if failed:
                label += f"（{failed} 个失败）"
            status.update(label=label, state="complete", expanded=False)
        else:
            status.update(label="入库失败，请查看上方提示", state="error", expanded=True)


def turn_tags_into_html(text: str) -> str:
    """把独立成行的「【来源：…】」标注换成带样式的块，正文保持不变。"""
    lines = []
    for line in (text or "").split("\n"):
        stripped = line.strip()
        if stripped.startswith("【来源：") and stripped.endswith("】"):
            lines.append(f'<div class="kb-cite">📎 {stripped[1:-1]}</div>')
        else:
            lines.append(line)
    return "\n".join(lines)


def render_answer(text: str) -> None:
    """渲染答案正文（含逐段来源标注）。"""
    st.markdown(turn_tags_into_html(text), unsafe_allow_html=True)


def cited_paragraphs(segments: list[dict] | None) -> dict[int, list[int]]:
    """反查：每个片段编号分别被答案的第几段引用。"""
    mapping: dict[int, list[int]] = {}
    for segment in segments or []:
        for index in segment.get("indexes") or []:
            mapping.setdefault(int(index), []).append(segment.get("paragraph"))
    return mapping


def render_sources(sources: list[dict] | None, segments: list[dict] | None = None) -> None:
    """把引用来源渲染成带边框的卡片列表（含重排分与被引用段落）。"""
    if not sources:
        return
    cited = cited_paragraphs(segments)
    with st.expander(f"📎 引用来源 · {len(sources)} 个片段", expanded=False):
        for src in sources:
            with st.container(border=True):
                head = f"**[{src.get('index')}] {src.get('citation') or src.get('file_name')}**"
                rerank_score = src.get("rerank_score")
                vector_score = src.get("vector_score", src.get("score"))
                if rerank_score is not None:
                    head += f"　`重排分 {rerank_score:.4f}`"
                    if isinstance(vector_score, (int, float)):
                        head += f"　`召回相似度 {vector_score:.4f}`"
                    recall_rank = src.get("recall_rank")
                    if recall_rank and recall_rank != src.get("index"):
                        total = src.get("recall_total")
                        head += f"　`粗排第 {recall_rank}" + (f"/{total} 位" if total else " 位") + "`"
                elif isinstance(src.get("score"), (int, float)):
                    head += f"　`相似度 {src['score']:.4f}`"
                st.markdown(head)

                paragraphs = cited.get(src.get("index"))
                if paragraphs:
                    st.caption("被引用段落：" + "、".join(f"第 {n} 段" for n in paragraphs))

                snippet = (src.get("snippet") or src.get("preview") or "").strip()
                if snippet:
                    st.caption(snippet)


def answer_question(
    question: str,
    top_k: int,
    threshold: float | None,
    use_rerank: bool,
    recall_k: int | None,
    inline_citations: bool,
) -> None:
    """一轮问答：渲染用户气泡 → 两阶段检索 + 流式生成 → 逐段溯源 → 写入会话历史。"""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    sources: list[dict] = []
    result: dict = {}
    raw = ""
    display = ""

    with st.chat_message("assistant"):
        # 阶段提示：用一个可展开的状态块展示「检索 → 生成」的进度
        trace = st.status("正在检索知识库……", expanded=False)
        trace.write(
            f"检索参数：TOP_K = {top_k}，相似度下限 = "
            f"{'不过滤' if threshold is None else threshold}，"
            f"重排序 = {rerank_label if use_rerank else '关闭'}"
            + (f"（粗排候选 {recall_k} 条）" if use_rerank and recall_k else "")
        )

        placeholder = st.empty()
        buffer: list[str] = []

        def paint(finished: bool) -> None:
            """把缓冲区画出来：未结束时只标注「已写完」的段落，末尾加光标。"""
            text = "".join(buffer)
            if not text:
                return
            if not inline_citations:
                placeholder.markdown(text + ("" if finished else " ▍"))
                return
            body = annotate_answer(text, sources) if finished else annotate_partial(text, sources)
            placeholder.markdown(turn_tags_into_html(body) + ("" if finished else " ▍"), unsafe_allow_html=True)

        try:
            for kind, payload in kb_chat_stream(
                question,
                top_k=top_k,
                score_threshold=threshold,
                rerank=use_rerank,
                recall_k=recall_k if use_rerank else None,
            ):
                if kind == "sources":
                    sources.clear()
                    sources.extend(payload)
                    if payload:
                        best = payload[0]
                        detail = f"召回 {len(payload)} 个相关片段，最相关：{best['citation']}"
                        if best.get("rerank_score") is not None:
                            model = best.get("rerank_model") or rerank_label
                            detail += f"（{model} 重排分 {best['rerank_score']:.4f}）"
                        trace.write(detail)
                    else:
                        trace.write("未召回任何片段")
                elif kind == "delta":
                    buffer.append(payload)
                    paint(finished=False)
                elif kind == "end":
                    result.update(payload)

            raw = "".join(buffer)
            if raw and inline_citations:
                # end 事件里已经算好了带标注的完整版本，直接复用（单一事实来源）
                display = result.get("answer_annotated") or annotate_answer(raw, sources)
            else:
                display = raw
            placeholder.markdown(
                turn_tags_into_html(display) if inline_citations else display,
                unsafe_allow_html=inline_citations,
            )

            if result.get("found"):
                info = result.get("retrieval") or {}
                label = f"已依据 {len(sources)} 个知识片段作答"
                if info.get("rerank"):
                    label += (
                        f"（{info.get('rerank_model')} 重排：{info.get('candidates')} 条候选"
                        f" → 保留 {info.get('returned')} 条）"
                    )
                trace.update(label=label, state="complete", expanded=False)
            else:
                trace.update(label="知识库中没有检索到相关内容", state="error", expanded=False)
        except Exception as exc:  # noqa: BLE001
            trace.update(label=f"生成失败：{exc}", state="error", expanded=True)
            st.error(f"生成失败：{exc}")

        if result and not result.get("found"):
            st.info("小贴士：确认相关文档已上传入库；也可以调大 TOP_K，或把相似度下限设为 0 再试。")

        # 溯源体检：模型没按要求标编号时如实告知，别让用户以为「没有来源」
        segments = result.get("segments") or []
        if result.get("found") and inline_citations and segments and not any(s.get("tag") for s in segments):
            st.warning("本次回答未包含引用编号，无法逐段定位来源，请以下方「引用来源」列表为准。")

        render_sources(sources, segments)

    if raw:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": display or raw,
                "sources": sources,
                "segments": result.get("segments") or [],
            }
        )


# ----------------------------------------------------------------------
# 会话状态
# ----------------------------------------------------------------------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("ingested", {})

configured = settings.is_configured()

# ----------------------------------------------------------------------
# 侧边栏
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📚 企业知识库")
    st.caption(
        f"对话模型 `{settings.llm_model}`　向量模型 `{settings.kb_embedding_model}`\n\n"
        f"重排序模型 `{settings.rerank_model}`"
    )

    if configured:
        st.markdown(
            f'<span class="kb-ring"></span>服务已就绪', unsafe_allow_html=True
        )
    else:
        st.markdown('<span class="kb-dot"></span>未配置 API Key', unsafe_allow_html=True)
        st.error("未检测到有效的 `DASHSCOPE_API_KEY`。", icon="🔑")
        st.code("DASHSCOPE_API_KEY=sk-你的Key", language="bash")
        st.caption("在项目根目录的 `.env` 中填写后刷新页面。")

    st.divider()

    # ---------- 入库参数（放在上传之前，便于先定好再传） ----------
    with st.expander("入库参数（仅影响之后上传的文件）", expanded=False):
        chunk_size = int(
            st.number_input(
                "分片大小（字符）", min_value=200, max_value=2000,
                value=DEFAULT_CHUNK_SIZE, step=50,
            )
        )
        chunk_overlap = int(
            st.number_input(
                "分片重叠（字符）", min_value=0, max_value=500,
                value=DEFAULT_CHUNK_OVERLAP, step=10,
            )
        )
        if chunk_overlap >= chunk_size:
            chunk_overlap = max(0, chunk_size // 10)
            st.warning(f"重叠不能大于等于分片大小，已自动改为 {chunk_overlap}。")

    # ---------- 1. 上传并自动入库 ----------
    st.markdown("**1 · 上传文档**")
    uploaded = st.file_uploader(
        f"支持 {' / '.join(t.upper() for t in ACCEPTED_TYPES)}，可多选；上传后自动解析入库",
        type=ACCEPTED_TYPES,
        accept_multiple_files=True,
        disabled=not configured,
    )

    if uploaded and configured:
        auto_ingest(uploaded, chunk_size, chunk_overlap)

    if st.session_state.ingested:
        with st.expander(f"已入库文件 · {len(st.session_state.ingested)} 个", expanded=False):
            for item in st.session_state.ingested.values():
                st.markdown(f"- `{item['name']}`　{item['chunks']} 分片")

    # ---------- 2. 知识库状态 ----------
    st.divider()
    st.markdown("**2 · 知识库状态**")
    chunk_count = 0
    # 实际生效的排序模型（配置的模型在当前账号下不可用时，会自动回退到备用模型）
    rerank_label = settings.rerank_model
    if not configured:
        st.caption("配置 API Key 后显示。")
    else:
        try:
            info = db_info()
            chunk_count = int(info["chunks"])
            rerank_label = info.get("rerank_model") or settings.rerank_model
            st.metric("已入库知识分片", chunk_count)
            st.caption(
                f"集合 `{info['collection']}`　相似度 `{info['distance']}`\n\n"
                f"重排序 `{info['rerank_model'] or '已关闭'}`\n\n"
                f"目录 `{info['persist_dir']}`"
            )
        except Exception as exc:  # noqa: BLE001
            st.caption(f"知识库暂不可用：{exc}")

    # ---------- 3. 检索参数 ----------
    st.divider()
    st.markdown("**3 · 检索参数**")
    top_k = st.slider("召回片段数 TOP_K", min_value=1, max_value=10, value=DEFAULT_TOP_K)
    threshold_value = st.slider(
        "相似度下限（0 = 不过滤）", min_value=0.0, max_value=1.0, value=0.0, step=0.02
    )
    threshold: float | None = threshold_value or None
    st.caption("余弦相似度越大越相关，建议取值 0.30 ~ 0.40；作用在向量粗排阶段。")

    use_rerank = st.checkbox(
        f"启用重排序（{rerank_label}）",
        value=settings.rerank_enabled,
        help=(
            "两阶段检索：先用向量多召回一批候选，再用 gte-rerank 交叉编码器逐对打分精排，"
            "显著提升 Top-1 命中率。模型在当前账号下不可用时会自动回退到备用模型；"
            "调用失败则降级为向量粗排结果，不影响问答。"
        ),
    )
    auto_recall = settings.rerank_recall_k or max(
        top_k * settings.rerank_recall_multiplier, 10
    )
    recall_k = st.slider(
        "粗排候选数（交给重排序精排）",
        min_value=max(2, top_k),
        max_value=30,
        value=min(30, max(auto_recall, top_k)),
        disabled=not use_rerank,
        help="候选越多越可能捞到真正有用的片段，但重排序耗时与调用量也随之增加。",
    )

    inline_citations = st.checkbox(
        "答案内联标注引用来源",
        value=True,
        help="在每段答案末尾追加【来源：文件（第 X 页）】，编号由系统按检索结果映射，模型无法伪造。",
    )

    # ---------- 4. 会话与数据 ----------
    st.divider()
    st.markdown("**4 · 会话与数据**")
    if st.button("清空对话记录", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    with st.expander("危险操作", expanded=False):
        confirmed = st.checkbox("我确认要清空知识库（不可恢复）")
        if st.button(
            "清空知识库", use_container_width=True, disabled=not confirmed,
            help="删除 ./chroma_kb 中的全部向量数据",
        ):
            try:
                reset_db()
                st.session_state.ingested = {}
                st.session_state.messages = []
                flash("知识库已清空，可以重新上传文档了。", "success")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"清空失败：{exc}")


# ----------------------------------------------------------------------
# 主区：对话窗口
# ----------------------------------------------------------------------
st.title("企业知识库智能问答")
st.caption(
    "基于 RAG 两阶段检索：向量粗排召回候选 → gte-rerank 精排 → 通义千问严格依据原文作答，"
    "并在每段答案末尾标注引用的是哪个文件的哪一页。"
)
render_flash()

if chunk_count == 0 and st.session_state.messages:
    st.info("知识库还是空的 —— 请在左侧上传 PDF / TXT / DOCX 文档并自动入库后重新提问。")

# ---- 空态引导 ----
if not st.session_state.messages:
    st.markdown(
        """
        <div class="kb-hero">
          <h3>👋 开始你的第一个提问</h3>
          <p>
            ① 在左侧上传企业文档（<b>PDF / TXT / DOCX</b>），上传后自动解析、切分并写入向量库；<br>
            ② 在下方输入框提问，系统会先<b>向量粗排</b>多召回候选，再用 <b>gte-rerank 精排</b>挑出真正有用的片段；<br>
            ③ 通义千问只依据原文作答，且<b>每段末尾都标注来源文件与页码</b>，可展开「引用来源」逐条核对原文片段。
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.caption("试试这些问题：")
    columns = st.columns(len(EXAMPLE_QUESTIONS))
    for column, example in zip(columns, EXAMPLE_QUESTIONS):
        if column.button(example, use_container_width=True, disabled=not configured):
            st.session_state.pending = example
            st.rerun()

# ---- 历史消息（多轮对话） ----
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_answer(message["content"])
        else:
            st.markdown(message["content"])
        render_sources(message.get("sources"), message.get("segments"))

# ---- 输入框 ----
question = st.chat_input(
    "请输入你的问题，例如：年假有多少天？",
    disabled=not configured,
)
if not question:
    # 点击示例问题后，用 pending 补位
    question = st.session_state.pop("pending", None)
else:
    st.session_state.pop("pending", None)

if question:
    answer_question(
        question,
        top_k=top_k,
        threshold=threshold,
        use_rerank=use_rerank,
        recall_k=recall_k,
        inline_citations=inline_citations,
    )
