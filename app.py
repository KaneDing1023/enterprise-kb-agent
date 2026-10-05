"""企业知识库问答 —— Streamlit 前端。

启动：
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402
from src.document_loader import load_directory, load_document  # noqa: E402
from src.rag_chain import stream_answer  # noqa: E402
from src.text_splitter import split_documents  # noqa: E402
from src.vector_store import add_documents, collection_info, reset_collection  # noqa: E402

st.set_page_config(page_title="企业知识库问答", page_icon="📚", layout="wide")


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def save_uploaded_files(files) -> list[Path]:
    """把上传文件落到 data/raw，返回本地路径列表。"""
    settings.ensure_dirs()
    saved: list[Path] = []
    for file in files:
        target = settings.raw_dir / file.name
        target.write_bytes(file.getbuffer())
        saved.append(target)
    return saved


def ingest(paths: list[Path]) -> int:
    """加载 -> 切分 -> 写入向量库，返回写入分片数。"""
    documents = []
    for path in paths:
        documents.extend(load_document(path) if path.is_file() else load_directory(path))
    if not documents:
        return 0
    return add_documents(split_documents(documents))


# ----------------------------------------------------------------------
# 侧边栏
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("📚 企业知识库")
    st.caption(f"模型：{settings.llm_model}｜向量：{settings.embedding_model}")

    if not settings.is_configured():
        st.error("未检测到 DASHSCOPE_API_KEY，请先在 .env 中填写真实 Key。")
        st.code("DASHSCOPE_API_KEY=sk-你的Key", language="bash")

    st.divider()
    st.subheader("1. 上传文档")
    uploaded = st.file_uploader(
        "支持 PDF / DOCX / TXT / MD，可多选",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("保存并入库", type="primary", use_container_width=True):
        if not uploaded:
            st.warning("请先选择文件。")
        elif not settings.is_configured():
            st.error("请先配置 DASHSCOPE_API_KEY。")
        else:
            with st.spinner("正在解析、切分并向量化……"):
                try:
                    paths = save_uploaded_files(uploaded)
                    count = ingest(paths)
                    st.success(f"已入库 {count} 个知识分片。")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"入库失败：{exc}")

    if st.button("入库 data/raw 目录已有文件", use_container_width=True):
        with st.spinner("正在处理 data/raw ……"):
            try:
                count = ingest([settings.raw_dir])
                st.success(f"已入库 {count} 个知识分片。")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"入库失败：{exc}")

    st.divider()
    st.subheader("2. 知识库状态")
    try:
        info = collection_info()
        st.metric("已入库分片", info["chunks"])
        st.caption(f"集合：{info['collection']}")
        st.caption(f"目录：{info['persist_dir']}")
    except Exception as exc:  # noqa: BLE001
        st.caption(f"暂未初始化：{exc}")

    if st.button("清空知识库", use_container_width=True):
        try:
            reset_collection()
            st.success("已清空。")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"清空失败：{exc}")

    st.divider()
    st.subheader("3. 检索参数")
    top_k = st.slider("召回片段数 TOP_K", 1, 10, settings.top_k)
    use_threshold = st.checkbox("启用相似度过滤", value=False)
    threshold = st.slider("相似度阈值", 0.0, 1.0, 0.3, 0.05) if use_threshold else None

    if st.button("清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ----------------------------------------------------------------------
# 主区
# ----------------------------------------------------------------------
st.title("企业知识库智能问答")
st.caption("基于 RAG：先从企业文档中检索相关内容，再由通义千问依据原文作答并给出引用来源。")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("引用来源"):
                for src in message["sources"]:
                    page = f"（第 {src['page']} 页）" if src.get("page") else ""
                    st.markdown(f"**[{src['index']}] {src['file_name']}**{page}")
                    st.caption(src["preview"] + "……")

question = st.chat_input("请输入你的问题，例如：请假流程是怎样的？")

if question:
    if not settings.is_configured():
        st.error("请先在侧边栏配置 DASHSCOPE_API_KEY。")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            holder: dict = {"sources": []}

            def generator():
                for kind, payload in stream_answer(question, k=top_k, score_threshold=threshold):
                    if kind == "sources":
                        holder["sources"] = payload
                    elif kind == "delta":
                        yield payload

            try:
                reply = st.write_stream(generator()) or ""
            except Exception as exc:  # noqa: BLE001
                st.error(f"生成失败：{exc}")
                reply = ""

            sources = holder["sources"]
            if sources:
                with st.expander("引用来源"):
                    for src in sources:
                        page = f"（第 {src['page']} 页）" if src.get("page") else ""
                        st.markdown(f"**[{src['index']}] {src['file_name']}**{page}")
                        st.caption(src["preview"] + "……")

        if reply:
            st.session_state.messages.append(
                {"role": "assistant", "content": reply, "sources": sources}
            )
