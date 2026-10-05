# 项目长期记忆

## 项目定位
企业知识库 RAG 问答系统。用户上传企业内部文档（PDF/Word/TXT/MD），系统解析切分向量化入库，
回答问题时先**两阶段检索**（向量粗排 → `gte-rerank` 精排）再让大模型依据原文作答，
并给出引用来源 + **每段末尾标注【来源：文件（第 X 页）】**。

## 技术栈约定
- LLM / Embedding / 重排序：阿里云百炼 DashScope —— **直接调用 dashscope SDK**
  （`TextEmbedding` / `Generation` / `TextReRank`），不引入 langchain-community 的 DashScope 封装。
- 文档加载：使用 langchain-community 提供的官方 DocumentLoader
  （PyPDFLoader / TextLoader / Docx2txtLoader），统一入口 `src/doc_processor.py::load_and_split`。
- 向量库：Chroma，本地持久化到 `vector_store/`。
- 知识库服务（Agent 应用场景）：`src/kb.py` —— `text-embedding-v4` + Chroma 持久化到 `./chroma_kb`，
  核心函数 `add_docs_to_db(docs)` / `search_from_db(query, top_k=3, rerank=?, recall_k=?)` /
  `kb_chat(query)` / `kb_chat_stream(query)`（流式，事件协议 `sources → delta… → end`，
  未变）；集合用 cosine 空间。返回值恒含 `answer_annotated` / `segments` / `retrieval`。
  与主链路（v3 + `vector_store/`）并存：v3/v4 向量不在同一语义空间，不可混用。
- 重排序：`src/reranker.py`（`DashScopeReranker` / `get_reranker()`）。约定三条铁律：
  ① 候选数 ≤ Top-K 时跳过重排；② 任何排序异常都**降级**为粗排顺序、不抛错；
  ③ 模型不可用（400/403/404，如 `gte-rerank` 未开通返回 403）时按 `RERANK_FALLBACK_MODELS`
  自动回退并记住可用模型，实际模型名写在 `rerank_model` / `active_model` 里对外暴露。
- 溯源：`src/citations.py`。**编号→文件名/页码的映射由系统完成，模型只提供 `[n]` 编号**，
  编号对不上就不标（宁缺毋滥）；`annotate_partial()` 只标注已写完的段落，供流式界面用。
- 切分：`langchain-text-splitters` 的 RecursiveCharacterTextSplitter，分隔符已按中文习惯定制
  （`\n\n → \n → 。！？；， → 空格`）。通用函数默认 chunk_size=500 / chunk_overlap=50。
- 前端：Streamlit，两套入口**并存**：`app.py`（主链路 v3 + `vector_store/`）、
  `app_agent.py`（Agent 应用 v4 + `./chroma_kb`）。新增界面优先放进 `app_agent.py`；
  流式回答不用 `st.write_stream`，用 `placeholder = st.empty()` 才能边流式边逐段标注来源。
- 配置：全部环境变量集中在根目录 `.env`，由 `config.py` 的 `settings` 单例读取，业务代码不直接碰 os.environ。

## 代码约定
- `src/` 下每个模块单一职责：document_loader / text_splitter / embeddings / vector_store /
  retriever / rag_chain / kb / reranker / citations。
- 入库用「来源+页码+内容」的 MD5 作为稳定 id，保证重复入库不产生脏数据。
- 回答必须标注引用来源；检索不到内容时返回固定话术，不允许模型自由发挥。
- 检索命中（hit）结构对所有调用方保持一致：未重排时 `vector_score/rerank_score/rerank_model/
  recall_rank/recall_total` 显式为 None，界面/CLI 不做分支判断。
- 中文字符串、注释、文档统一用中文。
- 新增配置项一律先进 `config.py` 的 `Settings`，再同步 `.env` / `.env.example` / README 表格。

## 环境
- 虚拟环境：项目根目录 `.venv`，Python 3.13.14。
- 启动 Web：`streamlit run app.py`（主链路） / `streamlit run app_agent.py`（Agent 应用）；
  批量入库：`python scripts/ingest.py` 或 `python src/kb.py --ingest <路径>`。
- 验证手段：`python tests/test_smoke.py`、`python tests/test_kb.py`（均不消耗额度）；
  界面层用 `streamlit.testing.v1.AppTest` 真实执行脚本做冒烟（可配合假 uploader 与
  `dataclasses.replace(config.settings, ...)` 指向临时目录，避免污染真实库；
  也可用 `CHROMA_KB_DIR` / `KB_COLLECTION_NAME` 环境变量指向临时库）。
- `tests/test_kb.py` 里的 `search()` / `chat()` / `chat_stream()` 测试入口**默认 rerank=False**，
  需要验证重排的用例显式传 `rerank=True` + `patched_reranker(FakeReranker())`，避免单元测试触发真实调用。
