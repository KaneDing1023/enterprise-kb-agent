# 企业知识库 RAG

基于 **LangChain + Chroma + 阿里云百炼 DashScope（通义千问）** 搭建的企业内部知识库问答系统。  
上传 PDF / Word / Markdown 文档后，系统自动解析、切分、向量化入库；提问时采用**两阶段检索**  
（向量粗排召回候选 → `gte-rerank` 交叉编码器精排），再由大模型严格依据原文作答，  
并在**每段答案末尾标注引用的是哪个文件的哪一页**。

---

## 一、依赖包清单（pip 安装）

```bash
pip install -r requirements.txt
```

| 包名                         | 版本要求     | 作用                                                |
| -------------------------- | -------- | ------------------------------------------------- |
| `langchain`                | >=0.3.0  | RAG 编排框架                                          |
| `langchain-chroma`         | >=0.2.0  | LangChain 的 Chroma 向量库集成                          |
| `langchain-text-splitters` | >=0.3.0  | 文本切分器（中文分隔符已适配）                                   |
| `langchain-community`      | >=0.4.0  | 官方 DocumentLoader：PDF / TXT / DOCX 加载             |
| `docx2txt`                 | >=0.9.0  | `Docx2txtLoader` 读取 .docx 的底层依赖                   |
| `chromadb`                 | >=0.5.0  | 本地向量数据库（持久化到 `vector_store/`）                     |
| `dashscope`                | >=1.20.0 | 阿里云百炼 SDK：Embedding + 通义千问 LLM + `gte-rerank` 重排序 |
| `python-dotenv`            | >=1.0.0  | 读取 `.env` 配置                                      |
| `pypdf`                    | >=5.0.0  | PDF 解析                                            |
| `python-docx`              | >=1.1.0  | Word 解析                                           |
| `streamlit`                | >=1.40.0 | Web 问答界面                                          |

> 说明：Embedding、LLM 与重排序（`gte-rerank`）均直接调用 `dashscope` SDK  
> （不经过 `langchain-community` 的 DashScope 封装），**重排序不需要额外依赖**；  
> `langchain` 主包、`langchain-chroma` 会自动带上 `langchain-core`。  
> `langchain-community` 仅用于官方 DocumentLoader（`PyPDFLoader` / `TextLoader` / `Docx2txtLoader`）。

---

## 二、快速开始

```bash
# 1. 创建并激活虚拟环境
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置密钥：复制模板并填入真实 Key
copy .env.example .env          # Windows
# cp .env.example .env          # macOS / Linux
# 编辑 .env，把 DASHSCOPE_API_KEY 换成真实值
# 申请地址：https://bailian.console.aliyun.com/  ->  API-KEY 管理

# 4. 把文档放进 data/raw/，然后命令行批量入库
python scripts/ingest.py

# 5. 启动 Web 界面
streamlit run app.py          # 主链路版（text-embedding-v3 -> ./vector_store）
streamlit run app_agent.py    # Agent 应用版（text-embedding-v4 -> ./chroma_kb）
```

浏览器打开 <http://localhost:8501> 即可提问。也可以直接在界面左侧上传文档，上传后会自动解析入库。

### 常用命令

```bash
python scripts/ingest.py                     # 入库 data/raw 下全部文档
python scripts/ingest.py data/raw/员工手册.pdf # 入库指定文件
python scripts/ingest.py --reset             # 清空向量库后重新入库
python scripts/ingest.py --chunk-size 1000 --chunk-overlap 150
python src/kb.py --ingest data/raw/员工手册.pdf # 知识库场景入库（text-embedding-v4 -> ./chroma_kb）
python src/kb.py "年假有多少天？" --top-k 3      # 知识库场景检索（向量粗排 + gte-rerank 精排）
python src/kb.py "年假有多少天？" --no-rerank    # 只用向量粗排，关闭重排序
python src/kb.py --chat "年假有多少天？"          # 知识库场景问答（检索 + qwen-plus + 逐段溯源）
python tests/test_smoke.py                   # 冒烟测试（不消耗 API 额度）
python tests/test_kb.py                      # 知识库函数测试（不消耗 API 额度）
```

---

## 三、文档处理函数 `load_and_split()`

`src/doc_processor.py` 是统一的文档处理入口，内部使用 **LangChain 官方 DocumentLoader**，按扩展名分发：

| 格式             | 使用的 Loader       | 说明                                 |
| -------------- | ---------------- | ---------------------------------- |
| `.pdf`         | `PyPDFLoader`    | 逐页解析，`metadata["page"]` 带页码，便于标注引用 |
| `.txt` / `.md` | `TextLoader`     | 自动探测 UTF-8 / GB18030 编码，避免中文乱码     |
| `.docx`        | `Docx2txtLoader` | 抽取正文段落文本                           |

切分统一用 `RecursiveCharacterTextSplitter`，默认 `chunk_size=500`、`chunk_overlap=50`。

```python
from src.doc_processor import load_and_split

docs = load_and_split("data/raw/员工手册.pdf")   # 默认 500 / 50

for d in docs:
    print(d.metadata)
    # {'source': '...\\员工手册.pdf', 'page': 0, 'file_name': '员工手册.pdf',
    #  'file_type': 'pdf', 'chunk_index': 0, 'chunk_total': 12}
    print(d.page_content)
```

参数可按需覆盖：

```python
docs = load_and_split("data/raw/制度.docx", chunk_size=800, chunk_overlap=100)
```

命令行单独试跑：

```bash
python src/doc_processor.py data/raw/员工手册.pdf
```

**关于 chunk_overlap 的一个细节**：配置值是 50，但实测相邻分片的实际重叠约为 38 字符。  
这是 `RecursiveCharacterTextSplitter` 的预期行为——它只按分隔符边界回退，不会为了凑满 50  
而把一个完整的句子切成半个。所以实际重叠量在 `0 ~ chunk_overlap` 之间浮动，属于正常现象。

---

## 四、向量知识库核心函数 `add_docs_to_db()` / `search_from_db()` / `kb_chat()`

`src/kb.py` 封装了「百炼 `text-embedding-v4` + Chroma 本地持久化 + `gte-rerank` 两阶段检索」的  
知识库读写与问答，对外暴露四个核心函数，向量数据落盘在 `./chroma_kb`：

```python
from src.doc_processor import load_and_split
from src.kb import add_docs_to_db, search_from_db

# 1) 入库：把切分后的 Document 批量写入向量数据库，返回写入的切片数
docs = load_and_split("data/raw/员工手册.pdf")
add_docs_to_db(docs)

# 2) 检索：向量粗排召回候选 -> gte-rerank 精排 -> 返回最相关的 3 条片段
for hit in search_from_db("年假有多少天？", top_k=3, rerank=True):
    print(hit["rank"], hit["citation"], hit["rerank_score"], hit["vector_score"])
    print(hit["content"])
```

`search_from_db()` 的返回结构（按最终相关性从高到低）：

| 字段             | 说明                                     |
| -------------- | -------------------------------------- |
| `rank`         | 最终排名，从 1 开始（重排后重新编号）                   |
| `content`      | 文档片段正文                                 |
| `score`        | 最终排序依据：重排时为 `gte-rerank` 相关性分，否则为余弦相似度 |
| `vector_score` | 向量粗排的余弦相似度 —— 便于对比「重排把哪条捞上来了」          |
| `rerank_score` | `gte-rerank` 相关性分数；未重排或重排失败时为 `None`   |
| `rerank_model` | 本次**实际生效**的排序模型名（可能因回退与配置值不同）          |
| `recall_rank`  | 该片段在向量粗排阶段的排名；未重排时为 `None`             |
| `recall_total` | 进入精排的候选总数（含被淘汰的）                       |
| `file_name`    | 来源文件名                                  |
| `page`         | 来源页码；TXT / DOCX / MD 没有页码概念时为 `None`   |
| `source`       | 来源文件路径                                 |
| `citation`     | 可直接展示的引用标签，如 `员工手册.pdf（第 3 页）`         |

实现要点：

- **向量模型**：`text-embedding-v4`（Qwen3-Embedding 系列，默认 1024 维，单次最多 10 条文本）。  
  通过 LangChain 的 `Embeddings` 接口接入，底层直接调用 dashscope SDK，  
  并开启 `text_type` 非对称检索优化（文档侧 `document`、查询侧 `query`）。
- **本地持久化**：Chroma 落盘在 `./chroma_kb`，进程退出后数据仍在磁盘上，下次启动直接复用。
- **两阶段检索**：向量粗排多召回候选，再由 `gte-rerank` 精排（详见 [4.1](#41-两阶段检索与重排序gte-rerank)）。
- **幂等去重**：以「来源 + 页码 + 块号 + 内容」的 MD5 作为稳定 id（写入为 upsert 语义），  
  同一份文档重复入库不会产生脏数据。
- **空库保护**：库内无内容时 `search_from_db()` 直接返回 `[]`；`top_k` 超过库内总数会自动收敛，不会报错。

命令行试跑：

```bash
python src/kb.py --ingest data/raw/员工手册.pdf  # 入库（加 --reset 先清空）
python src/kb.py "年假有多少天？" --top-k 3        # 检索（默认已开重排，--no-rerank 可关闭）
python src/kb.py "年假有多少天？" --recall-k 20    # 指定粗排候选数
```

### 4.1 两阶段检索与重排序（`gte-rerank`）

**为什么需要重排序**：向量检索算的是「问题与片段的整体语义接近程度」，而不是  
「这个片段对回答该问题有多大用」。两者在长文档、口语化提问、多个相似主题并存的场景下经常不一致 ——  
真正能回答问题的片段可能排在第 3~8 位，排第 1 的只是一段「话题相近但没答到点上」的文字。

`gte-rerank` 是交叉编码器（cross-encoder）：把「问题 + 片段」拼在一起**逐对**打分，  
精度明显高于向量内积，缺点是必须逐条计算、无法预建索引。所以这里用两阶段：

```
用户问题
  → 向量粗排    Chroma 召回 recall_k 条候选（默认 TOP_K × 3，且不低于 10 条）
  → 阈值过滤    按「相似度下限」丢掉明显不相关的候选（作用在余弦相似度上）
  → gte-rerank  候选数多于 Top-K 时逐对打分重新排序
  → 取 Top-K    重新编号 rank = 1..K
```

```python
from src.kb import search_from_db

# 默认：跟随 RERANK_ENABLED 配置（默认开启），自动决定候选数
hits = search_from_db("年假有多少天？", top_k=3)

# 显式开/关，并指定粗排候选数（候选越多越可能捞到真正有用的片段）
hits = search_from_db("年假有多少天？", top_k=3, rerank=True, recall_k=20)
hits = search_from_db("年假有多少天？", top_k=3, rerank=False)

for hit in hits:
    print(hit["rank"], hit["citation"], hit["rerank_score"], hit["vector_score"])
```

三条设计上的取舍：

- **候选数不超过 Top-K 时跳过重排**：库里只有 3 条、你却只要 3 条时，重排没有任何择优空间，  
  直接省掉这次调用。
- **失败即降级**：排序服务不可用（网络 / 额度 / 限流）时**不抛异常**，打日志后退回向量粗排顺序，  
  问答链路照常可用 —— 排序是优化项，不该成为单点故障。
- **模型自动回退**：百炼上模型名与账号权限绑定 —— 例如 `gte-rerank` 若未在控制台开通，  
  会返回 `403 Access denied`，此时自动改用 `RERANK_FALLBACK_MODELS`（默认 `gte-rerank-v2`），  
  并**记住**可用的那个，后续调用不再重试失败模型。实际生效的模型名会写在  
  `rerank_model` 字段里，界面加载状态也会显示，不会出现「说的是 A、用的是 B」。

> 如果你的账号已开通 `gte-rerank`，把 `.env` 的 `RERANK_FALLBACK_MODELS` 留空即可。

### 4.2 逐段溯源：每段答案末尾标注来源文件与页码

**问题**：大模型答案里通常只有 `[1]` `[2]` 这样的编号，用户还得自己去「引用来源」面板里对号入座；  
而且编号是模型自由生成的，一旦写错，很容易把 A 文件的结论挂到 B 文件头上。

**做法**：模型只负责给编号，编号 → 真实文件名/页码的映射完全由系统完成（`src/citations.py`）：

```
提示词要求：每一段末尾必须标注片段编号（如 [1]、[1][3]），编号必须来自参考资料
    ↓
按空行把答案切段 → 逐段解析编号 → 用编号在检索结果里取真实来源
    ↓
每段末尾追加【来源：员工手册.pdf（第 2 页）】，并返回逐段结构化的 segments
```

```python
from src.kb import kb_chat

result = kb_chat("年假有多少天？")

print(result["answer_annotated"])
# 员工累计工作满一年不满十年的，年假为五天。
#
# 【来源：员工手册.pdf（第 2 页）】
#
# 年假申请需在系统提交，经主管审批后生效。
#
# 【来源：员工手册.pdf（第 3 页）】

for seg in result["segments"]:
    print(seg["paragraph"], seg["indexes"], seg["tag"])
    # 1 [2] 【来源：员工手册.pdf（第 2 页）】
    # 2 [1] 【来源：员工手册.pdf（第 3 页）】
```

`segments` 每项的结构：

| 字段          | 说明                                          |
| ----------- | ------------------------------------------- |
| `paragraph` | 段号，从 1 开始                                   |
| `text`      | 去掉编号标记后的段落正文（编号已转成来源标注）                     |
| `raw`       | 原始段落文本（保留编号，便于排查）                           |
| `indexes`   | 该段引用的片段编号（已过滤掉模型编造的、不存在的编号）                 |
| `citations` | 对应的来源条目（含文件名 / 页码 / 对应片段 `snippet`，可直接展示原文） |
| `tag`       | 段落末尾要追加的标注文本，如 `【来源：员工手册.pdf（第 2 页）】`       |

两条原则：

- **系统映射，模型无法伪造**：文件名与页码取自检索命中的元数据，模型只提供编号。  
  编号对不上就当作没标（`indexes` 里不会出现不存在的编号），宁可空着也不猜。
- **标记编号的段落才标注**：正文里没写编号的段落不加标注，界面会提示  
  「本次回答未包含引用编号，请以引用来源列表为准」，不制造虚假的确定性。

流式场景（界面打字机效果）用 `annotate_partial()`：只给**已经写完的段落**补标注，  
最后一段保持原样，等生成结束后再用 `annotate_answer()` 换成完整版本，不会出现「标了一半的标注」。

### 4.3 问答函数 `kb_chat()`

`kb_chat(query)` 把「两阶段检索 → 拼上下文 → 调用通义千问 `qwen-plus` 生成回答 → 逐段溯源」  
串成一条完整 RAG 链路：

```python
from src.kb import kb_chat

result = kb_chat("年假有多少天？", top_k=3, rerank=True)

print(result["answer"])             # 模型生成的回答原文（含 [1] 编号）
print(result["answer_annotated"])   # 每段末尾已标注【来源：文件（第 X 页）】
for src in result["sources"]:       # 答案引用的文档来源
    print(src["index"], src["citation"], src["rerank_score"], src["vector_score"])
```

返回结构为一个字典：

| 字段                 | 说明                                                                                                                                                                                      |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `query`            | 原始问题（已去首尾空白）                                                                                                                                                                            |
| `answer`           | 模型生成的回答原文；检索不到内容时为固定话术                                                                                                                                                                  |
| `answer_annotated` | 逐段溯源版回答（每段末尾追加【来源：…】）                                                                                                                                                                   |
| `segments`         | 逐段溯源结构，见 [4.2](#42-逐段溯源每段答案末尾标注来源文件与页码)                                                                                                                                                 |
| `sources`          | 引用来源列表，每项含 `index` / `file_name` / `page` / `source` / `citation` / `score` / `vector_score` / `rerank_score` / `rerank_model` / `recall_rank` / `recall_total` / `preview` / `snippet` |
| `hits`             | 原始检索命中（结构同 `search_from_db()`），便于二次加工                                                                                                                                                   |
| `retrieval`        | 本轮检索概况：`rerank` / `rerank_model` / `candidates` / `returned`                                                                                                                            |
| `found`            | 是否基于知识库给出了回答（检索到片段即为 `True`）                                                                                                                                                            |

**防幻觉设计**：

- **严格系统提示词**：明确要求模型「只使用检索到的参考资料作答，禁止使用参考资料之外的任何知识，  
  禁止编造、猜测、推测」；参考资料按 `[1]`、`[2]` 编号并标注来源，模型需在**每一段末尾**标注对应编号。
- **资料不足即拒答**：若参考资料不足以回答，模型必须回复「无法从知识库中获取相关信息」。
- **空检索短路**：`search_from_db()` 一条都没命中时，不再调用大模型，直接返回固定话术  
  （见常量 `KB_NO_ANSWER`），既省额度又杜绝模型自由发挥。

命令行问答：

```bash
python src/kb.py --chat "年假有多少天？"            # 检索 + qwen-plus 生成 + 逐段来源标注
```

### 4.4 流式问答 `kb_chat_stream()`

`kb_chat_stream()` 与 `kb_chat()` 走同一条链路（同样的两阶段检索、同样的严格提示词），  
区别只是**边生成边返回**，供 Web 界面做打字机效果与分阶段加载提示：

```python
from src.kb import kb_chat_stream

for kind, payload in kb_chat_stream("年假有多少天？", top_k=3):
    if kind == "sources":          # 先给来源，界面可以先展示
        print([src["citation"] for src in payload])
    elif kind == "delta":          # 逐段文本，自行拼接
        print(payload, end="", flush=True)
    elif kind == "end":            # 最后一个事件，payload 结构同 kb_chat() 返回值
        print("\n", payload["found"], payload["answer_annotated"])
```


| `kind`    | `payload`                                                                        |
| --------- | -------------------------------------------------------------------------------- |
| `sources` | 引用来源列表，在开始生成回答**之前**抛出一次                                                         |
| `delta`   | 本次新增的回答片段（字符串）                                                                   |
| `end`     | 完整结果字典（结构同 `kb_chat()`，含 `answer_annotated` / `segments` / `retrieval`），恒为最后一个事件 |

知识库为空或检索不到内容时，事件序列为 `sources(空) → delta(固定话术) → end(found=False)`，  
全程不调用大模型。

### 4.5 Agent 应用 Web 界面 `app_agent.py`

`app_agent.py` 是上面这套知识库链路（`text-embedding-v4` + `./chroma_kb` + `qwen-plus`）  
的 Streamlit 界面，与主链路的 `app.py` 并存、互不干扰：

```bash
streamlit run app_agent.py      # 打开 http://localhost:8501
```

| 区域           | 能力                                                                                      |
| ------------ | --------------------------------------------------------------------------------------- |
| 左侧边栏 · 上传文档  | 支持 **PDF / TXT / DOCX / MD**，可多选；**上传后自动**解析 → 切分 → 写入向量库，并逐文件显示进度与结果                   |
| 左侧边栏 · 知识库状态 | 已入库分片数、集合名、相似度空间、重排序模型（实际生效的）、持久化目录                                                     |
| 左侧边栏 · 参数    | 入库分片大小 / 重叠；检索 `TOP_K`、相似度下限、**重排序开关与粗排候选数**、答案内联标注开关                                   |
| 左侧边栏 · 会话与数据 | 清空对话记录；「危险操作」下二次确认后清空知识库                                                                |
| 主区 · 对话      | 多轮对话、流式回答、**每段末尾内联来源标注**、可展开的**引用来源卡片**（文件名 / 页码 / 重排分 / 召回相似度 / 粗排位次 / 对应原文片段 / 被引用段落） |
| 主区 · 状态提示    | 可折叠的分阶段状态块：`正在检索知识库 → 已依据 N 个片段作答（gte-rerank 重排：10 条候选 → 保留 3 条）`；检索不到时提示调参方向           |
| 主区 · 空态      | 使用引导卡片 + 三个示例问题快捷按钮                                                                     |

实现要点：

- **上传即入库**：以「文件名 + 文件大小 + 切分参数」的 MD5 作为去重指纹，同一次会话内  
  重复 `rerun` 或重新上传同一文件都不会重复消耗 Embedding 额度；改了切分参数则视为新任务重新切分。
- **幂等**：写入走 `add_docs_to_db()`，Chroma 为 upsert 语义，重复入库不会产生脏数据。
- **单文件容错**：某个文件解析失败（如扫描版 PDF 无文本）只提示该文件，不影响其它文件。
- **流式 + 逐段标注**：答案正文用占位符逐字渲染，`annotate_partial()` 只给**已写完的段落**  
  补来源标注，生成结束后换成 `answer_annotated` 的完整版本，不会出现标了一半的标注。
- **诚实呈现**：检索不到内容时不调用大模型，直接展示固定话术并给出调参建议；  
  重排序被跳过 / 降级 / 回退到备用模型时，状态块里都能看到实际发生了什么。

---

## 五、目录结构

```
企业知识库 Agent/
├── .env                      # 真实密钥（已被 git 忽略，切勿提交）
├── .env.example              # 环境变量模板，随代码一起提交
├── .gitignore
├── requirements.txt          # 依赖清单
├── README.md
├── config.py                 # 全局配置：统一从 .env 读取
├── app.py                    # Streamlit 入口 · 主链路（上传 / 入库 / 问答）
├── app_agent.py              # Streamlit 入口 · Agent 应用（上传即入库 + 两阶段检索 + 逐段溯源）
│
├── src/                      # 核心业务代码
│   ├── __init__.py
│   ├── doc_processor.py      # ★ 通用入口 load_and_split()：官方 Loader 加载 + 切分
│   ├── document_loader.py    # 文档解析：PDF / DOCX / TXT / MD -> Document
│   ├── text_splitter.py      # 中文友好切分，带 chunk_index 元数据
│   ├── embeddings.py         # DashScope Embedding 封装（LangChain 接口）
│   ├── kb.py                 # ★ 知识库核心：两阶段检索 / kb_chat() / kb_chat_stream()
│   ├── reranker.py           # ★ 重排序封装：gte-rerank 精排（模型回退 + 失败降级）
│   ├── citations.py          # ★ 逐段溯源：编号 -> 文件名/页码映射 + 段落标注
│   ├── vector_store.py       # Chroma 持久化：写入 / 计数 / 清空
│   ├── retriever.py          # 相似度检索 + 上下文拼装 + 引用来源
│   └── rag_chain.py          # 检索增强生成（同步 / 流式）
│
├── scripts/
│   └── ingest.py             # 命令行批量入库脚本
│
├── data/
│   ├── raw/                  # 原始文档（知识源，放这里）
│   └── processed/            # 中间产物：解析结果、切分结果等
│
├── vector_store/             # 主链路 Chroma 持久化数据（可重建，已被 git 忽略）
│
├── chroma_kb/                # 知识库场景 Chroma 持久化数据（可重建，已被 git 忽略）
│
├── tests/
│   ├── test_smoke.py         # 冒烟测试
│   └── test_kb.py            # kb.py 单元测试（伪 Embedding + 伪 Reranker，不消耗额度）
│
└── .streamlit/
    └── config.toml           # 界面主题与上传大小限制
```

---

## 六、配置项说明（`.env`）

| 变量                         | 默认值                 | 说明                                                            |
| -------------------------- | ------------------- | ------------------------------------------------------------- |
| `DASHSCOPE_API_KEY`        | —                   | **必填**，百炼平台 API Key                                           |
| `EMBEDDING_MODEL`          | `text-embedding-v3` | 主链路（`vector_store/`）向量化模型                                     |
| `LLM_MODEL`                | `qwen-plus`         | 对话模型，可选 `qwen-max` / `qwen-turbo` / `qwen-long`               |
| `CHROMA_PERSIST_DIR`       | `vector_store`      | 主链路向量库持久化目录                                                   |
| `COLLECTION_NAME`          | `enterprise_kb`     | 主链路集合名称                                                       |
| `KB_EMBEDDING_MODEL`       | `text-embedding-v4` | 知识库场景（`src/kb.py`）向量化模型                                       |
| `CHROMA_KB_DIR`            | `./chroma_kb`       | 知识库场景持久化目录（`src/kb.py`）                                       |
| `KB_COLLECTION_NAME`       | `enterprise_kb`     | 知识库场景集合名称                                                     |
| `CHUNK_SIZE`               | `800`               | 单分片最大字符数                                                      |
| `CHUNK_OVERLAP`            | `120`               | 相邻分片重叠字符数                                                     |
| `TOP_K`                    | `4`                 | 每次召回的分片数                                                      |
| `RERANK_ENABLED`           | `true`              | 是否启用重排序（两阶段检索的精排阶段）                                           |
| `RERANK_MODEL`             | `gte-rerank`        | 排序模型（需在百炼控制台开通，未开通会返回 403）                                    |
| `RERANK_FALLBACK_MODELS`   | `gte-rerank-v2`     | 配置模型不可用时按顺序自动回退的备用模型（逗号分隔）                                    |
| `RERANK_RECALL_K`          | `0`                 | 向量粗排候选数；`0` = 自动（`TOP_K × RERANK_RECALL_MULTIPLIER`，不低于 10 条） |
| `RERANK_RECALL_MULTIPLIER` | `3`                 | 自动计算候选数时的倍数                                                   |

> 注意：`text-embedding-v4` 与 `text-embedding-v3` 的向量不在同一语义空间，即使维度相同也不能相互比较。  
> 因此 `src/kb.py`（v4）与主链路（v3）各自使用独立的持久化目录与集合，互不干扰。

---

## 七、实现要点

- **解析**：`pypdf` 按页解析 PDF（保留页码用于引用）；`python-docx` 同时提取正文段落与表格内容。
- **切分**：分隔符序列为 `段落 → 换行 → 。！？；，`，适配中文长句，避免在句子中间断开。
- **向量化**：`text-embedding-v4`（知识库场景）/ `text-embedding-v3`（主链路），均直接调用 dashscope SDK；  
  v4 开启 `text_type` 非对称检索优化，知识库集合使用 `cosine` 相似度空间。
- **两阶段检索（召回 → 精排）**：向量粗排多召回候选（默认 `TOP_K × 3`，不低于 10 条），  
  再用 `gte-rerank` 交叉编码器逐对打分重排，取最终 Top-K —— 提升 Top-1 命中率。  
  候选数不超过 Top-K 时跳过重排；调用失败降级为粗排顺序；模型未开通时自动回退到备用模型。
- **去重**：入库使用「来源 + 页码 + 内容」的 MD5 作为稳定 id，重复入库不会产生脏数据。
- **防幻觉**：Prompt 明确要求只依据参考资料作答，资料不足时回复「根据现有资料无法回答该问题」。  
  知识库问答链路 `kb_chat()` 的严格提示词进一步要求「只依据检索到的知识库原文作答，禁止编造」，  
  找不到时回复「无法从知识库中获取相关信息」。
- **可溯源**：提示词要求**每段末尾**标注片段编号，系统再把编号映射成真实的文件名与页码，  
  在每段末尾追加 `【来源：员工手册.pdf（第 2 页）】`，并返回逐段结构化的 `segments`；  
  同时提供「引用来源」面板，展示文件名、页码、重排分、召回相似度与对应原文片段。  
  **编号 → 文件的映射由系统完成，模型无法伪造来源。**

---

## 八、常见问题

**Q：提示 `缺少有效的 DASHSCOPE_API_KEY`？**  
A：检查 `.env` 中的 Key 是否已替换占位符，且文件位于项目根目录。

**Q：入库时报 `Arrearage` / `InvalidApiKey`？**  
A：百炼账号欠费或 Key 失效，到 <https://bailian.console.aliyun.com/> 检查。

**Q：日志出现 `排序模型 gte-rerank 在当前账号下不可用，已切换到 gte-rerank-v2`？**  
A：这是预期行为，不是报错。`gte-rerank` 需要先在百炼控制台**开通该模型**，未开通会返回  
`403 Access denied`；此时程序自动改用备用模型 `gte-rerank-v2` 并记住它，问答不受影响。  
想固定用某个模型，把 `.env` 的 `RERANK_MODEL` 改成它、并清空 `RERANK_FALLBACK_MODELS` 即可。

**Q：重排序会不会显著变慢 / 变贵？**  
A：每轮问答多一次排序调用（候选越多越慢），但比「让大模型读错上下文」便宜得多。  
想省额度可以调小侧边栏的「粗排候选数」，或直接关掉重排序开关（`RERANK_ENABLED=false`）。

**Q：回答里某段没有【来源：…】标注？**  
A：说明模型这一段没有输出引用编号，系统不会替它猜来源（宁缺毋滥），界面会提示以「引用来源」列表为准。  
可以调大 `TOP_K` / 粗排候选数，或换 `qwen-max` 这类更强的模型来提高标注率。

**Q：检索不到内容？**  
A：确认已成功入库（侧边栏「已入库分片」不为 0）；调大 `TOP_K`，或关闭相似度过滤。

**Q：想换向量模型？**  
A：改 `.env` 的 `EMBEDDING_MODEL`（或 `KB_EMBEDDING_MODEL`）后需重新入库  
（`python scripts/ingest.py --reset` 或 `python src/kb.py --reset --ingest <路径>`）——  
不同模型维度和空间不通用。

**Q：`src/kb.py` 和 `scripts/ingest.py` 有什么区别？**  
A：两者都做「切分 → 向量化 → 入库」，只是入口不同：`src/kb.py` 是给业务代码调用的  
核心函数（`text-embedding-v4` + `./chroma_kb` + `gte-rerank`），`scripts/ingest.py` 是主链路的  
命令行批量入库脚本（`EMBEDDING_MODEL` + `vector_store/`）。
