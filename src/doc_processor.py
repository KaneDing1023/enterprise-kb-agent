"""通用文档处理模块：加载 PDF / TXT / DOCX 并切分为 Document 列表。

对外只暴露一个入口 :func:`load_and_split`，内部使用 **LangChain 官方 DocumentLoader**：

===========  ==========================================================
格式         使用的 Loader
===========  ==========================================================
``.pdf``     ``langchain_community.document_loaders.PyPDFLoader``
``.txt``     ``langchain_community.document_loaders.TextLoader``
``.docx``    ``langchain_community.document_loaders.Docx2txtLoader``
===========  ==========================================================

所有 Loader 都实现了统一的 ``BaseLoader`` 接口，因此可以放进同一张注册表里
按扩展名分发，新增格式只需要往 ``LOADER_REGISTRY`` 里加一条映射。

切分统一使用 ``RecursiveCharacterTextSplitter``，默认 ``chunk_size=500``、
``chunk_overlap=50``。

命令行试跑::

    python src/doc_processor.py data/raw/员工手册.pdf
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from pathlib import Path

# langchain-community 上游已进入 sunset（仅维护、不再新增功能），但截至目前它仍是
# PyPDFLoader / TextLoader / Docx2txtLoader 的唯一来源 —— langchain_classic 里没有这些
# Loader，官方也还没有对应的独立集成包（langchain-pypdf 不存在）。
# 这里定向屏蔽这条每次导入都会打印的 DeprecationWarning，避免污染日志；
# 迁移进展见 https://github.com/langchain-ai/langchain-community/issues/674
warnings.filterwarnings(
    "ignore",
    message=r".*langchain-community.*is being sunset.*",
    category=DeprecationWarning,
)

from langchain_community.document_loaders import (  # noqa: E402
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.document_loaders import BaseLoader  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 默认参数
# ----------------------------------------------------------------------

# 每个切片的目标字符数。500 字符大约对应中文 250~350 字，
# 既能保证语义完整，又不至于让检索粒度太粗。
DEFAULT_CHUNK_SIZE = 500

# 相邻切片之间重叠的字符数，用于避免关键句刚好被切断在两片交界处。
# 经验取值是 chunk_size 的 10% 左右。
DEFAULT_CHUNK_OVERLAP = 50

# 切分优先级：先从最"硬"的语义边界切，逐级降级到最细的字符级。
# 前四项是针对中文语料补的（。！？；，），LangChain 默认只认英文标点，
# 直接用在中文文档上会把整段话切成碎块。
DEFAULT_SEPARATORS = [
    "\n\n",  # 段落分隔（最理想，切出来的片语义最完整）
    "\n",    # 换行
    "。",    # 中文句号
    "！",    # 中文感叹号
    "？",    # 中文问号
    "；",    # 中文分号
    "，",    # 中文逗号
    " ",     # 英文空格
    "",      # 兜底：按字符硬切，保证一定能切到目标长度
]

# 文本编码探测顺序。中文 Windows 环境常见的 GBK/GB18030 排在 UTF-8 之后，
# 因为 UTF-8 校验更严格，先试它可以避免把 UTF-8 文件误判成 GBK。
_ENCODING_CANDIDATES = ("utf-8", "utf-8-sig", "gb18030", "big5")


# ----------------------------------------------------------------------
# 内部辅助函数
# ----------------------------------------------------------------------
def _detect_encoding(path: Path, sample_size: int = 64 * 1024) -> str:
    """探测文本文件的编码。

    只读取文件开头的一小段做试探，避免把整个大文件读进内存。
    依次尝试 UTF-8 / GB18030 等候选编码，全部失败则退回 UTF-8。

    Args:
        path: 文本文件路径。
        sample_size: 用于探测的字节数，默认 64KB。

    Returns:
        探测到的编码名称，可直接传给 ``TextLoader``。
    """
    sample = path.read_bytes()[:sample_size]
    for encoding in _ENCODING_CANDIDATES:
        try:
            sample.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            # 当前编码解不出来，换下一个继续试
            continue
    logger.warning("无法识别 %s 的编码，将按 utf-8 读取", path.name)
    return "utf-8"


def _build_loader(path: Path) -> BaseLoader:
    """根据文件扩展名构造对应的 LangChain DocumentLoader 实例。

    Args:
        path: 待加载的文件路径。

    Returns:
        尚未执行加载的 ``BaseLoader`` 实例。

    Raises:
        ValueError: 扩展名不在支持列表中。
    """
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        # PyPDFLoader 内部用 pypdf 逐页解析，每页产出一个 Document，
        # 并在 metadata 里带上 page（页码从 0 开始），方便回答时标注引用页码。
        return PyPDFLoader(str(path))

    if suffix in {".txt", ".md", ".markdown"}:
        # TextLoader 需要一个明确的编码才能正确读取中文；
        # 默认的 utf-8 碰到 GBK 文件会直接抛 UnicodeDecodeError。
        return TextLoader(str(path), encoding=_detect_encoding(path))

    if suffix == ".docx":
        # Docx2txtLoader 会抽取正文段落文本，同时也能抓到部分表格内容。
        return Docx2txtLoader(str(path))

    raise ValueError(
        f"不支持的文件类型：{suffix or '（无扩展名）'}。"
        f"当前支持：{', '.join(sorted(LOADER_REGISTRY))}"
    )


# 扩展名 -> Loader 工厂 的注册表。
# 想支持新格式（比如 .csv / .html）时，只需要在这里加一行映射即可，
# load_documents / load_and_split 都会自动生效。
LOADER_REGISTRY: dict[str, Callable[[Path], BaseLoader]] = {
    ".pdf": _build_loader,
    ".txt": _build_loader,
    ".md": _build_loader,
    ".markdown": _build_loader,
    ".docx": _build_loader,
}


def _normalize_metadata(doc: Document, path: Path) -> Document:
    """统一补齐 Document 的元数据字段，方便后续溯源与引用。

    不同 Loader 写入的 metadata 键名并不一致（比如 PDF 只有 ``source``，
    而我们需要统一的 ``file_name`` 来展示引用来源），这里做一次标准化。

    Args:
        doc: 单个已加载的 Document。
        path: 该文档的原始文件路径。

    Returns:
        补齐元数据后的同一个 Document 对象。
    """
    doc.metadata.update(
        {
            "source": str(path),              # 绝对路径，全局唯一
            "file_name": path.name,           # 文件名，用于界面展示
            "file_type": path.suffix.lower().lstrip("."),  # pdf / txt / docx
        }
    )
    return doc


# ----------------------------------------------------------------------
# 对外接口
# ----------------------------------------------------------------------
def load_documents(file_path: str | Path) -> list[Document]:
    """加载单个文档，返回切分**之前**的原始 Document 列表。

    通常不需要直接调用它，除非你想拿到未切分的全文。

    Args:
        file_path: 文件路径，支持 str 与 Path。

    Returns:
        原始 Document 列表。PDF 为「一页一个 Document」，TXT/DOCX 为整篇一个。

    Raises:
        FileNotFoundError: 路径不存在或不是一个文件。
        ValueError: 文件类型不受支持。
    """
    path = Path(file_path).expanduser().resolve()

    # ---- 1. 前置校验：把错误尽早暴露出来，附带清晰的提示 ----
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    if not path.is_file():
        raise ValueError(f"传入的不是一个文件：{path}")

    # ---- 2. 按扩展名挑选 Loader 并执行加载 ----
    loader = _build_loader(path)
    try:
        documents = loader.load()
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 —— 把底层库的原始报错包装成更好懂的提示
        raise RuntimeError(f"加载 {path.name} 失败：{exc}") from exc

    # ---- 3. 过滤空文档：扫描版 PDF 会出现「有页面但没有文本」的情况 ----
    documents = [doc for doc in (d for d in documents if d.page_content.strip())]
    if not documents:
        logger.warning(
            "%s 未解析出任何文本（可能是扫描件或空文件），已跳过", path.name
        )
        return []

    # ---- 4. 统一元数据 ----
    documents = [_normalize_metadata(doc, path) for doc in documents]

    logger.info("已加载 %s，得到 %d 个原始片段", path.name, len(documents))
    return documents


def load_and_split(
    file_path: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """加载文档并切分为适合向量化的 Document 列表（本模块的主入口）。

    处理流程::

        文件路径
          -> 按扩展名选择 LangChain DocumentLoader（PDF / TXT / DOCX）
          -> loader.load()  得到原始 Document
          -> RecursiveCharacterTextSplitter 递归切分
          -> 补充 chunk_index / chunk_total 元数据
          -> 返回 list[Document]

    Args:
        file_path: 文件路径，支持 PDF / TXT / MD / DOCX。
        chunk_size: 每个切片的最大字符数，默认 500。
        chunk_overlap: 相邻切片的重叠字符数，默认 50。

    Returns:
        切分后的 Document 列表。即使原文件为空也返回空列表，不会返回 None。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 文件类型不支持，或 chunk_overlap >= chunk_size。

    Example:
        >>> docs = load_and_split("data/raw/员工手册.pdf")
        >>> docs[0].page_content
        '第一章 总则\\n本公司员工考勤管理遵循……'
        >>> docs[0].metadata["file_name"]
        '员工手册.pdf'
    """
    # ---- 参数兜底：overlap 不允许大于等于 chunk_size，否则切分会死循环 ----
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap({chunk_overlap}) 必须小于 chunk_size({chunk_size})"
        )

    # ---- 第 1 步：加载原始文档 ----
    documents = load_documents(file_path)
    if not documents:
        return []

    # ---- 第 2 步：构造切分器 ----
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=DEFAULT_SEPARATORS,  # 使用中文优先的分隔符序列
        length_function=len,            # 以「字符数」而非 token 数衡量长度
        keep_separator=True,            # 保留 。！？ 等分隔符，避免句子读起来断头
    )

    # ---- 第 3 步：执行切分。split_documents 会自动继承并复制原 metadata ----
    chunks = splitter.split_documents(documents)

    # ---- 第 4 步：补充切片序号，便于界面展示"第 3/12 片"以及排序 ----
    total = len(chunks)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_total"] = total

    logger.info(
        "切分完成：%s，%d 个原始片段 -> %d 个知识分片（chunk_size=%d, overlap=%d）",
        Path(file_path).name, len(documents), total, chunk_size, chunk_overlap,
    )
    return chunks


# ----------------------------------------------------------------------
# 命令行入口：python src/doc_processor.py <文件路径>
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if len(sys.argv) < 2:
        print("用法：python src/doc_processor.py <文件路径>")
        raise SystemExit(1)

    result = load_and_split(sys.argv[1])
    print(f"\n共得到 {len(result)} 个分片\n")
    for i, chunk in enumerate(result[:3], start=1):
        print(f"--- 分片 {i} / 元数据 {chunk.metadata} ---")
        print(chunk.page_content)
        print()
