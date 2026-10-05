"""答案溯源：把「带 [n] 编号的回答」映射到具体来源文件与对应原文片段。

问题背景
--------
大模型生成的答案里通常只有 ``[1]`` ``[2]`` 这样的编号，用户看到编号还得自己
在「引用来源」面板里对号入座，溯源体验很割裂；而且编号是模型**自由生成**的，
一旦模型写错编号，很容易把 A 文件的结论挂到 B 文件头上。

本模块的做法是「模型给编号、系统做映射」：

1. 提示词要求模型**每一段**末尾标注其依据的片段编号（如 ``[1]``、``[1][3]``）；
2. 本模块按空行把答案切段，逐段解析出编号；
3. 用编号去 :func:`src.kb.search_from_db` 返回的来源列表里取**真实**的文件名、
   页码和原文片段，在段落末尾追加一行 ``【来源：员工手册.pdf（第 2 页）】``。

编号 → 文件的映射完全由系统完成，模型无法伪造文件名与页码，溯源结果可信；
同时对话里保留了可点击/可核对的原文片段（``snippet``），便于逐条验证。

对外函数
--------
- :func:`split_paragraphs` —— 答案分段
- :func:`build_segments` —— 逐段解析出它引用了哪些来源（结构化，供界面与接口使用）
- :func:`annotate_answer` —— 生成「每段末尾标注来源」的完整答案文本
- :func:`annotate_partial` —— 流式场景：只标注**已经写完**的段落
"""

from __future__ import annotations

import re

__all__ = [
    "CITATION_TAG_TEMPLATE",
    "split_paragraphs",
    "extract_markers",
    "strip_markers",
    "format_citation_tag",
    "build_segments",
    "annotate_answer",
    "annotate_partial",
]

# 段落末尾的来源标注格式，例如【来源：员工手册.pdf（第 2 页）】
CITATION_TAG_TEMPLATE = "【来源：{labels}】"

# 模型标注的片段编号：兼容 [1] / 【1】 / ［1］ 三种写法（全角半角都可能出现）
_MARKER_RE = re.compile(r"[\[【［]\s*(\d{1,3})\s*[\]】］]")

# 单条标注里最多列出的文件数，避免一段引用十几个来源时标签过长
MAX_LABELS = 3


def split_paragraphs(text: str) -> list[str]:
    """把答案切成段落。

    默认按**空行**切分（Markdown 的自然段）；如果整段答案没有任何空行，
    再退化为按单换行切分（兼容模型把所有内容挤在一起的输出）。
    """
    text = (text or "").strip()
    if not text:
        return []

    blocks = [block.strip() for block in re.split(r"\n\s*\n+", text)]
    blocks = [block for block in blocks if block]
    if len(blocks) <= 1:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if len(lines) > 1:
            return lines
    return blocks


def extract_markers(text: str) -> list[int]:
    """提取文本里的片段编号，去重并保持出现顺序。"""
    seen: set[int] = set()
    indexes: list[int] = []
    for raw in _MARKER_RE.findall(text or ""):
        number = int(raw)
        if number not in seen:
            seen.add(number)
            indexes.append(number)
    return indexes


def strip_markers(text: str) -> str:
    """去掉编号标记（编号已转成「来源」标注，留在正文里反而干扰阅读）。"""
    cleaned = _MARKER_RE.sub("", text or "")
    # 收掉因删标记留下的多余空格：
    #   行尾空格，以及中文标点前的空格（模型常写成「十五天 [2]。」）
    cleaned = re.sub(r"[ \t]+(?=\n|$)", "", cleaned)
    cleaned = re.sub(r"[ \t]+(?=[。，、；：！？）」』】》])", "", cleaned)
    return cleaned.strip()


def _as_citation(source: dict) -> dict:
    """从来源条目里抽出溯源需要的字段（统一成界面/接口可直接用的结构）。"""
    return {
        "index": source.get("index"),
        "citation": source.get("citation") or source.get("file_name") or "未知来源",
        "file_name": source.get("file_name"),
        "page": source.get("page"),
        "source": source.get("source"),
        "snippet": source.get("snippet") or source.get("preview") or "",
        "rerank_score": source.get("rerank_score"),
        "score": source.get("score"),
    }


def format_citation_tag(citations: list[dict]) -> str:
    """把一组来源渲染成段落末尾的标注文本；同一文件只出现一次。"""
    labels: list[str] = []
    for item in citations or []:
        label = str(item.get("citation") or "").strip()
        if label and label not in labels:
            labels.append(label)
    if not labels:
        return ""
    if len(labels) > MAX_LABELS:
        labels = labels[:MAX_LABELS] + [f"等 {len(labels)} 处"]
    return CITATION_TAG_TEMPLATE.format(labels="；".join(labels))


def build_segments(answer: str, sources: list[dict] | None = None) -> list[dict]:
    """逐段解析答案，得到「段落 → 引用来源」的结构化结果。

    Args:
        answer: 模型生成的答案（含 ``[n]`` 编号）。
        sources: 来源列表，结构同 :func:`src.kb.search_from_db` 的返回值经
            ``_to_source()`` 压缩后的条目（含 ``index`` / ``citation`` / ``snippet``）。

    Returns:
        分段列表，每项包含：

        =============  ==========================================================
        字段            含义
        =============  ==========================================================
        ``paragraph``  段号，从 1 开始
        ``text``       去掉编号标记后的段落正文
        ``raw``        原始段落文本（保留编号，便于排查）
        ``indexes``    该段引用的片段编号（已过滤掉不存在的编号）
        ``citations``  对应的来源条目（含文件名 / 页码 / 对应片段 snippet）
        ``tag``        段落末尾要追加的标注文本，如 ``【来源：员工手册.pdf（第 2 页）】``
        =============  ==========================================================
    """
    by_index = {
        int(src["index"]): src
        for src in (sources or [])
        if src.get("index") is not None
    }

    segments: list[dict] = []
    for number, raw in enumerate(split_paragraphs(answer), 1):
        found = extract_markers(raw)
        valid = [n for n in found if n in by_index]
        citations = [_as_citation(by_index[n]) for n in valid]
        segments.append(
            {
                "paragraph": number,
                "text": strip_markers(raw),
                "raw": raw,
                "indexes": valid,
                "citations": citations,
                "tag": format_citation_tag(citations),
            }
        )
    return segments


def annotate_answer(answer: str, sources: list[dict] | None = None) -> str:
    """生成「每段答案末尾标注引用来源」的完整文本。

    段落末尾会追加一行 :data:`CITATION_TAG_TEMPLATE` 格式的标注；
    没有标注到编号的段落保持原样（宁可不标，也不猜）。

    Example:
        >>> annotate_answer("年假为五天。[1]\\n\\n申请需主管审批。[2]", sources)
        '年假为五天。\\n\\n【来源：员工手册.pdf（第 2 页）】\\n\\n申请需主管审批。\\n\\n【来源：员工手册.pdf（第 3 页）】'
    """
    original = answer or ""
    if not original.strip():
        return original

    segments = build_segments(original, sources)
    if not segments:
        return original

    blocks: list[str] = []
    for segment in segments:
        body = segment["text"] or segment["raw"]
        tag = segment["tag"]
        # 用空行把标注独立成一行：Markdown / CLI 里都清晰可辨
        blocks.append(f"{body}\n\n{tag}" if tag else body)
    return "\n\n".join(blocks)


def annotate_partial(text: str, sources: list[dict] | None = None) -> str:
    """流式场景：只为**已经写完**的段落补来源标注，最后一段原样保留。

    模型是按 token 逐段吐字的，最后一段随时可能被追加内容。这里只在文本里
    出现了空行（即上一段已结束）时才标注前面的段落，最后一段等 ``end`` 事件到达后
    由 :func:`annotate_answer` 统一标注，避免出现「标了一半的标注」。
    """
    raw = text or ""
    if not raw.strip():
        return raw

    finished, separator, tail = raw.rpartition("\n\n")
    if not separator:
        return raw  # 只有一段且尚未结束，先不加标注
    return annotate_answer(finished, sources) + separator + tail
