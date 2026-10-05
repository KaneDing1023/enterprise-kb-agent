"""重排序（Rerank）封装：阿里云百炼 ``gte-rerank`` 文本排序模型。

为什么要重排序
--------------
向量检索（``text-embedding-v4`` + Chroma）擅长「从海量切片里**快速**捞出候选」，
但它算的是「问题与片段的整体语义接近程度」，而不是「这个片段对回答该问题有多大用」。
两者在长文档、口语化提问、多个相似主题并存的场景里经常不一致 ——
真正能回答问题的片段可能排在第 3~8 位，而排第 1 的只是一段「话题相近但没答到点上」的文字。

``gte-rerank`` 是交叉编码器（cross-encoder）：把「问题 + 片段」拼在一起**逐对**打分，
精度明显高于向量内积，缺点是慢（必须逐条算，无法预计算索引）。所以标准做法是两阶段：

    粗排（召回）：向量检索召回 recall_k 条候选（多召回，宁滥勿缺）
    精排（排序）：gte-rerank 对这 recall_k 条重新打分排序，截取最终 Top-K

这样既保住向量检索的速度，又拿到交叉编码器的准确度，Top-1 命中率提升最明显。

设计要点
--------
* 直接调用 dashscope SDK 的 :class:`dashscope.TextReRank`，不引入新依赖。
* **模型自动回退**：百炼上模型名与账号权限是绑定的 —— 例如 ``gte-rerank``
  在某些账号/地域下会返回 ``403 Access denied``（未开通），而 ``gte-rerank-v2``
  正常可用。因此这里按「配置模型 → 备用模型（``RERANK_FALLBACK_MODELS``）」
  的顺序依次尝试，命中可用模型后就**记住**它，后续调用不再重试失败的那个。
  实际生效的模型名会随结果一起返回，界面上不会出现「说的是 A、用的是 B」。
* **失败降级**：rerank 调用失败（网络抖动 / 额度 / 限流 / 模型名写错）时**不抛异常**，
  打一条 warning 后原样返回向量粗排顺序，保证问答链路可用性优先。
* **空文本保护**：空白片段不参与打分，但索引严格对齐，避免把分数配错到别的片段上。
* **单例缓存**：同一个模型只建一个实例，重复问答不会反复初始化。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# 允许 `python src/reranker.py` 直接运行：脚本模式下 sys.path[0] 是 src/，
# 看不到项目根目录里的 config.py，这里先把它补进去。
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import dashscope  # noqa: E402
from dashscope import TextReRank  # noqa: E402

from config import settings  # noqa: E402

__all__ = [
    "DashScopeReranker",
    "get_reranker",
    "rerank_hits",
    "DEFAULT_RERANK_MODEL",
    "MODEL_UNAVAILABLE_CODES",
    "MAX_DOC_CHARS",
]

logger = logging.getLogger(__name__)

# 默认排序模型。gte-rerank 是百炼的通用文本排序模型，中文效果好。
# 注意：模型需要先在百炼控制台开通，未开通会返回 403（见 MODEL_UNAVAILABLE_CODES）。
DEFAULT_RERANK_MODEL = "gte-rerank"

# 这些状态码表示「这个模型名在当前账号下不可用」，可以直接换下一个备选模型：
#   400 Model not exist（模型名不存在）
#   403 Access denied（模型未开通 / 无权限）
#   404 路径或模型不存在
MODEL_UNAVAILABLE_CODES = frozenset({400, 403, 404})

# 单条候选文本送入排序模型前的截断长度（字符）。
# 本项目的切片通常 ≤ 800 字符，这里留足余量；截断只为防止异常超长文本触发请求报错。
MAX_DOC_CHARS = 4000


def _pick(obj: object, key: str, default: object = None) -> object:
    """兼容「字典」与「对象属性」两种返回形态（dashscope 的响应两种都可能出现）。"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class DashScopeReranker:
    """基于百炼 ``gte-rerank`` 的文本排序器。

    Args:
        model: 排序模型名，默认取 ``settings.rerank_model``（``gte-rerank``）。
        api_key: 百炼 API Key，默认取 ``settings.dashscope_api_key``。
        fallback_models: 备用模型名序列，默认取 ``settings.rerank_fallback_models``。
            配置的模型在账号下不可用时按顺序回退。

    Attributes:
        active_model: **实际生效**的模型名。首次调用若配置模型不可用，
            会自动切到备选模型并固定下来，后续调用直接用可用的那个。

    Example:
        >>> reranker = DashScopeReranker()
        >>> reranker.rerank("年假有多少天？", ["年假为五天。", "量子计算是前沿领域。"], top_n=1)
        [{'index': 0, 'relevance_score': 0.98}]
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        fallback_models: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.model = model or settings.rerank_model or DEFAULT_RERANK_MODEL
        self.api_key = api_key or settings.dashscope_api_key
        if fallback_models is None:
            fallback_models = settings.rerank_fallback_models
        self.fallback_models = list(fallback_models or [])
        self.active_model = self.model

        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY 为空，请先配置 .env 文件。")
        dashscope.api_key = self.api_key

    def _candidates(self) -> list[str]:
        """按「当前生效模型 → 备用模型」的顺序给出要尝试的模型名（去重）。"""
        names = [self.active_model, *self.fallback_models]
        ordered: list[str] = []
        for name in names:
            if name and name not in ordered:
                ordered.append(name)
        return ordered

    # ------------------------------------------------------------------
    # 底层：给一批候选文本打分排序
    # ------------------------------------------------------------------
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[dict]:
        """对候选文本按与查询的相关性重新排序。

        Args:
            query: 用户问题。
            documents: 候选文本列表（顺序即原始索引顺序）。
            top_n: 需要返回的条数；``None`` 表示全部返回。

        Returns:
            按相关性从高到低排序的列表，每项形如
            ``{"index": 原始下标, "relevance_score": 相关性分数}``。
            查询为空、候选为空或全部为空白时返回 ``[]``。

        Raises:
            RuntimeError: 所有候选模型都调用失败（附各模型的失败原因）。
        """
        query = (query or "").strip()

        # 空白文本不参与打分，但保留原始下标，保证结果能回填到正确的片段上
        pairs = [(i, (doc or "").strip()) for i, doc in enumerate(documents or [])]
        pairs = [(i, text) for i, text in pairs if text]
        if not query or not pairs:
            return []

        limit = len(pairs) if top_n is None else max(1, min(int(top_n), len(pairs)))
        texts = [text[:MAX_DOC_CHARS] for _, text in pairs]

        failures: list[str] = []
        for name in self._candidates():
            try:
                response = TextReRank.call(
                    model=name,
                    query=query,
                    documents=texts,
                    return_documents=False,  # 只要分数，原文我们本地就有，省 token
                    top_n=limit,
                )
            except Exception as exc:  # noqa: BLE001 —— 换下一个备选模型继续试
                failures.append(f"{name}: {exc}")
                continue

            status = getattr(response, "status_code", None)
            if status != 200:
                message = str(getattr(response, "message", ""))
                if status in MODEL_UNAVAILABLE_CODES:
                    # 模型名不存在 / 未开通 —— 换个备选模型再试
                    failures.append(f"{name} [{status}] {message}")
                    continue
                raise RuntimeError(f"gte-rerank（{name}）调用失败 [{status}] {message}")

            if name != self.active_model:
                logger.warning(
                    "排序模型 %s 在当前账号下不可用，已切换到 %s（%s）",
                    self.active_model,
                    name,
                    "; ".join(failures),
                )
                self.active_model = name

            return self._parse(response, pairs)

        raise RuntimeError("；".join(failures) or "没有可用的排序模型")

    def _parse(self, response: object, pairs: list[tuple[int, str]]) -> list[dict]:
        """把模型返回的分数解析成 ``[{"index", "relevance_score"}]``（下标已还原）。"""
        items = _pick(response.output, "results", []) or []
        scored: list[tuple[float, int]] = []
        for item in items:
            raw_index = _pick(item, "index")
            if raw_index is None:
                continue
            score = _pick(item, "relevance_score", 0.0)
            try:
                index = int(raw_index)
                score = float(score)
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(pairs):
                # 把「候选列表里的下标」还原成「调用方传入 documents 的下标」
                scored.append((score, pairs[index][0]))

        # 模型返回已按分数降序，这里再排一次并以下标做稳定兜底，避免接口行为变化影响顺序
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [{"index": index, "relevance_score": round(score, 6)} for score, index in scored]

    # ------------------------------------------------------------------
    # 上层：直接重排检索命中结果（kb.search_from_db 用）
    # ------------------------------------------------------------------
    def rerank_hits(
        self,
        query: str,
        hits: list[dict],
        top_n: int | None = None,
    ) -> list[dict]:
        """对 :func:`src.kb.search_from_db` 的命中列表做精排，返回新的 Top-N。

        每条命中会被补充三个字段（不修改传入的原字典）：

        ==================  ==================================================
        字段                 含义
        ==================  ==================================================
        ``recall_rank``     该片段在**向量粗排**阶段的排名（重排前的位置）
        ``recall_total``    进入精排的候选总数（含被淘汰的），便于展示提升幅度
        ``vector_score``    向量粗排的余弦相似度
        ``rerank_score``    gte-rerank 的相关性分数（重排后的排序依据）
        ``rerank_model``    本次**实际生效**的排序模型名（可能与配置值不同，
                            配置模型不可用时会自动回退到备用模型）
        ==================  ==================================================

        同时 ``score`` 会被改写为 ``rerank_score``、``rank`` 重新编号为 1..N，
        这样下游（阈值判断、界面展示、逐段溯源）看到的都是「最终排序」。

        **任何异常都不会抛出**：失败时原样返回 ``hits[:top_n]``（仅重编号 rank），
        ``rerank_score`` 保持 ``None``，界面上依旧能正常回答，只是退化为粗排结果。
        """
        if not hits:
            return []

        limit = len(hits) if top_n is None else max(1, min(int(top_n), len(hits)))
        try:
            ranked = self.rerank(query, [hit.get("content") or "" for hit in hits], top_n=limit)
        except Exception as exc:  # noqa: BLE001 —— 排序失败不该拖垮问答
            logger.warning("重排序失败，已降级为向量粗排顺序：%s", exc)
            ranked = []

        if not ranked:
            return [_renumber(hit, rank) for rank, hit in enumerate(hits[:limit], 1)]

        total = len(hits)  # 进入精排的候选总数，用于界面展示「粗排 N 条 → 保留 M 条」
        results: list[dict] = []
        for rank, item in enumerate(ranked, 1):
            hit = dict(hits[item["index"]])
            vector_score = hit.get("vector_score")
            if vector_score is None:
                vector_score = hit.get("score")
            hit["recall_rank"] = hit.get("rank")
            hit["recall_total"] = total
            hit["vector_score"] = vector_score
            hit["rerank_score"] = item["relevance_score"]
            hit["rerank_model"] = self.active_model
            hit["score"] = item["relevance_score"]
            hit["rank"] = rank
            results.append(hit)
        return results


def _renumber(hit: dict, rank: int) -> dict:
    """降级路径：只重编号，不覆盖 score，并明确标记「未经重排」。"""
    item = dict(hit)
    item["rank"] = rank
    if item.get("vector_score") is None:
        item["vector_score"] = hit.get("score")
    item["rerank_score"] = None
    item["rerank_model"] = None
    item["recall_rank"] = hit.get("rank")
    item["recall_total"] = None
    return item


# ----------------------------------------------------------------------
# 单例：同一个模型只建一个实例
# ----------------------------------------------------------------------
_reranker_cache: dict[str, DashScopeReranker] = {}


def get_reranker(model: str | None = None) -> DashScopeReranker:
    """获取（并复用）排序器实例。"""
    name = model or settings.rerank_model or DEFAULT_RERANK_MODEL
    if name not in _reranker_cache:
        _reranker_cache[name] = DashScopeReranker(model=name)
        logger.info("已加载重排序模型：%s", name)
    return _reranker_cache[name]


def rerank_hits(query: str, hits: list[dict], top_n: int | None = None) -> list[dict]:
    """便捷函数：用默认排序模型对命中列表做精排（失败自动降级）。"""
    return get_reranker().rerank_hits(query, hits, top_n=top_n)
