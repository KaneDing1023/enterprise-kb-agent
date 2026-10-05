"""项目全局配置：所有环境变量统一在这里读取，业务代码只 import settings 即可。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录（本文件所在目录）
BASE_DIR = Path(__file__).resolve().parent

# 加载 .env（不覆盖已存在的系统环境变量）
load_dotenv(BASE_DIR / ".env", override=False)


def _env_str(key: str, default: str = "") -> str:
    value = os.getenv(key)
    return default if value is None or not value.strip() else value.strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env_str(key) or default)
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    value = _env_str(key).lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "y", "on", "是", "开"}


def _env_list(key: str, default: str = "") -> tuple[str, ...]:
    """读取逗号分隔的列表配置。"""
    raw = _env_str(key) or default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    """运行时配置，来源于 .env 文件。"""

    # ---------- 路径 ----------
    base_dir: Path = BASE_DIR
    raw_dir: Path = BASE_DIR / "data" / "raw"
    processed_dir: Path = BASE_DIR / "data" / "processed"
    persist_dir: Path = field(
        default_factory=lambda: (BASE_DIR / _env_str("CHROMA_PERSIST_DIR", "vector_store")).resolve()
    )

    # ---------- DashScope ----------
    dashscope_api_key: str = field(default_factory=lambda: _env_str("DASHSCOPE_API_KEY"))
    embedding_model: str = field(default_factory=lambda: _env_str("EMBEDDING_MODEL", "text-embedding-v3"))
    llm_model: str = field(default_factory=lambda: _env_str("LLM_MODEL", "qwen-plus"))

    # ---------- 向量库 ----------
    collection_name: str = field(default_factory=lambda: _env_str("COLLECTION_NAME", "enterprise_kb"))

    # ---------- Agent 知识库（Chroma + text-embedding-v4，持久化到 ./chroma_kb）----------
    kb_persist_dir: Path = field(
        default_factory=lambda: (BASE_DIR / _env_str("CHROMA_KB_DIR", "chroma_kb")).resolve()
    )
    kb_collection_name: str = field(
        default_factory=lambda: _env_str("KB_COLLECTION_NAME", "enterprise_kb")
    )
    kb_embedding_model: str = field(
        default_factory=lambda: _env_str("KB_EMBEDDING_MODEL", "text-embedding-v4")
    )

    # ---------- 切分 / 检索 ----------
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 120))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 4))

    # ---------- 重排序（Rerank）----------
    # 两阶段检索：向量粗排召回 RERANK_RECALL_K 条候选 -> gte-rerank 精排 -> 取 Top-K
    rerank_enabled: bool = field(default_factory=lambda: _env_bool("RERANK_ENABLED", True))
    rerank_model: str = field(default_factory=lambda: _env_str("RERANK_MODEL", "gte-rerank"))
    # 备用模型：配置的模型在账号下不可用（如未开通返回 403）时按顺序自动回退，
    # 逗号分隔。百炼当前对外服务的是 gte-rerank-v2。
    rerank_fallback_models: tuple[str, ...] = field(
        default_factory=lambda: _env_list("RERANK_FALLBACK_MODELS", "gte-rerank-v2")
    )
    # 粗排候选数；0 表示自动（TOP_K × RERANK_RECALL_MULTIPLIER，且不低于 10 条）
    rerank_recall_k: int = field(default_factory=lambda: _env_int("RERANK_RECALL_K", 0))
    rerank_recall_multiplier: int = field(
        default_factory=lambda: max(1, _env_int("RERANK_RECALL_MULTIPLIER", 3))
    )

    # ---------- 行为 ----------
    def ensure_dirs(self) -> None:
        """确保数据目录存在。"""
        for directory in (self.raw_dir, self.processed_dir, self.persist_dir, self.kb_persist_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def is_configured(self) -> bool:
        """是否已填写真实的 API Key。"""
        key = self.dashscope_api_key
        return bool(key) and not key.startswith("sk-xxx")

    def validate(self) -> None:
        """校验关键配置，未通过则抛出带修复提示的异常。"""
        if not self.is_configured():
            raise RuntimeError(
                "缺少有效的 DASHSCOPE_API_KEY。\n"
                f"请打开 {self.base_dir / '.env'}，把 DASHSCOPE_API_KEY 替换为真实 Key 后重试。\n"
                "申请地址：https://bailian.console.aliyun.com/"
            )


settings = Settings()
