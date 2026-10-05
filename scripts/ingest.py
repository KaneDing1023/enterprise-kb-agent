"""知识入库脚本（命令行）。

用法：
    python scripts/ingest.py                      # 入库 data/raw 下所有文档
    python scripts/ingest.py data/raw/员工手册.pdf  # 入库指定文件或目录
    python scripts/ingest.py --reset              # 先清空向量库再入库
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 让脚本能 import 到项目根目录下的 config / src
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402
from src.document_loader import load_directory, load_document  # noqa: E402
from src.text_splitter import split_documents  # noqa: E402
from src.vector_store import add_documents, collection_info, reset_collection  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="企业知识库文档入库")
    parser.add_argument("path", nargs="?", default=None, help="文件或目录路径，默认 data/raw")
    parser.add_argument("--reset", action="store_true", help="入库前清空向量库")
    parser.add_argument("--chunk-size", type=int, default=None, help="切片长度，默认取 .env")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="切片重叠，默认取 .env")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings.validate()
    settings.ensure_dirs()

    target = Path(args.path).resolve() if args.path else settings.raw_dir
    if not target.exists():
        logger.error("路径不存在：%s", target)
        return 1

    if args.reset:
        logger.warning("--reset 已开启，正在清空向量库……")
        reset_collection()

    logger.info("开始加载：%s", target)
    documents = load_document(target) if target.is_file() else load_directory(target)
    if not documents:
        logger.warning("没有加载到任何内容，请确认目录下存在 pdf / docx / txt / md 文件。")
        return 1

    chunks = split_documents(documents, args.chunk_size, args.chunk_overlap)
    logger.info("切分完成：%d 个原始片段 -> %d 个知识分片", len(documents), len(chunks))

    written = add_documents(chunks)
    logger.info("入库完成，共写入 %d 个分片。", written)

    info = collection_info()
    logger.info(
        "当前知识库：collection=%s，分片总数=%d，持久化目录=%s",
        info["collection"], info["chunks"], info["persist_dir"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
