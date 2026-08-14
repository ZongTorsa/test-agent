"""统一项目日志格式，避免在业务模块中散落 print 调试输出。"""

from __future__ import annotations

import logging
import os


def get_logger(name: str) -> logging.Logger:
    """返回已配置的模块日志器；日志级别由 LOG_LEVEL 控制。"""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    # 保留项目日志，隐藏 HTTP 请求等第三方库的常规噪声。
    for library_name in ("httpx", "httpx2", "openai", "chromadb"):
        logging.getLogger(library_name).setLevel(logging.WARNING)
    return logging.getLogger(name)
