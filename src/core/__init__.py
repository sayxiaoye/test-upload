# src/core/__init__.py
"""
Core 模块：提供文件操作、API 请求和配置管理功能
"""

# 1. 定义公开 API（控制 from core import * 的行为）
__all__ = [
    # 文件操作
    "save_note",
    "read_note",
    "append_note",
    "file_exists",
    "divide",
    # API 请求
    "fetch_post",
    "get_posts_by_user",
    "create_post",
    # 配置管理
    "get_config",
]

# 2. 简化导入路径（可选）
from src.core.api_client import (
    create_post,
    fetch_post,
    get_posts_by_user,
)
from src.core.config import get_config
from src.core.file_ops import (
    append_note,
    divide,
    file_exists,
    read_note,
    save_note,
)
