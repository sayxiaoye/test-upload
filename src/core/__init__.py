"""core 包：配置管理 + PDF 文档解析。"""

from src.core.config import get_config  # 全局配置单例
from src.core.pdf_processor import (  # PDF 文本提取
    extract_full_text,
    extract_text_from_pdf,
)

__all__ = [
    "get_config",
    "extract_text_from_pdf",
    "extract_full_text",
]
