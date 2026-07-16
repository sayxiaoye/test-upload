"""
my-first-project 主入口

用法：
    python -m src.main
"""

import logging
import os

from dotenv import load_dotenv

from src.core.file_ops import read_note, save_note

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# load .env file
load_dotenv()


def main():
    logging.info("程序启动")
    # 从环境变量读取配置
    name = os.getenv("MY_NAME", "Default User")
    logging.error(f"当前用户, {name}")

    # 使用示例
    try:
        save_note("data/note.txt", "Hello from main!")
        logging.info("文件写入成功")

        content = read_note("data/note.txt")
        logging.info(f"读取内容：{content}")
    except Exception as e:
        logging.error(f"操作失败：{e}")
        raise
    logging.info("程序正常结束")


if __name__ == "__main__":
    main()
