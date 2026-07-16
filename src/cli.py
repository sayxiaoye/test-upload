"""
CLI 入口： my-first-project命令工具

用法：
    python -m src.cli --help
    python -m src.cli --read data/note.txt
    python -m src.cli --save data/note.txt "Hello, CLI!"
    python -m src.cli --fetch 1
"""

import typer

from src.core.api_client import fetch_post
from src.core.config import get_config
from src.core.file_ops import file_exists, read_note, save_note

# 创建 CLI 应用
app = typer.Typer(
    name="my-project",
    help="my firs CLI",
    add_completion=False,
)


@app.command()
def read(filepath: str):
    """
    读取文件内容

    Args:
        filepath:文件路径
    """
    content = read_note(filepath)
    typer.echo(f"文件内容:\n{content}")


@app.command()
def save(filepath: str, content: str):
    """
    保存内容到文件

    Args:
        filepath:文件路径
        content: 写入的内容
    """
    save_note(filepath, content)
    typer.echo(f"✅ 已保存到: {filepath}")


@app.command()
def exists(filepath: str):
    """
    检查文件是否存在

    Args:
        filepath: 文件路径
    """
    if file_exists(filepath):
        typer.echo(f"✅ 文件存在: {filepath}")
    else:
        typer.echo(f"❌ 文件不存在: {filepath}")


@app.command()
def fetch(post_id: int):
    """
    从 API 获取一篇文章

    Args:
        post_id: 文章 ID
    """
    result = fetch_post(post_id)
    if result:
        typer.echo(f"标题: {result.get('title')}")
        typer.echo(f"内容: {result.get('body', '')[:100]}...")
    else:
        typer.echo("❌ 获取失败")


@app.command()
def config(key: str | None = None):
    """
    查看配置

    Args:
        key: 配置键（可选），如 "app.name"
    """
    cfg = get_config()
    if key:
        value = cfg.get(key)
        if value is not None:
            typer.echo(f"{key} = {value}")
        else:
            typer.echo(f"❌ 配置键不存在: {key}")
    else:
        typer.echo("📋 当前配置:")
        for k, v in cfg.to_dict().items():
            typer.echo(f"  {k}: {v}")


if __name__ == "__main__":
    app()
