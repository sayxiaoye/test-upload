import json

from src.core import fetch_post


def save_post_to_file(post_id: int, filename: str = "data/post.json") -> None:
    """获取文章并保存为json格式"""
    post = fetch_post(post_id)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(post, f, ensure_ascii=False, indent=2)  # 将数据持久化到文件
    print(f"✅ 已保存到 {filename}")


def load_post_from_file(filename: str = "data/post.json") -> dict:
    """从json文件读取文章"""
    with open(filename, encoding="utf-8") as f:
        return json.load(f)  # type: ignore


if __name__ == "__main__":
    # 保存
    save_post_to_file(1)

    # 读取
    post = load_post_from_file()
    print(f"标题：{post.get('title')}")
