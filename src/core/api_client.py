import requests


def fetch_post(post_id: int):
    """
    从 JSONPlaceholder API 获取一篇指定 ID 的文章。

    Args:
        post_id: 文章的 ID

    Returns:
        一个包含文章信息的字典
    """
    url = f"https://jsonplaceholder.typicode.com/posts/{post_id}"

    try:
        response = requests.get(url)
        # 检查请求是否成功 (状态码 200)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API 请求失败: {e}")
        return {}


if __name__ == "__main__":
    # 简单测试一下
    post = fetch_post(1)
    print(f"获取到的文章标题: {post.get('title')}")
