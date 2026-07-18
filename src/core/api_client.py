from typing import cast

import requests

# 认证请求头
# headers = {
#     "Authorization": "Bearer your-api-key",
#     "Content-Type": "application/json",
#     "User-Agent": "my-project/1.0"
# }
# response = requests.get(url, headers=headers)

# 创建 Session，复用连接和 Cookie
# session = requests.Session()
# session.headers.update({
#     "Authorization": "Bearer your-api-key",
#     "Content-Type": "application/json"
# })

# 使用同一个 session 发送多个请求
# response1 = session.get("https://api.example.com/users")
# response2 = session.post("https://api.example.com/posts", json=data)


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


def get_posts_by_user(user_id: int, limit: int = 10) -> list[dict]:
    """获取指定用户的所有文章"""
    url = "https://jsonplaceholder.typicode.com/posts"
    params = {"userId": user_id, "_limit": limit}
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return cast(list[dict], response.json())


def create_post(title: str, body: str, user_id: int = 1) -> dict:
    """创建新文章"""
    url = "https://jsonplaceholder.typicode.com/posts"
    payload = {"title": title, "body": body, "user_id": user_id}
    response = requests.post(url, json=payload, timeout=10)
    response.raise_for_status()
    return cast(dict, response.json())


# PUT：更新资源
def update_post(post_id: int, title: str, body: str) -> dict:
    url = f"https://jsonplaceholder.typicode.com/posts/{post_id}"
    payload = {"title": title, "body": body}
    response = requests.put(url, json=payload)
    response.raise_for_status()
    return cast(dict, response.json())


# DELETE：删除资源
def delete_post(post_id: int) -> bool:
    url = f"https://jsonplaceholder.typicode.com/posts/{post_id}"
    response = requests.delete(url)
    return response.status_code == 200


if __name__ == "__main__":
    # 简单测试一下
    post = fetch_post(1)
    print(f"获取到的文章标题: {post.get('title')}")

    print("\n" + "=" * 50)

    # 测试 1：获取用户的所有文章
    print("\n📋 获取用户 1 的文章:")
    posts = get_posts_by_user(1, limit=3)
    for p in posts:
        print(f"  - {p.get('title')}")

    print("\n" + "=" * 50)

    # 测试 2：创建新文章
    print("\n📝 创建新文章:")
    new_post = create_post(
        title="Python 学习笔记",
        body="今天学习了 requests 库的 GET 和 POST 方法...",
        user_id=1,
    )
    print(f"  新文章 ID: {new_post.get('id')}")
    print(f"  标题: {new_post.get('title')}")

    # 测试更新
    updated = update_post(1, "新标题", "新内容")
    print(f"更新后标题: {updated.get('title')}")

    # 测试删除
    deleted = delete_post(1)
    print(f"删除成功: {deleted}")
