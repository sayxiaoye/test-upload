from unittest.mock import Mock, patch

from src.core.api_client import fetch_post


def test_fetch_post_success():
    # 1. 创建模拟响应
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"titles": "Mocked Title!!"}

    # 2. 用 patch 替换 requests.get
    with patch("src.api_client.requests.get") as mock_get:
        mock_get.return_value = mock_response

        # 3. 调用被测函数
        result = fetch_post(1)

        # 4. 验证结果
        assert result.get("titles") == "Mocked Title!!"

        # 5. 验证被调用的参数
        mock_get.assert_called_once_with("https://jsonplaceholder.typicode.com/posts/1")


def test_fetch_post_failure():
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.json.return_value = {}

    # 2. 用 patch 替换 requests.get
    with patch("src.api_client.requests.get") as mock_get:
        mock_get.return_value = mock_response
        # 3. 调用被测函数
        result = fetch_post(999)
        # 4. 验证结果
        assert result == {}
