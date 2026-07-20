"""
文本处理模块
使用正则表达式进行文本提取、清洗和匹配
"""

import re


def extract_emails(text: str) -> list[str]:
    """提取所有邮箱地址"""
    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    return re.findall(pattern, text)


def extract_phones(text: str) -> list[str]:
    """提取所有手机号（中国内地 11 位）"""
    pattern = r"1[3-9]\d{9}"
    return re.findall(pattern, text)


def extract_urls(text: str) -> list[str]:
    pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    return re.findall(pattern, text)


def extract_dates(text: str) -> list[str]:
    """提取日期格式 YYYY-MM-DD"""
    pattern = r"\d{4}-\d{2}-\d{2}"
    return re.findall(pattern, text)


def clean_extra_spaces(text: str) -> str:
    """清理多余空格和换行"""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_hashtags(text: str) -> list[str]:
    """提取 #话题标签"""
    pattern = r"#\w+"
    return re.findall(pattern, text)


def is_valid_email(email: str) -> bool:
    """验证有效邮箱"""
    pattern = r"^[a-zA-Z0-8._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def is_valid_phone(phone: str) -> bool:
    """验证有效手机"""
    pattern = r"^1[3-9]\d{9}$"
    return bool(re.match(pattern, phone))


if __name__ == "__main__":
    # 测试文本
    text = """
    联系方式：
    邮箱：alice@example.com, bob@test.cn, charlie@company.com
    电话：13812345678, 15987654321, 010-12345678
    网站：https://example.com, www.python.org, https://docs.python.org/3/
    日期：2024-01-15, 2024-12-31
    话题：#Python #AI #数据科学
    这里是    一段  有  多余  空格的  文本。
    """

    print("=" * 60)
    print("📧 提取邮箱:", extract_emails(text))
    print("📱 提取电话:", extract_phones(text))
    print("🔗 提取 URL:", extract_urls(text))
    print("📅 提取日期:", extract_dates(text))
    print("🏷️ 提取话题:", extract_hashtags(text))
    print("=" * 60)

    # 文本清洗
    clean = clean_extra_spaces(text)
    print("🧹 清洗后文本:")
    print(clean[:200] + "...")
    print("=" * 60)

    # 验证
    print("✅ 验证邮箱 alice@example.com:", is_valid_email("alice@example.com"))
    print("❌ 验证邮箱 alice@example:", is_valid_email("alice@example"))
    print("✅ 验证手机 13812345678:", is_valid_phone("13812345678"))
    print("❌ 验证手机 12345678901:", is_valid_phone("12345678901"))
