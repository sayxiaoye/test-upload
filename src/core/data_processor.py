"""
数据处理模块
支持 CSV、Excel 文件的读写和清洗
"""

from pathlib import Path

import pandas as pd


def read_csv(filepath: str) -> pd.DataFrame:
    """读取 CSV 文件"""
    return pd.read_csv(filepath, encoding="utf-8")


def read_excel(filepath: str, sheet_name: int = 0) -> pd.DataFrame:
    """读取 Excel 文件"""
    return pd.read_excel(filepath, sheet_name=sheet_name)


def save_csv(df: pd.DataFrame, filepath: str) -> None:
    """保存 CSV 文件"""
    df.to_csv(filepath, index=False, encoding="utf-8")


def save_excel(df: pd.DataFrame, filepath: str, sheet_name: str = "Sheet1") -> None:
    """保存 Excle 文件"""
    df.to_excel(filepath, index=False, sheet_name=sheet_name)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """数据清理： 去重、处理空值"""
    df = df.drop_duplicates()  # 去重
    df = df.dropna()  # 删除空值行
    return df


def filter_by_city(df: pd.DataFrame, city_name: str) -> pd.DataFrame:
    """筛选by城市"""
    return df[df["city"] == city_name]


def top_by_score(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """按分数排名， 取前n名"""
    return df.sort_values("score", ascending=False).head(n)


def group_by_city(df: pd.DataFrame) -> pd.DataFrame:
    """按城市分组统计平均分"""
    return df.groupby("city")["score"].mean().reset_index()


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """添加年龄分组列"""
    bins = [0, 25, 30, 100]
    labels = ["青年", "中年", "资深"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)
    return df


if __name__ == "__main__":
    # 确保 data 目录存在
    Path("data").mkdir(exist_ok=True)

    # 1. 读取CSV
    df = read_csv("data/sample_data.csv")
    print("📋 原始数据:")
    print(df)
    print()

    # 2. 数据清洗
    df_clean = clean_data(df)
    print("🧹 清洗后数据:")
    print(df_clean)
    print()

    # 3. 按城市筛选
    beijing = filter_by_city(df, "北京")
    print("🏙️ 北京的数据:")
    print(beijing)
    print()

    # 4. 按分数排名
    top3 = top_by_score(df, 3)
    print("🏆 前三名:")
    print(top3)
    print()

    # 5. 按城市分组统计
    city_stats = group_by_city(df)
    print("📊 各城市平均分:")
    print(city_stats)
    print()

    # 6. 添加年龄分组
    df_with_group = add_age_group(df)
    print("👥 添加年龄分组:")
    print(df_with_group)
    print()

    # 7. 保存结果
    save_csv(df_with_group, "data/processed_data.csv")
    print("✅ 已保存到 data/processed_data.csv")
