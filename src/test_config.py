from src.core import get_config


def main():
    # 加载配置
    config = get_config("config/config.yaml")

    print("=" * 50)
    print("配置加载测试")
    print("=" * 50)

    # 读取配置
    print(f"应用名字：{config.get('app.name')}")
    print(f"版本：{config.get('app.version')}")
    print(f"debug模式: {config.get('app.debug')}")
    print(f"服务器端口：{config.get('server.port')}")
    print(f"日志级别：{config.get('logging.level')}")
    print(f"模型：{config.get('models.default')}")
    print(f"温度：{config.get('models.temperature')}")
    print(f"Chuck size: {config.get('rag.chunk_size')}")
    print(f"Top-K:{config.get('rag.top_k')}")
    print("=" * 50)

    # 测试默认值
    print(f"不存在的配置（返回默认值）: {config.get('nonexistent.key', 'default')}")

    # 查看所有配置
    print("\n所有配置:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
