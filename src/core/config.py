"""
配置管理模块
支持YAML配置文件 + .env 环境变量
优先级：环境变量 > YAML配置 > 默认值
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# 加载 .env(在模块加载时执行)
load_dotenv()


class Config:
    """
    配置类：加载 YAML 配置文件， 支持环境变量覆盖

    优先级：环境变量 > YAML配置 > 默认值

    使用实例：
        config = Config("config/config.yaml")
        print(config.get("app.name"))
        print(config.get("server.port"))

        # 支持嵌套路径访问
        chunk_size = config.get("rag.chunk_size", 512)

        # 访问所有配置
        print(config.app)
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化配置，加载 YAML 文件"""
        self._raw_config: dict[str, Any] = {}
        self._load_yaml(config_path)
        self._apply_env_overrides()

    def _load_yaml(self, config_path: str) -> None:
        """加载 YAML 配置文件"""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在：{config_path}")

        with open(path, encoding="utf-8") as f:
            self._raw_config = yaml.safe_load(f) or {}

    def _apply_env_overrides(self) -> None:
        """
        用环境变量覆盖 YAML 配置
        规则： ENV_APP_NAME 会覆盖 app.name
        """
        for key in os.environ:
            if key.startswith("ENV_") and len(key) > 4:
                # ENV_APP_NAME -> app.name
                path = key[4:].lower().replace("_", ".")
                value = os.environ[key]
                self._set_nested(path, self._convert_value(value))

    def _set_nested(self, path: str, value: Any) -> None:
        """按路径设置配置值"""
        parts = path.split(".")
        config = self._raw_config
        for part in parts[:-1]:
            config = config.setdefault(part, {})
        config[parts[-1]] = value

    def _convert_value(self, value: str) -> Any:
        """将字符串转换为适当类型"""
        # 布尔型
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        # 数字
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    def get(self, path: str, default: Any = None) -> Any:
        """
        通过点号路径获取配置值

        Args:
            path: 配置路径， 如 "app.name"
            default: 如果不存在则返回此值

        Returns:
            配置值
        """
        parts = path.split(".")
        current = self._raw_config
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def __getattr__(self, name: str) -> Any:
        """支持 config.app 直接访问"""
        return self.get(name)

    def to_dict(self) -> dict:
        """返回完整配置字典"""
        return self._raw_config


# 懒加载单列，避免重复加载
_config_instance: Config | None = None


def get_config(config_path: str = "config/config.yaml") -> Config:
    """获取全局配置单例"""
    global _config_instance  # noqa: PLW0603
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance
