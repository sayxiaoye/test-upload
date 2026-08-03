"""测试配置管理模块（E5 补测试）。"""

import os

import pytest
import yaml

from src.core.config import Config, get_config


class TestConfig:
    """Config 类的单元测试。"""

    @pytest.fixture(autouse=True)
    def _clean_env_overrides(self, monkeypatch):
        """清除 .env 中的 ENV_ 覆盖，避免污染测试。"""
        for key in list(os.environ):
            if key.startswith("ENV_"):
                monkeypatch.delenv(key, raising=False)

    @pytest.fixture
    def temp_config_file(self, tmp_path):
        """创建临时配置文件。"""
        config_path = tmp_path / "test_config.yaml"
        data = {
            "app": {"name": "test-app", "version": "1.0"},
            "server": {"port": 8080, "debug": True},
            "nested": {"key": "value"},
        }
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f)
        return str(config_path)

    def test_load_valid_config(self, temp_config_file):
        """加载有效配置文件。"""
        cfg = Config(temp_config_file)
        assert cfg.get("app.name") == "test-app"
        assert cfg.get("app.version") == "1.0"

    def test_load_nonexistent_file(self):
        """加载不存在的文件抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError, match="配置文件不存在"):
            Config("nonexistent_path.yaml")

    def test_get_with_default(self, temp_config_file):
        """get() 在 key 不存在时返回默认值。"""
        cfg = Config(temp_config_file)
        assert cfg.get("nonexistent.key") is None
        assert cfg.get("nonexistent.key", "fallback") == "fallback"

    def test_get_nested_path(self, temp_config_file):
        """支持点号分隔的嵌套路径。"""
        cfg = Config(temp_config_file)
        assert cfg.get("nested.key") == "value"
        assert cfg.get("server.port") == 8080

    def test_getattr_access(self, temp_config_file):
        """支持 config.app 属性式访问。"""
        cfg = Config(temp_config_file)
        app = cfg.app  # __getattr__ → get("app")
        assert isinstance(app, dict)
        assert app["name"] == "test-app"

    def test_to_dict(self, temp_config_file):
        """to_dict 返回完整配置字典。"""
        cfg = Config(temp_config_file)
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["app"]["name"] == "test-app"

    def test_env_override_bool_conversion(self, tmp_path, monkeypatch):
        """环境变量覆盖：布尔值转换。"""
        config_path = tmp_path / "cfg.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump({"app": {"debug": False}}, f)

        # 设置 ENV_ 环境变量覆盖
        monkeypatch.setenv("ENV_APP_DEBUG", "true")
        cfg = Config(str(config_path))
        assert cfg.get("app.debug") is True  # "true" → True

    def test_env_override_int_conversion(self, tmp_path, monkeypatch):
        """环境变量覆盖：整数转换。"""
        config_path = tmp_path / "cfg.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump({"server": {"port": 8000}}, f)

        monkeypatch.setenv("ENV_SERVER_PORT", "9090")
        cfg = Config(str(config_path))
        assert cfg.get("server.port") == 9090  # "9090" → 9090

    def test_env_override_float_conversion(self, tmp_path, monkeypatch):
        """环境变量覆盖：浮点数转换。"""
        config_path = tmp_path / "cfg.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump({"models": {"temperature": 0.5}}, f)

        monkeypatch.setenv("ENV_MODELS_TEMPERATURE", "0.9")
        cfg = Config(str(config_path))
        assert cfg.get("models.temperature") == 0.9  # "0.9" → 0.9


class TestGetConfigSingleton:
    """全局配置单例测试。"""

    def test_get_config_returns_config(self):
        """get_config 返回 Config 实例。"""
        cfg = get_config()
        assert isinstance(cfg, Config)

    def test_get_config_is_singleton(self):
        """多次调用返回同一实例。"""
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2
