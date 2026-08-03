FROM python:3.12-slim

WORKDIR /app

# 先复制依赖配置文件
COPY pyproject.toml ./

# 安装项目依赖 + Web 服务所需的额外依赖
RUN pip install --no-cache-dir ".[dev]"

# 复制项目源码和配置
COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/

# 创建日志和下载目录
RUN mkdir -p logs downloads

# 暴露 FastAPI 默认端口
EXPOSE 8000

# 启动 FastAPI 服务（host 必须为 0.0.0.0 才能在容器外访问）
CMD ["python", "-m", "src.app.main", "serve", "--host", "0.0.0.0", "--port", "8000"]
