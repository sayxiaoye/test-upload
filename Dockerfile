FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./

RUN pip install --no-cache-dir -e .

COPY src/ ./src/
COPY config/ ./config/
COPY tests/ ./tests/

CMD ["python", "-m", "src.cli", "--help"]
