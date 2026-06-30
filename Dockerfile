FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# 构建依赖（部分包需要编译）
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# 先装 CPU-only torch，避免 sentence-transformers/bertopic 拉超大 CUDA 轮子
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
# 只装运行依赖（利用层缓存）
RUN pip install --no-cache-dir \
        "bertopic>=0.16,<0.17" "scikit-learn>=1.3" "jieba>=0.42" \
        "pandas>=2.0" "pyarrow>=14.0" "numpy>=1.24,<2.0" "openai>=1.30" \
        "SQLAlchemy>=2.0" "PyMySQL>=1.1" "redis>=5.0" "fastapi>=0.110" \
        "uvicorn[standard]>=0.29" "pydantic>=2.6" "pydantic-settings>=2.2" \
        "Faker>=25.0" "boto3>=1.34" "tenacity>=8.2"

COPY . .

EXPOSE 8000

# 默认启动 FastAPI；批处理 / 造数据通过 docker compose run 覆盖 command
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
