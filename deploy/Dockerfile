FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

ARG UV_EXTRAS=""

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md /app/
COPY app /app/app

RUN if [ -n "$UV_EXTRAS" ]; then uv sync --no-dev --extra "$UV_EXTRAS"; else uv sync --no-dev; fi

RUN mkdir -p /app/data/generated /app/data/metadata /app/data/lora /app/data/datasets

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
