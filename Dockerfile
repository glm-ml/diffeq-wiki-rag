FROM ghcr.io/astral-sh/uv:0.13.7 AS uv

FROM python:3.12-slim AS builder

COPY --from=uv /uv /uvx /bin/
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app
COPY pyproject.toml uv.lock ./
# Ставим CPU-сборку torch: CUDA-зависимости не нужны и весят несколько ГБ.
RUN uv export --frozen --no-dev --no-hashes --output-file requirements.txt \
    && grep -v -E '^(nvidia-|cuda-|triton)' requirements.txt > requirements-cpu.txt \
    && uv venv /app/.venv \
    && uv pip install --python /app/.venv/bin/python \
        --torch-backend=cpu -r requirements-cpu.txt


FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/models

WORKDIR /app

RUN groupadd --system app \
    && useradd --system --gid app --home /app app \
    && mkdir -p /models \
    && chown -R app:app /models

COPY --chown=app:app --from=builder /app/.venv /app/.venv
COPY --chown=app:app app ./app
COPY --chown=app:app data ./data
COPY --chown=app:app docker/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
