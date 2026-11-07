"""Настройки сервиса. Читаются из переменных окружения и `.env`."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # DeepSeek (OpenAI-совместимый API)
    deepseek_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1200

    # Эмбеддинги: мультиязычная e5, чтобы русские вопросы находили англоязычные статьи
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = 384
    normalize_embeddings: bool = True

    # Qdrant: пустой URL -> встроенный режим (папка qdrant_path)
    qdrant_url: str = ""
    qdrant_path: str = "qdrant_data"
    collection_name: str = "diffeq_wiki"
    top_k: int = 5

    # Агент
    max_iterations: int = 6
    max_output_chars: int = 2500
    enable_web_search: bool = True

    # Публичный адрес для Open Graph / Telegram-превью
    public_base_url: str = "https://195.62.49.192.nip.io"

    # Данные
    corpus_path: str = "data/corpus.jsonl"


@lru_cache
def get_settings() -> Settings:
    return Settings()
