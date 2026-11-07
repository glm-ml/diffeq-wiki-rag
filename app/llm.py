"""Клиент DeepSeek через OpenAI-совместимый интерфейс."""

from langchain_openai import ChatOpenAI

from app.settings import get_settings


def get_llm() -> ChatOpenAI:
    settings = get_settings()
    if not settings.deepseek_api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY не задан. Скопируйте .env.example в .env "
            "и впишите ключ из https://platform.deepseek.com/api_keys"
        )
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.deepseek_api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
