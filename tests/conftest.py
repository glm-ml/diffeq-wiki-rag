import os

import pytest

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("ENABLE_WEB_SEARCH", "false")

from app.settings import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
