import os
from unittest import mock
import pytest
from pydantic import ValidationError
from config.settings import Settings

def test_settings_default_values():
    """Test that settings has the expected default values when env is empty."""
    with mock.patch.dict(os.environ, {}, clear=True):
        settings = Settings()
        assert settings.ENV == "development"
        assert settings.LOG_LEVEL == "INFO"
        assert settings.GCP_PROJECT_ID == "agenticuse"
        assert settings.GCP_LOCATION == "us-central1"
        assert settings.FREE_LLM_API_BASE_URL == "http://localhost:8000/v1"
        assert settings.CRITICAL_MODEL == "gemini-1.5-flash"
        assert settings.QDRANT_PORT == 6333

def test_settings_env_override():
    """Test that settings can be overridden via environment variables."""
    custom_env = {
        "ENV": "production",
        "LOG_LEVEL": "DEBUG",
        "GCP_PROJECT_ID": "custom-project",
        "GCP_LOCATION": "europe-west1",
        "FREE_LLM_API_BASE_URL": "https://api.example.com",
        "FREE_LLM_API_KEY": "test-key",
        "CRITICAL_MODEL": "custom-model",
        "QDRANT_PORT": "1234",
    }
    with mock.patch.dict(os.environ, custom_env, clear=True):
        settings = Settings()
        assert settings.ENV == "production"
        assert settings.LOG_LEVEL == "DEBUG"
        assert settings.GCP_PROJECT_ID == "custom-project"
        assert settings.GCP_LOCATION == "europe-west1"
        assert settings.FREE_LLM_API_BASE_URL == "https://api.example.com"
        assert settings.FREE_LLM_API_KEY == "test-key"
        assert settings.CRITICAL_MODEL == "custom-model"
        assert settings.QDRANT_PORT == 1234

def test_settings_invalid_type():
    """Test that setting an invalid type for a field (e.g. non-int port) raises ValidationError."""
    with mock.patch.dict(os.environ, {"QDRANT_PORT": "invalid-port"}, clear=True):
        with pytest.raises(ValidationError):
            Settings()

def test_logger_integration():
    """Test that the structlog logger is initialized and callable."""
    from config import logger
    logger.info("testing log integration", key="value")
