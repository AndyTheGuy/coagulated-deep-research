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
        assert settings.GCP_LOCATION == "global"
        assert settings.FREE_LLM_API_BASE_URL == "http://localhost:8000/v1"
        assert settings.CRITICAL_MODEL == "gemini-3.5-flash"
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


def test_mock_llm_state_isolation_and_fallback():
    """Test that is_mock_llm_enabled/set_mock_llm_enabled isolates state per thread and ignores dynamic os.environ changes."""
    import threading
    from config.settings import is_mock_llm_enabled, set_mock_llm_enabled

    # Ensure clean start state
    set_mock_llm_enabled(False)
    with mock.patch.dict(os.environ, {"MOCK_LLM": "false"}):
        assert not is_mock_llm_enabled()

        # 1. Test set/get inside the same thread
        set_mock_llm_enabled(True)
        assert is_mock_llm_enabled()

        # Reset
        set_mock_llm_enabled(False)
        assert not is_mock_llm_enabled()

        # 2. Test that dynamic os.environ changes are ignored
        with mock.patch.dict(os.environ, {"MOCK_LLM": "true"}):
            assert not is_mock_llm_enabled()

        # 3. Test thread isolation
        thread_results = {}

        def thread_a_func():
            set_mock_llm_enabled(True)
            thread_results["A_after_set"] = is_mock_llm_enabled()

        def thread_b_func():
            # Thread B starts after Thread A has set its value to True
            thread_results["B_initial"] = is_mock_llm_enabled()
            set_mock_llm_enabled(False)
            thread_results["B_after_set_false"] = is_mock_llm_enabled()

        t_a = threading.Thread(target=thread_a_func)
        t_b = threading.Thread(target=thread_b_func)

        t_a.start()
        t_a.join()

        t_b.start()
        t_b.join()

        # Thread A set to True should have been isolated to Thread A
        assert thread_results["A_after_set"] is True
        # Thread B should have started with False (default)
        assert thread_results["B_initial"] is False
        assert thread_results["B_after_set_false"] is False
        # Main thread should still be False
        assert not is_mock_llm_enabled()
