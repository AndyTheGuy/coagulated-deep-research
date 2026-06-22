import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # General Configuration
    ENV: str = Field(default="development")
    LOG_LEVEL: str = Field(default="INFO")

    # Google Vertex AI (Primary Provider for Critical Tasks)
    GCP_PROJECT_ID: str = Field(default="agenticuse")
    GCP_LOCATION: str = Field(default="global")

    # FreeLLMAPI Configuration (Standard/Bulk Tasks)
    FREE_LLM_API_BASE_URL: str = Field(default="http://localhost:8000/v1")
    FREE_LLM_API_KEY: str = Field(default="")
    USE_FREE_LLM_API: bool = Field(default=False)

    # LLM Routing - Model Mapping
    CRITICAL_MODEL: str = Field(default="gemini-3.5-flash")
    STANDARD_MODEL: str = Field(default="auto")
    BULK_MODEL: str = Field(default="auto")

    # Web Search Configuration
    SEARXNG_URL: str = Field(default="http://localhost:8080")

    # Vector Database (Qdrant)
    QDRANT_HOST: str = Field(default="localhost")
    QDRANT_PORT: int = Field(default=6333)
    QDRANT_API_KEY: str = Field(default="")

    # Model Context Protocol (MCP) Configuration
    MCP_SERVERS: dict = Field(default={
        "sequential_thinking": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
            "mock": True
        },
        "knowledge_graph": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-knowledge-graph"],
            "mock": True
        },
        "puppeteer": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-puppeteer"],
            "mock": True
        }
    })

# Singleton settings instance
settings = Settings()

