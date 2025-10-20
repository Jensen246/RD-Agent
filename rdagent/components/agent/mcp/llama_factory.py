"""
LLaMA Factory MCP Server Configuration

Environment Variables:
    LLAMA_FACTORY_MCP_URL: MCP service URL (default: http://localhost:8125/mcp)
    LLAMA_FACTORY_MCP_TIMEOUT: Timeout in seconds (default: 60)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """LLaMA Factory MCP configuration settings."""
    
    url: str = "http://localhost:8125"
    timeout: int = 120
    enable_cache: bool = False
    # set LLAMA_FACTORY_MCP_ENABLE_CACHE=true in .env to enable cache
    
    model_config = SettingsConfigDict(
        env_prefix="LLAMA_FACTORY_MCP_",
        extra="allow", # Does it allow extrasettings
    )


SETTINGS = Settings()

