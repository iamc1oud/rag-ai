from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    CHAT_MODEL: str = Field(...)
    OLLAMA_URL: str = Field(...)
    EMBED_MODEL: str = Field(...)

    model_config = SettingsConfigDict(case_sensitive=False, env_file='.env')

settings = Config()
