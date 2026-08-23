from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    # Required -- no sensible default, set in .env.
    CHAT_MODEL: str = Field(...)
    OLLAMA_URL: str = Field(...)
    EMBED_MODEL: str = Field(...)

    # Everything below has a default so existing .env files keep working, but
    # all of it is overridable the same way -- one place (this file / .env),
    # not scattered as module-level constants across ingest/chunking/
    # vectorstore/chain, which is where these used to live.
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    K: int = 5
    SCORE_THRESHOLD: float = 0.45
    PERSIST_DIR: str = str(Path(__file__).resolve().parent / "chroma_db")
    COLLECTION_NAME: str = "rag_ai"

    model_config = SettingsConfigDict(case_sensitive=False, env_file='.env')


settings = Config()
