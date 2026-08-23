from langchain_core.embeddings import Embeddings
from langchain_classic.base_language import BaseLanguageModel
from config import settings
import typing
from langchain_core.language_models import BaseChatModel
from langchain_ollama import OllamaEmbeddings

class EmbeddingService:
    def __init__(self):
        self.model: typing.Optional[str] = settings.EMBED_MODEL
        print(self.model)
        self._embed: typing.Optional[Embeddings] = None

    async def invoke(self, prompt: str):
        if self._embed is None:
            raise ValueError("Embedding model is not initialized. Call `init` first.")

        response = await self._embed.aembed_query(prompt)

        return response

    async def init(self):
        if self.model is None:
            raise ValueError("Model is not set. Set `CHAT_MODEL` environment variable.")

        self._embed = OllamaEmbeddings(model=self.model)

    async def get_embed(self):
        if self._embed is None:
            await self.init()
        return self._embed
