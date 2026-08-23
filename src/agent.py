from config import settings
import typing
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

class Agent:
    def __init__(self):
        self.model: typing.Optional[str] = settings.CHAT_MODEL
        self.temperature = 0.7
        self._llm: typing.Optional[BaseChatModel] = None

    def invoke(self, prompt: str):
        if self._llm is None:
            raise ValueError("LLM is not initialized. Call `init` first.")

        response = self._llm.invoke(prompt)

        return response

    def init(self):
        if self.model is None:
            raise ValueError("Model is not set. Set `CHAT_MODEL` environment variable.")

        self._llm = ChatOllama(model=self.model, temperature=self.temperature)

    def get_llm(self):
        if self._llm is None:
            self.init()
        return self._llm
