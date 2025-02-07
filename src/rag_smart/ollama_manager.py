from abc import ABC, abstractmethod
from langchain_ollama import OllamaEmbeddings
from config_manager import ConfigManager


class OllamaManager(ABC):
    def __init__(self, model_name: str, config_manager: ConfigManager):
        self.model_name = model_name
        self.config = config_manager.get_model_config(model_name)
        self.base_url = self.config.get('base_url', "http://localhost:11434")
        self.temperature = self.config.get('temperature', 1.0)
        self.llm = None
        self.embeddings = None
        #print(f'OllamaManager base: model_name: {self.model_name},  base_url: {self.base_url} ')
        self.initialize_model()

    @abstractmethod
    def initialize_model(self):
        pass

    def get_embeddings(self):
        if not self.embeddings:
            self.embeddings = OllamaEmbeddings(
                base_url=self.base_url,
                model=self.model_name
            )
        return self.embeddings

