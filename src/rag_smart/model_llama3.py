import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from ollama_manager import OllamaManager

class Llama3Manager(OllamaManager):
    def initialize_model(self):
        if self.llm is None:
            self.llm = OllamaLLM(
                base_url=self.base_url,
                model=self.model_name,
                temperature=self.temperature,
                request_timeout = 60,
                headers={"User-Agent": os.getenv("USER_AGENT", "LangChain-Client")}
            )
