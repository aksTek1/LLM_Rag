from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from ollama_manager import OllamaManager
from prompt_manager import PromptManager
from document_manager import DocumentManager
from typing import Dict, List, Optional
from src.rag_smart.config_manager import ConfigManager
from langsmith import traceable


class RAGApplication:
    def __init__(self, model_manager: OllamaManager, config_manager: ConfigManager):
        self.model_manager = model_manager
        self.config_manager = config_manager
        self.prompt_manager = PromptManager(config_manager)
        self.document_manager = DocumentManager(config_manager)
        self.initialize_all_vectorstore()

    @traceable(project_name='LLM_Project_rag_smart')
    def initialize_all_vectorstore(self):
        for source_id in self.document_manager.sources:
            if source_id not in self.document_manager.sources:
                self.document_manager.initialize_vectorstore(source_id,
                                                         self.model_manager.get_embeddings())

    @traceable(project_name='LLM_Project_rag_smart')
    def add_document_source(self, source_type: str, location: str, source_name: Optional[str] = None) -> str:
        if source_type == "url":
            source_id = self.document_manager.add_url_source(location, source_name)
        elif source_type == "file":
            source_id = self.document_manager.add_file_source(location, source_name)
        elif source_type == "directory":
            source_id = self.document_manager.add_directory_source(location, source_name)
        else:
            raise ValueError(f"Unknown source type: {source_type}")

        self.document_manager.initialize_vectorstore(
            source_id,
            self.model_manager.get_embeddings()
        )
        return source_id

    @traceable(project_name='LLM_Project_rag_smart')
    def query(self, question: str, prompt_name: str = "default") -> str:
        relevant_sources = self.prompt_manager.get_relevant_sources(prompt_name)
        print(f'relevant_sources: {relevant_sources} for prompt_name: {prompt_name}')

        # If no specific sources are defined, use all available sources
        if not relevant_sources:
            relevant_sources = list(self.document_manager.vectorstores.keys())
            print(f'listing all relevant_sources from vector stores: {relevant_sources}')

        # Combine retrievers from relevant sources
        retrievers = [
            self.document_manager.vectorstores[source_id].as_retriever()
            for source_id in relevant_sources
            if source_id in self.document_manager.vectorstores
        ]

        if not retrievers:
            raise ValueError("No relevant vector stores found for this prompt")
        else:
            print(f'the relevant vector stores found for prompt: {prompt_name} are shown in the list: {retrievers}')
            for i, ret in enumerate(retrievers):
                print(f'ret[{i}]: {retrievers[i]}')

        # use the first retriever
        prompt_template = self.prompt_manager.get_prompt(prompt_name)
        # Create prompt template
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        print(f'Query: prompt: {prompt}, question:{question}')
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.model_manager.llm,
            chain_type="stuff",
            retriever=retrievers[0],
            chain_type_kwargs={"prompt": prompt}
        )

        return qa_chain.invoke(question)
