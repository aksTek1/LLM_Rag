import hashlib
import os
from typing import Dict, Optional
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from document_source import DocumentSource
from config_manager import ConfigManager


class DocumentManager:
    def __init__(self, config_manager: ConfigManager):
        embedding_config = config_manager.get_embedding_config()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=embedding_config.get('chunk_size', 1000),
            chunk_overlap=embedding_config.get('chunk_overlap', 200)
        )

        self.documents_dir = config_manager.config['documents_dir'].replace(
                                            "{PROJECT_ROOT}", config_manager.project_root)
        self.base_persist_dir = config_manager.config['vector_store_dir'].replace(
                                            "{PROJECT_ROOT}", config_manager.project_root)
        self.sources: Dict[str, DocumentSource] = {}
        self.vectorstores: Dict[str, Chroma] = {}
        self.load_initial_documents()


    def load_initial_documents(self):
        """Load all documents from the configured documents directory"""
        if os.path.exists(self.documents_dir):
            for root, _, files in os.walk(self.documents_dir):
                for file in files:
                    if file.endswith(('.txt', '.md', '.pdf')):  # Add more extensions as needed
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, self.documents_dir)
                        self.add_file_source(file_path, source_name=relative_path)


    def generate_source_id(self, location: str) -> str:
        return hashlib.md5(location.encode()).hexdigest()[:8]


    def add_url_source(self, url: str, source_name: Optional[str] = None) -> str:
        source_id = self.generate_source_id(url)
        self.sources[source_id] = DocumentSource(source_id, "url", url)
        return source_id


    def add_file_source(self, file_path: str, source_name: Optional[str] = None) -> str:
        source_id = self.generate_source_id(file_path)
        self.sources[source_id] = DocumentSource(source_id, "file", file_path)
        return source_id


    def add_directory_source(self, dir_path: str, source_name: Optional[str] = None) -> str:
        source_id = self.generate_source_id(dir_path)
        self.sources[source_id] = DocumentSource(source_id, "directory", dir_path)
        return source_id

    def load_documents(self, source_id: str):
        source = self.sources.get(source_id)
        if not source:
            raise ValueError(f"Source {source_id} not found")

        if source.source_type == "url":
            loader = WebBaseLoader(source.location)
        elif source.source_type == "file":
            loader = TextLoader(source.location)
        elif source.source_type == "directory":
            loader = DirectoryLoader(source.location)
        else:
            raise ValueError(f"Unknown source type: {source.source_type}")

        documents = loader.load()
        return self.text_splitter.split_documents(documents)

    def initialize_vectorstore(self, source_id: str, embeddings, documents=None):
        if documents is None:
            documents = self.load_documents(source_id)

        persist_dir = os.path.join(self.base_persist_dir, source_id)
        self.vectorstores[source_id] = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_dir
        )
