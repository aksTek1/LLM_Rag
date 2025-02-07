import os
from typing import List, Optional
from src.rag_smart.config_manager import ConfigManager


class PromptManager:
    """
    Manages prompt templates from files
    """

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize Prompt Manager

        :param prompts_directory: Directory containing prompt template files
        """
        self.prompts_directory = config_manager.config['prompts_dir'].replace(
                                            "{PROJECT_ROOT}", config_manager.project_root)
        self.prompts = {}
        self.prompt_metadata = {}
        self._load_prompts()

    def _load_prompts(self):
        """
        Load all prompt templates from files in the specified directory
        """
        # Ensure directory exists
        if not os.path.exists(self.prompts_directory):
            raise ValueError(f"Prompt directory not found: {self.prompts_directory}")

        # Load prompts from .txt files
        for filename in os.listdir(self.prompts_directory):
            if '_rag_smart_doc_' in filename and filename.endswith('.txt'):
                prompt_name = os.path.splitext(filename)[0]
                print(f'prompt_name: {prompt_name}')
                file_path = os.path.join(self.prompts_directory, filename)

                with open(file_path, 'r') as f:
                    self.prompts[prompt_name] = f.read().strip()

    def get_prompt(self, prompt_name: str) -> Optional[str]:
        """
        Retrieve a specific prompt template

        :param prompt_name: Name of the prompt template
        :return: Prompt template or None
        """
        return self.prompts.get(prompt_name)

    def list_prompts(self) -> List[str]:
        """
        List all available prompt names

        :return: List of prompt names
        """
        return list(self.prompts.keys())

    def get_relevant_sources(self, prompt_name: str) -> List[str]:
        """Return list of source IDs that are relevant for this prompt"""
        metadata = self.prompt_metadata.get(prompt_name, {})
        return metadata.get('relevant_sources', [])