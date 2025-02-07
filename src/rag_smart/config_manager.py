import yaml
import os

class ConfigManager:
    def __init__(self, config_path: str = "config.yaml", project_root: str = "./"):
        self.config_path = config_path
        self.project_root = project_root
        self.config = self.load_config()

    def load_config(self) -> dict:
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def get_model_config(self, model_name: str) -> dict:
        return self.config['models'].get(model_name, {})

    def get_embedding_config(self) -> dict:
        return self.config.get('embeddings', {})
