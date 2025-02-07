import os
import json
from dotenv import load_dotenv
from langsmith import Client
from src.rag_smart.rag_application import RAGApplication
from src.rag_smart.model_llama3 import Llama3Manager
from src.rag_smart.config_manager import ConfigManager


def get_project_root(current_file, levels_up):
    """Get the absolute path to the project root directory"""
    # Get the directory containing the current script
    current_dir = os.path.dirname(os.path.abspath(current_file))
    print(f'get_project_root- current_dir: {current_dir}, file:{__file__} ')
    for curLevel in range(levels_up-1):
        print(f'get_project_root- curLevel: {curLevel}, current_dir:{current_dir}')
        current_dir = os.path.dirname(current_dir)

    print(f'get_project_root- current_dir: {current_dir} ')
    # Go up one level to reach project root
    project_root = os.path.dirname(current_dir)
    return project_root


def main():
    # Get project root directory.
    # the root of the project is 2 levels up from current dir
    levels_up = 2
    project_root = get_project_root(__file__, levels_up)
    print(f'project_root:{project_root}')

    # Construct absolute paths for the configs
    config_path = os.path.join(project_root, "config", "config_rag_smart.yaml")
    json_input_path = os.path.join(project_root, "data", "input", "input_rag_smart.json")
    output_path = os.path.join(project_root, "data", "output", "results_rag_smart.json")


    # Enable LangSmith tracing
    load_dotenv(dotenv_path=project_root+'/.env',override=True)
    langsmith_api_key = os.getenv('LANGSMITH_API_KEY')
    rag_smart_langsmith_client = Client(api_key=langsmith_api_key)

    # Initialize RAG application
    config_manager = ConfigManager(config_path, project_root)
    llama_manager = Llama3Manager("llama3", config_manager)
    app = RAGApplication(llama_manager, config_manager)


    url_list = ["https://en.wikipedia.org/wiki/Constitution_of_the_United_States",
                "https://constitutioncenter.org/the-constitution/full-text"]

    # Adding url sources
    for urls in url_list:
        url_source = app.add_document_source("url", urls, "constitution_docs")


    # Load queries from JSON
    with open(json_input_path, 'r') as f:
        input_data = json.load(f)

    # Process queries
    results = app.query(input_data['queries'],
                        input_data.get('prompt', 'default'))

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()