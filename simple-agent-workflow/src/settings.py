import os

from llama_index.core import Settings
from llama_index.llms.dashscope import DashScope
# from llama_index.llms.openai import OpenAI
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from dotenv import load_dotenv

load_dotenv()

def init_settings():
    
    # Ensure DashScope API key is set
    if os.getenv("HF_TOKEN") is None:
        raise RuntimeError("HF_TOKEN is missing in environment variables")
    
    # Use DashScope LLM which supports function calling
    # Available models: qwen-turbo, qwen-plus, qwen-max, qwen-vl-plus, qwen-vl-max
    Settings.llm = DashScope(
        model="qwen-max",
        api_key=os.getenv("DASHSCOPE_API_KEY")
    )

    # Settings.llm=HuggingFaceInferenceAPI(
    #     token=os.environ["HF_TOKEN"],
    #     model_name="moonshotai/Kimi-K2-Instruct",
    #     provider="auto"
    # )
    
    # Use Qwen embedding model
    # Settings.embed_model = HuggingFaceEmbedding(
    #     model_name="Qwen/Qwen3-Embedding-0.6B"
    # )