import os

from llama_index.core import Settings
# from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

def init_settings():
    # Ensure DashScope API key is set
    if os.getenv("DASHSCOPE_API_KEY") is None:
        raise RuntimeError("DASHSCOPE_API_KEY is missing in environment variables")
    
    # Use DashScope LLM which supports function calling
    # Available models: qwen-turbo, qwen-plus, qwen-max, qwen-vl-plus, qwen-vl-max
    # Settings.llm = DashScope(
    #     model="qwen-max",
    #     api_key=os.getenv("DASHSCOPE_API_KEY")
    # )
    Settings.llm = HuggingFaceLLM(
        model_name="Qwen/Qwen2.5-7B-Instruct",  # 或其他开源版本
        tokenizer_name="Qwen/Qwen2.5-7B-Instruct",
        device_map="auto",
        model_kwargs={"torch_dtype": "auto"}
    )
    
    # Use Qwen embedding model
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="Qwen/Qwen3-Embedding-0.6B"
    )