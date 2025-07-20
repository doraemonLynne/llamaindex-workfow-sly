import os

from llama_index.core import Settings
# from llama_index.llms.dashscope import DashScope
# from llama_index.llms.openai import OpenAI
# from llama_index.llms.huggingface import HuggingFaceLLM
# from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama

load_dotenv()

def init_settings():
    
    # Ensure DashScope API key is set
    if os.getenv("DASHSCOPE_API_KEY") is None:
        raise RuntimeError("DASHSCOPE_API_KEY is missing in environment variables")
    
    # Use custom DashScope LLM wrapper with function calling enabled
    # Settings.llm = DashScope(
    #     model="qwen-max",
    #     api_key=os.getenv("DASHSCOPE_API_KEY")
    # )

    # Settings.llm=HuggingFaceInferenceAPI(
    #     token=os.environ["HF_TOKEN"],
    #     model_name="moonshotai/Kimi-K2-Instruct",
    #     provider="auto"
    # )

    # Fixed HuggingFaceLLM initialization
    
    # Settings.llm = HuggingFaceInferenceAPI(
    #     model_name="Qwen/Qwen3-0.6B",
    #     token=os.environ["HF_TOKEN"],
    #     provider="auto",  # this will use the best provider available
    # )

    # Alternative: Use Ollama (recommended for local deployment)
    Settings.llm = Ollama(
        model="llama3.2:3b",  # 或者其他你已安装的模型
        base_url="http://192.168.100.213:11434",  # Ollama 默认地址
        request_timeout=120.0,  # 增加超时时间
    )

    # 代码生成模型
    # Settings.llm = Ollama(model="codellama")
    
    # 轻量级模型
    # Settings.llm = Ollama(model="mistral")
    
    # 数学和推理模型
    # Settings.llm = Ollama(model="wizard-math")

    # Alternative: Use OpenAI (recommended for stability)
    # if os.getenv("OPENAI_API_KEY"):
    #     Settings.llm = OpenAI(
    #         model="gpt-4o-mini",
    #         api_key=os.getenv("OPENAI_API_KEY")
    #     )