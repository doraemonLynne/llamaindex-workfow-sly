from dotenv import load_dotenv
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.settings import Settings
from src.settings import init_settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_workflow() -> AgentWorkflow:
    load_dotenv()
    init_settings()
    res = AgentWorkflow.from_tools_or_functions( 
        tools_or_functions=[],
        llm=Settings.llm,
        system_prompt="You are a helpful assistant that can tell a joke about Llama.",
    )
    print(f"创建的工作流: {type(res).__name__}")
    return res