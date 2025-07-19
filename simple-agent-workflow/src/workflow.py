from dotenv import load_dotenv

from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.settings import Settings
# from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

# from src.index import get_index
# from src.query import get_query_engine_tool
# from src.citation import CITATION_SYSTEM_PROMPT, enable_citation
from src.settings import init_settings
import logging
from llama_index.server.models import ChatRequest
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_workflow(chat_request: Optional[ChatRequest] = None) -> AgentWorkflow:
    load_dotenv()
    init_settings()
    return AgentWorkflow.from_tools_or_functions( 
        tools_or_functions=[],
        llm=Settings.llm,
        system_prompt="You are a helpful assistant that can tell a joke about Llama.",
    )

# def create_workflow() -> AgentWorkflow:
#     load_dotenv()
#     init_settings()
#     index = get_index()
#     if index is None:
#         raise RuntimeError(
#             "Index not found! Please run `uv run generate` to index the data first."
#         )
#     # Create a query tool with citations enabled
#     query_tool = enable_citation(get_query_engine_tool(index=index))

#     # Define the system prompt for the agent
#     # Append the citation system prompt to the system prompt
#     system_prompt = """You are a helpful assistant"""
#     system_prompt += CITATION_SYSTEM_PROMPT

#     return AgentWorkflow.from_tools_or_functions(
#     tools_or_functions=[query_tool],
#     llm=Settings.llm,
#     system_prompt=system_prompt)

# workflow = create_workflow()