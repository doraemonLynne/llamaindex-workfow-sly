from dotenv import load_dotenv

from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.settings import Settings
from src.index import get_index
from src.query import get_query_engine_tool
from src.citation import CITATION_SYSTEM_PROMPT, enable_citation
from src.settings import init_settings
import os
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_workflow() -> AgentWorkflow:
    load_dotenv()
    init_settings()

    return AgentWorkflow.from_tools_or_functions(
    tools_or_functions=[],
    llm=Settings.llm,
    system_prompt="You are a helpful assistant")