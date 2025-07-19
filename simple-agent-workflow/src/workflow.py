from dotenv import load_dotenv
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.settings import Settings
from settings import init_settings
import logging
from llama_index.server.models import ChatRequest
from typing import Optional
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_workflow(chat_request: Optional[ChatRequest] = None) -> AgentWorkflow:
    load_dotenv()
    init_settings()
    res = AgentWorkflow.from_tools_or_functions( 
        tools_or_functions=[],
        llm=Settings.llm,
        system_prompt="You are a helpful assistant that can tell a joke about Llama.",
    )
    print(f"创建的工作流: {type(res).__name__}")
    return res

# 正确的测试函数
async def test_workflow_response():
    workflow = create_workflow()
    
    # 使用 user_msg 参数而不是 input
    test_message = "请告诉我一个关于羊驼的笑话"
    
    try:
        # 正确的调用方式
        response = await workflow.run(user_msg=test_message)
        
        # 打印响应结果
        print("=== 工作流响应 ===")
        print(f"响应类型: {type(response)}")
        print(f"响应内容: {response}")
        
        # 如果响应有特定属性，可以进一步提取
        if hasattr(response, 'response'):
            print(f"具体回答: {response.response}")
        elif hasattr(response, 'message'):
            print(f"消息内容: {response.message}")
        
    except Exception as e:
        print(f"运行工作流时出错: {e}")

# 运行测试
if __name__ == "__main__":
    asyncio.run(test_workflow_response())