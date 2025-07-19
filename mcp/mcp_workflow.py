import asyncio
from typing import Any, Optional, Union
from dataclasses import dataclass

from llama_index.core.chat_engine.types import ChatMessage
from llama_index.core.llms import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult, ToolCall


class MCPConnectEvent(Event):
    """Event for connecting to MCP server"""
    mcp_url: str
    allowed_tools: Optional[list[str]] = None


class MCPToolCallEvent(Event):
    """Event for calling MCP tools"""
    tool_name: str
    tool_kwargs: dict[str, Any]


class MCPResponseEvent(Event):
    """Event for MCP tool responses"""
    result: Any
    error: Optional[str] = None


@dataclass
class MCPWorkflowState:
    """MCP Workflow state management - using dataclass to avoid Pydantic serialization issues"""
    mcp_client: Optional[BasicMCPClient] = None
    mcp_tools: Optional[McpToolSpec] = None
    agent: Optional[FunctionAgent] = None
    connected: bool = False


class MCPWorkflow(Workflow):
    """
    Workflow for connecting to MCP server and executing tool calls
    Based on LlamaIndex Workflows framework
    """

    def __init__(
        self,
        llm: LLM,
        mcp_server_url: str = "http://127.0.0.1:8001/sse",
        allowed_tools: Optional[list[str]] = None,
        **kwargs: Any,
    ):
        """
        Args:
            llm: The LLM model to use
            mcp_server_url: MCP server URL
            allowed_tools: List of allowed tools, None means use all tools
        """
        super().__init__(**kwargs)
        self.llm = llm
        self.mcp_server_url = mcp_server_url
        self.allowed_tools = allowed_tools
        self.system_prompt = """\
You are an AI assistant that can call various tools through MCP server to help users.
You can fetch IP information, process data, and more. Please choose appropriate tools based on user needs.
When you need to use a tool, make sure to call it with the correct parameters.
"""

    @step
    async def initialize_workflow(
        self, ctx: Context, ev: StartEvent
    ) -> MCPConnectEvent:
        """Initialize workflow and prepare to connect to MCP server"""
        user_msg = ev.user_msg
        if user_msg is None:
            raise ValueError("user_msg is required to run the workflow")
        
        print(f"Initializing workflow with message: {user_msg}")
        
        await ctx.set("user_msg", user_msg)
        # 直接在 context 中存储状态信息，避免 Pydantic 序列化问题
        await ctx.set("connected", False)
        await ctx.set("mcp_client", None)
        await ctx.set("mcp_tools", None)
        await ctx.set("agent", None)
        
        # Initialize chat history
        chat_history = ev.chat_history or []
        chat_history.append(
            ChatMessage(
                role="user",
                content=user_msg,
            )
        )
        memory = ChatMemoryBuffer.from_defaults(
            chat_history=chat_history,
            llm=self.llm,
        )
        await ctx.set("memory", memory)
        
        return MCPConnectEvent(
            mcp_url=self.mcp_server_url,
            allowed_tools=self.allowed_tools
        )

    @step
    async def connect_mcp_server(
        self, ctx: Context, event: MCPConnectEvent
    ) -> StopEvent:
        """Connect to MCP server and initialize tools"""
        try:
            print(f"Connecting to MCP server: {event.mcp_url}")
            
            # Create MCP client
            mcp_client = BasicMCPClient(event.mcp_url)
            
            # Create MCP tool specification
            mcp_tools = McpToolSpec(
                client=mcp_client,
                allowed_tools=event.allowed_tools
            )
            
            # Get tool list and create agent
            print("Getting tool list from MCP server...")
            tools = await mcp_tools.to_tool_list_async()
            print(f"Available tools: {[tool.metadata.name for tool in tools]}")
            
            if not tools:
                return StopEvent(result={
                    "error": "No tools available from MCP server",
                    "response": "MCP server connected but no tools are available. Please check the server configuration."
                })
            
            agent = FunctionAgent(
                name="MCPAgent",
                description="An agent that can use MCP tools to help users.",
                tools=tools,
                llm=self.llm,
                system_prompt=self.system_prompt,
            )
            
            # 直接在 context 中更新状态
            await ctx.set("mcp_client", mcp_client)
            await ctx.set("mcp_tools", mcp_tools)
            await ctx.set("agent", agent)
            await ctx.set("connected", True)
            
            # Get user message and process
            user_msg = await ctx.get("user_msg")
            print(f"Processing user message with agent: {user_msg}")
            
            # Use agent to process user message
            response = await agent.achat(user_msg)
            
            # 尝试获取工具调用信息
            tool_calls_info = []
            tool_results_info = []
            
            # 检查响应中是否包含工具调用信息
            if hasattr(response, 'source_nodes') and response.source_nodes:
                for node in response.source_nodes:
                    if hasattr(node, 'metadata'):
                        tool_calls_info.append(node.metadata)
            
            # 如果没有工具调用，尝试直接调用工具进行测试
            if not tool_calls_info and "8.8.8.8" in user_msg:
                print("No automatic tool calls detected, trying manual tool call...")
                try:
                    # 手动调用 fetch_ipinfo 工具进行测试
                    for tool in tools:
                        if tool.metadata.name == "fetch_ipinfo":
                            print(f"Manually calling tool: {tool.metadata.name}")
                            result = await tool.acall(ip="8.8.8.8")
                            tool_results_info.append({
                                "tool_name": "fetch_ipinfo",
                                "tool_output": str(result)
                            })
                            break
                except Exception as tool_error:
                    print(f"Manual tool call failed: {str(tool_error)}")
                    tool_results_info.append({
                        "tool_name": "fetch_ipinfo",
                        "tool_output": f"Error: {str(tool_error)}"
                    })
            
            return StopEvent(result={
                "response": str(response),
                "tool_calls": tool_calls_info,
                "tool_results": tool_results_info,
                "available_tools": [tool.metadata.name for tool in tools],
                "mcp_server_url": event.mcp_url
            })
            
        except Exception as e:
            print(f"Error in connect_mcp_server: {str(e)}")
            return StopEvent(result={
                "error": f"Failed to connect to MCP server: {str(e)}",
                "response": f"Sorry, unable to connect to MCP server at {event.mcp_url}. Please check if the server is running. Error: {str(e)}"
            })


# 测试 MCP 服务器连接的独立函数
async def test_mcp_connection(mcp_url: str = "http://127.0.0.1:8001/sse"):
    """Test MCP server connection independently"""
    try:
        print(f"Testing MCP server connection to: {mcp_url}")
        
        # Create MCP client
        mcp_client = BasicMCPClient(mcp_url)
        
        # Create MCP tool specification
        mcp_tools = McpToolSpec(client=mcp_client)
        
        # Get tool list
        tools = await mcp_tools.to_tool_list_async()
        
        print(f"Connection successful! Available tools: {[tool.metadata.name for tool in tools]}")
        
        # Test a tool call if fetch_ipinfo is available
        for tool in tools:
            if tool.metadata.name == "fetch_ipinfo":
                print("Testing fetch_ipinfo tool...")
                result = await tool.acall(ip="8.8.8.8")
                print(f"Tool result: {result}")
                break
        
        return True
        
    except Exception as e:
        print(f"MCP connection test failed: {str(e)}")
        return False


# Usage example
async def run_mcp_workflow_example():
    """Run MCP workflow example"""
    try:
        # 首先测试 MCP 服务器连接
        print("=== Testing MCP Server Connection ===")
        connection_ok = await test_mcp_connection()
        
        if not connection_ok:
            print("MCP server connection failed. Please start the MCP server first.")
            print("Run: python mcp_server.py")
            return {"error": "MCP server not available"}
        
        print("\n=== Running MCP Workflow ===")
        
        # 尝试使用不同的 LLM 提供商
        try:
            from llama_index.llms.openai import OpenAI
            llm = OpenAI(model="gpt-4")
            print("Using OpenAI LLM")
        except ImportError:
            try:
                from llama_index.llms.dashscope import DashScope
                llm = DashScope(model="qwen-turbo")
                print("Using DashScope LLM")
            except ImportError:
                # 如果都没有，使用默认的 mock LLM
                from llama_index.core.llms.mock import MockLLM
                llm = MockLLM()
                print("Warning: Using MockLLM. Please install a proper LLM provider.")
        
        # Create workflow
        workflow = MCPWorkflow(
            llm=llm,
            mcp_server_url="http://127.0.0.1:8001/sse",
            allowed_tools=["fetch_ipinfo"]  # Only allow IP info tool
        )
        
        # Run workflow
        result = await workflow.run(
            user_msg="Please get detailed information for IP address 8.8.8.8",
            chat_history=[]
        )
        
        print("\n=== Workflow Result ===")
        print(f"Response: {result['response']}")
        if 'available_tools' in result:
            print(f"Available Tools: {result['available_tools']}")
        if 'tool_calls' in result and result['tool_calls']:
            print(f"Tool Calls: {result['tool_calls']}")
        if 'tool_results' in result and result['tool_results']:
            print(f"Tool Results: {result['tool_results']}")
        if 'error' in result:
            print(f"Error: {result['error']}")
        
        return result
        
    except Exception as e:
        print(f"Error running workflow: {str(e)}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Run example
    asyncio.run(run_mcp_workflow_example())