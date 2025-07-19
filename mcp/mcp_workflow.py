import asyncio
from typing import Any, Optional, Union
from pydantic import BaseModel

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


class MCPWorkflowState(BaseModel):
    """MCP Workflow state management"""
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
"""

    @step
    async def initialize_workflow(
        self, ctx: Context, ev: StartEvent
    ) -> MCPConnectEvent:
        """Initialize workflow and prepare to connect to MCP server"""
        user_msg = ev.user_msg
        if user_msg is None:
            raise ValueError("user_msg is required to run the workflow")
        
        await ctx.set("user_msg", user_msg)
        await ctx.set("state", MCPWorkflowState())
        
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
    ) -> Union[MCPToolCallEvent, StopEvent]:
        """Connect to MCP server and initialize tools"""
        try:
            # Create MCP client
            mcp_client = BasicMCPClient(event.mcp_url)
            
            # Create MCP tool specification
            mcp_tools = McpToolSpec(
                client=mcp_client,
                allowed_tools=event.allowed_tools
            )
            
            # Get tool list and create agent
            tools = await mcp_tools.to_tool_list_async()
            
            agent = FunctionAgent(
                name="MCPAgent",
                description="An agent that can use MCP tools to help users.",
                tools=tools,
                llm=self.llm,
                system_prompt=self.system_prompt,
            )
            
            # Update state
            state = await ctx.get("state")
            state.mcp_client = mcp_client
            state.mcp_tools = mcp_tools
            state.agent = agent
            state.connected = True
            await ctx.set("state", state)
            
            # Get user message and process
            user_msg = await ctx.get("user_msg")
            
            # Use agent to process user message
            agent_context = Context(agent)
            handler = agent.run(user_msg, ctx=agent_context)
            
            # Collect tool calls and results
            tool_calls = []
            tool_results = []
            
            async for event in handler.stream_events():
                if isinstance(event, ToolCall):
                    tool_calls.append(event)
                elif isinstance(event, ToolCallResult):
                    tool_results.append(event)
            
            # Get final response
            response = await handler
            
            return StopEvent(result={
                "response": str(response),
                "tool_calls": [{
                    "tool_name": tc.tool_name,
                    "tool_kwargs": tc.tool_kwargs
                } for tc in tool_calls],
                "tool_results": [{
                    "tool_name": tr.tool_name,
                    "tool_output": tr.tool_output
                } for tr in tool_results]
            })
            
        except Exception as e:
            return StopEvent(result={
                "error": f"Failed to connect to MCP server: {str(e)}",
                "response": "Sorry, unable to connect to MCP server. Please check if the server is running."
            })


# Usage example
async def run_mcp_workflow_example():
    """Run MCP workflow example"""
    from llama_index.llms.openai import OpenAI
    
    # Initialize LLM (you can replace with other LLMs)
    llm = OpenAI(model="gpt-4")
    
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
    
    print("Workflow Result:")
    print(f"Response: {result['response']}")
    if 'tool_calls' in result:
        print(f"Tool Calls: {result['tool_calls']}")
    if 'tool_results' in result:
        print(f"Tool Results: {result['tool_results']}")
    
    return result


if __name__ == "__main__":
    # Run example
    asyncio.run(run_mcp_workflow_example())