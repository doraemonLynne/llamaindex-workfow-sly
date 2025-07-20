import asyncio
from typing import Any, Optional, Union, List, Dict
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
    allowed_tools: Optional[List[str]] = None
    server_name: Optional[str] = None


class MCPToolCallEvent(Event):
    """Event for calling MCP tools"""
    tool_name: str
    tool_kwargs: Dict[str, Any]


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
    server_info: Optional[Dict[str, Any]] = None


class GenericMCPWorkflow(Workflow):
    """
    Generic workflow for connecting to any MCP server and using its tools.
    This workflow can adapt to different MCP servers and their capabilities.
    """

    def __init__(
        self,
        llm: LLM,
        mcp_server_url: str,
        server_name: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
        agent_name: Optional[str] = None,
        agent_description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Args:
            llm: The LLM model to use
            mcp_server_url: MCP server URL
            server_name: Name of the MCP server (for identification)
            allowed_tools: List of allowed tools, None means use all available tools
            agent_name: Custom name for the agent
            agent_description: Custom description for the agent
            system_prompt: Custom system prompt for the agent
        """
        super().__init__(**kwargs)
        self.llm = llm
        self.mcp_server_url = mcp_server_url
        self.server_name = server_name or "MCP Server"
        self.allowed_tools = allowed_tools
        self.agent_name = agent_name or f"{self.server_name}Agent"
        self.agent_description = agent_description or f"An agent that can use tools from {self.server_name}."
        self.system_prompt = system_prompt or self._get_default_system_prompt()

    def _get_default_system_prompt(self) -> str:
        """Generate a default system prompt based on server configuration"""
        return f"""
You are an AI assistant that can use tools provided by {self.server_name}.
You have access to various tools and capabilities through the MCP (Model Context Protocol) server.

When users make requests:
1. Analyze their requirements carefully
2. Choose the most appropriate tools available
3. Execute the tools with proper parameters
4. Provide clear explanations of what you're doing and why
5. Present results in a user-friendly format

Always be helpful, accurate, and explain your reasoning when using tools.
"""

    @step
    async def initialize_workflow(
        self, ctx: Context, ev: StartEvent
    ) -> MCPConnectEvent:
        """Initialize workflow and prepare to connect to MCP server"""
        user_msg = ev.user_msg
        if user_msg is None:
            raise ValueError("user_msg is required to run the workflow")
        
        print(f"Initializing {self.server_name} workflow with message: {user_msg}")
        
        # Store initial state
        await ctx.set("user_msg", user_msg)
        await ctx.set("connected", False)
        await ctx.set("mcp_client", None)
        await ctx.set("mcp_tools", None)
        await ctx.set("agent", None)
        await ctx.set("server_info", {})
        
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
            allowed_tools=self.allowed_tools,
            server_name=self.server_name
        )

    @step
    async def connect_mcp_server(
        self, ctx: Context, event: MCPConnectEvent
    ) -> StopEvent:
        """Connect to MCP server and process user request"""
        try:
            print(f"Connecting to {event.server_name or 'MCP server'}: {event.mcp_url}")
            
            # Create MCP client
            mcp_client = BasicMCPClient(event.mcp_url)
            
            # Create MCP tool specification
            mcp_tools = McpToolSpec(
                client=mcp_client,
                allowed_tools=event.allowed_tools
            )
            
            # Get tool list and create agent
            print("Getting tools from MCP server...")
            tools = await mcp_tools.to_tool_list_async()
            
            if not tools:
                return StopEvent(result={
                    "error": f"No tools available from {event.server_name}",
                    "response": f"Connected to {event.server_name} but no tools are available. Please check the server configuration.",
                    "server_url": event.mcp_url,
                    "server_name": event.server_name
                })
            
            print(f"Available tools: {[tool.metadata.name for tool in tools]}")
            
            # Create agent with available tools
            agent = FunctionAgent(
                name=self.agent_name,
                description=self.agent_description,
                tools=tools,
                llm=self.llm,
                system_prompt=self.system_prompt,
            )
            
            # Update context state
            server_info = {
                "server_name": event.server_name,
                "server_url": event.mcp_url,
                "available_tools": [tool.metadata.name for tool in tools],
                "tool_count": len(tools)
            }
            
            await ctx.set("mcp_client", mcp_client)
            await ctx.set("mcp_tools", mcp_tools)
            await ctx.set("agent", agent)
            await ctx.set("connected", True)
            await ctx.set("server_info", server_info)
            
            # Get user message and process
            user_msg = await ctx.get("user_msg")
            print(f"Processing request with {self.agent_name}: {user_msg}")
            
            # Use agent to process user message
            response = await agent.run(user_msg)
            
            # Collect tool execution information
            tool_calls_info = []
            tool_results_info = []
            
            # Extract tool call information if available
            if hasattr(response, 'source_nodes') and response.source_nodes:
                for node in response.source_nodes:
                    if hasattr(node, 'metadata'):
                        tool_calls_info.append(node.metadata)
            
            # Check for tool call results in the response
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_results_info.append({
                        "tool_name": tool_call.tool_name,
                        "tool_input": tool_call.tool_kwargs,
                        "tool_output": str(tool_call.tool_output) if hasattr(tool_call, 'tool_output') else None
                    })
            
            return StopEvent(result={
                "response": str(response),
                "tool_calls": tool_calls_info,
                "tool_results": tool_results_info,
                "server_info": server_info,
                "success": True
            })
            
        except Exception as e:
            error_msg = f"Failed to connect to {event.server_name or 'MCP server'}: {str(e)}"
            print(f"Error in connect_mcp_server: {error_msg}")
            
            return StopEvent(result={
                "error": error_msg,
                "response": f"Sorry, unable to connect to {event.server_name or 'MCP server'} at {event.mcp_url}. Please check if the server is running and accessible. Error: {str(e)}",
                "server_url": event.mcp_url,
                "server_name": event.server_name,
                "success": False
            })

    async def get_server_capabilities(self) -> Dict[str, Any]:
        """Get information about the connected MCP server's capabilities"""
        try:
            if hasattr(self, 'mcp_tools') and self.mcp_tools:
                tools = await self.mcp_tools.to_tool_list_async()
                return {
                    "server_name": self.server_name,
                    "server_url": self.mcp_server_url,
                    "available_tools": [{
                        "name": tool.metadata.name,
                        "description": tool.metadata.description,
                        "parameters": tool.metadata.fn_schema if hasattr(tool.metadata, 'fn_schema') else None
                    } for tool in tools],
                    "tool_count": len(tools)
                }
        except Exception as e:
            return {
                "error": f"Failed to get server capabilities: {str(e)}",
                "server_name": self.server_name,
                "server_url": self.mcp_server_url
            }


class MultiMCPWorkflow(Workflow):
    """
    Advanced workflow that can connect to multiple MCP servers simultaneously
    and coordinate between different tool sets.
    """

    def __init__(
        self,
        llm: LLM,
        mcp_servers: List[Dict[str, Any]],
        **kwargs: Any,
    ):
        """
        Args:
            llm: The LLM model to use
            mcp_servers: List of MCP server configurations, each containing:
                - url: MCP server URL
                - name: Server name
                - allowed_tools: Optional list of allowed tools
                - priority: Optional priority for tool selection
        """
        super().__init__(**kwargs)
        self.llm = llm
        self.mcp_servers = mcp_servers
        self.workflows = {}
        
        # Create individual workflows for each server
        for server_config in mcp_servers:
            workflow = GenericMCPWorkflow(
                llm=llm,
                mcp_server_url=server_config["url"],
                server_name=server_config.get("name"),
                allowed_tools=server_config.get("allowed_tools"),
                **kwargs
            )
            self.workflows[server_config["name"]] = workflow

    @step
    async def initialize_multi_workflow(
        self, ctx: Context, ev: StartEvent
    ) -> StopEvent:
        """Initialize and coordinate multiple MCP workflows"""
        user_msg = ev.user_msg
        if user_msg is None:
            raise ValueError("user_msg is required to run the workflow")
        
        print(f"Initializing multi-MCP workflow with {len(self.workflows)} servers")
        
        # Try to determine which server(s) to use based on user message
        # For now, we'll try all servers and return the best result
        results = []
        
        for server_name, workflow in self.workflows.items():
            try:
                print(f"Trying server: {server_name}")
                result = await workflow.run(user_msg=user_msg, chat_history=ev.chat_history)
                if result and not result.get("error"):
                    results.append({
                        "server_name": server_name,
                        "result": result,
                        "success": True
                    })
                else:
                    results.append({
                        "server_name": server_name,
                        "result": result,
                        "success": False
                    })
            except Exception as e:
                print(f"Error with server {server_name}: {str(e)}")
                results.append({
                    "server_name": server_name,
                    "error": str(e),
                    "success": False
                })
        
        # Find the best result (first successful one, or compile all results)
        successful_results = [r for r in results if r.get("success")]
        
        if successful_results:
            best_result = successful_results[0]
            return StopEvent(result={
                "response": best_result["result"].get("response"),
                "primary_server": best_result["server_name"],
                "all_results": results,
                "success": True
            })
        else:
            return StopEvent(result={
                "error": "No MCP servers were able to process the request successfully",
                "response": "Sorry, none of the available MCP servers could handle your request. Please check server configurations.",
                "all_results": results,
                "success": False
            })