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


class ChartMCPWorkflow(Workflow):
    """
    Workflow for connecting to MCP chart server and generating charts
    Based on antvis/mcp-server-chart project
    """

    def __init__(
        self,
        llm: LLM,
        mcp_server_url: str = "http://127.0.0.1:3000/sse",  # antvis chart server default
        allowed_tools: Optional[list[str]] = None,
        **kwargs: Any,
    ):
        """
        Args:
            llm: The LLM model to use
            mcp_server_url: MCP chart server URL (antvis/mcp-server-chart)
            allowed_tools: List of allowed chart tools, None means use all tools
        """
        super().__init__(**kwargs)
        self.llm = llm
        self.mcp_server_url = mcp_server_url
        self.allowed_tools = allowed_tools or [
            "generate_line_chart",
            "generate_bar_chart", 
            "generate_pie_chart",
            "generate_scatter_chart",
            "generate_area_chart",
            "generate_column_chart",
            "generate_histogram_chart",
            "generate_boxplot_chart",
            "generate_radar_chart",
            "generate_funnel_chart",
            "generate_treemap_chart",
            "generate_sankey_chart",
            "generate_word_cloud_chart",
            "generate_violin_chart",
            "generate_venn_chart",
            "generate_liquid_chart",
            "generate_dual_axes_chart",
            "generate_mind_map",
            "generate_organization_chart",
            "generate_flow_diagram",
            "generate_fishbone_diagram",
            "generate_network_graph",
            "generate_district_map",
            "generate_pin_map",
            "generate_path_map"
        ]
        self.system_prompt = """\
You are an AI assistant that can generate various types of charts and visualizations using AntV chart tools.
You can create 25+ different types of charts including:
- Basic charts: line, bar, pie, scatter, area, column
- Statistical charts: histogram, boxplot, violin
- Advanced charts: radar, funnel, treemap, sankey, word cloud
- Diagram charts: mind map, organization chart, flow diagram, fishbone diagram
- Geographic charts: district map, pin map, path map
- Other charts: venn diagram, liquid chart, dual axes chart, network graph

When users request charts, analyze their data and requirements to choose the most appropriate chart type.
Always provide clear explanations of why you chose a specific chart type.
"""

    @step
    async def initialize_workflow(
        self, ctx: Context, ev: StartEvent
    ) -> MCPConnectEvent:
        """Initialize workflow and prepare to connect to MCP chart server"""
        user_msg = ev.user_msg
        if user_msg is None:
            raise ValueError("user_msg is required to run the workflow")
        
        print(f"Initializing chart workflow with message: {user_msg}")
        
        await ctx.set("user_msg", user_msg)
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
    async def connect_chart_server(
        self, ctx: Context, event: MCPConnectEvent
    ) -> StopEvent:
        """Connect to MCP chart server and generate charts"""
        try:
            print(f"Connecting to AntV chart server: {event.mcp_url}")
            
            # Create MCP client
            mcp_client = BasicMCPClient(event.mcp_url)
            
            # Create MCP tool specification
            mcp_tools = McpToolSpec(
                client=mcp_client,
                allowed_tools=event.allowed_tools
            )
            
            # Get tool list and create agent
            print("Getting chart tools from MCP server...")
            tools = await mcp_tools.to_tool_list_async()
            print(f"Available chart tools: {[tool.metadata.name for tool in tools]}")
            
            if not tools:
                return StopEvent(result={
                    "error": "No chart tools available from MCP server",
                    "response": "Chart server connected but no tools are available. Please check the antvis/mcp-server-chart server configuration."
                })
            
            agent = FunctionAgent(
                name="ChartAgent",
                description="An agent that can generate various types of charts using AntV tools.",
                tools=tools,
                llm=self.llm,
                system_prompt=self.system_prompt,
            )
            
            # Update context state
            await ctx.set("mcp_client", mcp_client)
            await ctx.set("mcp_tools", mcp_tools)
            await ctx.set("agent", agent)
            await ctx.set("connected", True)
            
            # Get user message and process
            user_msg = await ctx.get("user_msg")
            print(f"Processing chart request with agent: {user_msg}")
            
            # Use agent to process user message and generate charts
            response = await agent.run(user_msg)
            
            # Collect tool calls and results
            tool_calls_info = []
            tool_results_info = []
            
            # Check if response contains chart generation information
            if hasattr(response, 'source_nodes') and response.source_nodes:
                for node in response.source_nodes:
                    if hasattr(node, 'metadata'):
                        tool_calls_info.append(node.metadata)
            
            # If no automatic chart generation, try manual chart generation based on user request
            if not tool_calls_info and any(keyword in user_msg.lower() for keyword in 
                ['chart', 'graph', 'plot', '图表', '图形', '可视化', 'visualization']):
                print("No automatic chart generation detected, trying manual chart generation...")
                try:
                    # Example: Generate a sample chart based on user request
                    chart_type = self._determine_chart_type(user_msg)
                    sample_data = self._generate_sample_data(chart_type)
                    
                    for tool in tools:
                        if tool.metadata.name == chart_type:
                            print(f"Manually calling chart tool: {tool.metadata.name}")
                            result = await tool.acall(**sample_data)
                            tool_results_info.append({
                                "tool_name": chart_type,
                                "tool_output": str(result),
                                "chart_data": sample_data
                            })
                            break
                except Exception as tool_error:
                    print(f"Manual chart generation failed: {str(tool_error)}")
                    tool_results_info.append({
                        "tool_name": "chart_generation",
                        "tool_output": f"Error: {str(tool_error)}"
                    })
            
            return StopEvent(result={
                "response": str(response),
                "tool_calls": tool_calls_info,
                "tool_results": tool_results_info,
                "available_chart_tools": [tool.metadata.name for tool in tools],
                "chart_server_url": event.mcp_url,
                "chart_capabilities": "25+ chart types including line, bar, pie, scatter, radar, treemap, sankey, word cloud, mind map, and geographic maps"
            })
            
        except Exception as e:
            print(f"Error in connect_chart_server: {str(e)}")
            return StopEvent(result={
                "error": f"Failed to connect to chart server: {str(e)}",
                "response": f"Sorry, unable to connect to AntV chart server at {event.mcp_url}. Please check if the mcp-server-chart is running. Error: {str(e)}"
            })
    
    def _determine_chart_type(self, user_msg: str) -> str:
        """Determine appropriate chart type based on user message"""
        msg_lower = user_msg.lower()
        
        if any(word in msg_lower for word in ['line', '线图', 'trend', '趋势']):
            return "generate_line_chart"
        elif any(word in msg_lower for word in ['bar', '柱状图', 'column', '条形图']):
            return "generate_bar_chart"
        elif any(word in msg_lower for word in ['pie', '饼图', 'proportion', '比例']):
            return "generate_pie_chart"
        elif any(word in msg_lower for word in ['scatter', '散点图', 'correlation', '相关性']):
            return "generate_scatter_chart"
        elif any(word in msg_lower for word in ['area', '面积图', 'filled']):
            return "generate_area_chart"
        elif any(word in msg_lower for word in ['radar', '雷达图', 'spider']):
            return "generate_radar_chart"
        elif any(word in msg_lower for word in ['word cloud', '词云', 'text']):
            return "generate_word_cloud_chart"
        elif any(word in msg_lower for word in ['mind map', '思维导图', 'mindmap']):
            return "generate_mind_map"
        elif any(word in msg_lower for word in ['organization', '组织架构图', 'org chart']):
            return "generate_organization_chart"
        elif any(word in msg_lower for word in ['flow', '流程图', 'flowchart']):
            return "generate_flow_diagram"
        else:
            return "generate_line_chart"  # Default to line chart
    
    def _generate_sample_data(self, chart_type: str) -> dict:
        """Generate sample data for different chart types"""
        if chart_type == "generate_line_chart":
            return {
                "data": [
                    {"month": "Jan", "value": 100},
                    {"month": "Feb", "value": 120},
                    {"month": "Mar", "value": 150},
                    {"month": "Apr", "value": 180},
                    {"month": "May", "value": 200}
                ],
                "xField": "month",
                "yField": "value",
                "title": "Sample Line Chart"
            }
        elif chart_type == "generate_bar_chart":
            return {
                "data": [
                    {"category": "A", "value": 40},
                    {"category": "B", "value": 60},
                    {"category": "C", "value": 80},
                    {"category": "D", "value": 50}
                ],
                "xField": "value",
                "yField": "category",
                "title": "Sample Bar Chart"
            }
        elif chart_type == "generate_pie_chart":
            return {
                "data": [
                    {"type": "A", "value": 27},
                    {"type": "B", "value": 25},
                    {"type": "C", "value": 18},
                    {"type": "D", "value": 15},
                    {"type": "E", "value": 10},
                    {"type": "F", "value": 5}
                ],
                "angleField": "value",
                "colorField": "type",
                "title": "Sample Pie Chart"
            }
        elif chart_type == "generate_word_cloud_chart":
            return {
                "data": [
                    {"text": "visualization", "value": 100},
                    {"text": "chart", "value": 80},
                    {"text": "data", "value": 70},
                    {"text": "analysis", "value": 60},
                    {"text": "graph", "value": 50}
                ],
                "wordField": "text",
                "weightField": "value",
                "title": "Sample Word Cloud"
            }
        else:
            # Default sample data
            return {
                "data": [{"x": 1, "y": 10}, {"x": 2, "y": 20}, {"x": 3, "y": 30}],
                "title": "Sample Chart"
            }