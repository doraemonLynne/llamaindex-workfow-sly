import asyncio
from settings import init_settings
from llama_index.core.settings import Settings
import dotenv

dotenv.load_dotenv()

# Load LLM
init_settings()
llm = Settings.llm

from llama_index.tools.mcp import McpToolSpec
from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult, ToolCall
from llama_index.core.workflow import Context

SYSTEM_PROMPT = """\
You are an AI assistant for Tool Calling.

Before you help a user, you need to work with tools to interact with Our Database
"""


async def get_agent(tools: McpToolSpec):
    tools = await tools.to_tool_list_async()
    print(f"tools: {[tool.metadata.name for tool in tools]}")
    agent = FunctionAgent(
        name="Agent",
        description="An agent that can work with Our Database software.",
        tools=tools,
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent


async def handle_user_message(
    message_content: str,
    agent: FunctionAgent,
    agent_context: Context,
    verbose: bool = False,
):
    handler = agent.run(message_content, ctx=agent_context)
    async for event in handler.stream_events():
        if verbose and type(event) == ToolCall:
            print(f"Calling tool {event.tool_name} with kwargs {event.tool_kwargs}")
        elif verbose and type(event) == ToolCallResult:
            print(f"Tool {event.tool_name} returned {event.tool_output}")

    response = await handler
    return str(response)


from llama_index.tools.mcp import BasicMCPClient, McpToolSpec


mcp_client = BasicMCPClient("http://127.0.0.1:1122/sse")
mcp_tool = McpToolSpec(client=mcp_client)

# get the agent
agent = asyncio.run(get_agent(mcp_tool)) 

# create the agent context
agent_context = Context(agent)

# Run the agent!
while True:
    user_input = input("Enter your message: ")
    if user_input == "exit":
        break
    print("User: ", user_input)
    response = asyncio.run(handle_user_message(user_input, agent, agent_context, verbose=True))
    print("Agent: ", response)