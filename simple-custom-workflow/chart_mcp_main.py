from fastapi import FastAPI
from llama_index.core.workflow import Workflow
from llama_index.server import LlamaIndexServer, UIConfig
from llama_index.server.models import ChatRequest
from settings import init_settings
from llama_index.core.settings import Settings
from workflow.chart_mcp_workflow import ChartMCPWorkflow



def create_workflow(chat_request: ChatRequest) -> Workflow:
    init_settings()
    workflow = ChartMCPWorkflow(
        llm=Settings.llm,
        mcp_server_url="http://localhost:1122/sse"
    )
    return workflow


def create_app() -> FastAPI:
    app = LlamaIndexServer(
        workflow_factory=create_workflow,
        env="dev",
        ui_config=UIConfig(
            starter_questions=[
                "Draw a pie chart",
                "Write a simple calculator app",
                "Write a guideline on how to use LLM effectively",
            ],
            component_dir="components",
            layout_dir="layout",
        ),
    )
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("chart_mcp_main:app", host="127.0.0.1", port=8081, reload=True)
