@echo off
echo Loading environment variables...
set MODEL=qwen-max
set EMBEDDING_MODEL=BAAI/bge-base-zh-v1.5
set "NEXT_PUBLIC_STARTER_QUESTIONS=[\"Letter standard in the document\",\"Summarize the document\"]"
set DASHSCOPE_API_KEY=sk-19b104f19fe64400a4c5ebb1e7dda583
set MCP_CHART_SERVER_URL=http://localhost:1122/sse
set MCP_CHART_TIMEOUT=30
set USE_MCP_WORKFLOW=true
echo Environment variables loaded successfully!