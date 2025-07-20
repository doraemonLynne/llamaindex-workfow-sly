#server.py
from mcp.server.fastmcp import FastMCP


import argparse

mcp = FastMCP("OnBase", port=3000)


@mcp.tool()
def get_document_info(id: int):
    """Gets document information from the client given a document ID.

    Args:
        id: The ID of the document to retrieve.

    Returns:
        dict: Document information retrieved from the client.

    Raises:
        ClientError: If there is an error retrieving the document from the client.
    """
    return {
        "doc_id": id,
        "doc_name": "Document Name",
        "doc_type": "Document Type",
        "doc_size": 123456,
        "doc_date": "2023-10-01",
        "doc_author": "Author Name",
        "doc_description": "Document description goes here.",
        "doc_tags": ["tag1", "tag2", "tag3"],
    }


if __name__ == "__main__":
    # Start the server
    print("🚀Starting server... ")

    # Debug Mode
    #  uv run mcp dev server.py

    # Production Mode
    # uv run server.py --server_type=sse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server_type", type=str, default="sse", choices=["sse", "stdio"]
    )
    print("Server type: ", parser.parse_args().server_type)
    print("Launching on Port: ", 3000)
    print('Check "http://localhost:3000/sse" for the server status')

    args = parser.parse_args()
    mcp.run(args.server_type)