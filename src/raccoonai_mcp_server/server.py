import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, AsyncIterator

from mcp.server.fastmcp import FastMCP, Context
from raccoonai import AsyncRaccoonAI


@dataclass
class RaccoonContext:
    """Context for the Raccoon API client."""
    client: AsyncRaccoonAI
    raccoon_passcode: str


@asynccontextmanager
async def raccoon_lifespan(server: FastMCP) -> AsyncIterator[RaccoonContext]:
    """Initialize the Raccoon AI client."""
    secret_key = os.environ.get("RACCOON_SECRET_KEY")
    if not secret_key:
        raise EnvironmentError("Warning: RACCOON_SECRET_KEY not found in environment variables")
    raccoon_passcode = os.environ.get("RACCOON_PASSCODE")
    if not raccoon_passcode:
        raise EnvironmentError("Warning: RACCOON_PASSCODE not found in environment variables")

    client = AsyncRaccoonAI(secret_key=secret_key)

    try:
        yield RaccoonContext(client=client, raccoon_passcode=raccoon_passcode)
    finally:
        pass


mcp = FastMCP(
    "raccoonai",
    description="MCP server for Raccoon AI LAM API",
    dependencies=["raccoonai"],
    lifespan=raccoon_lifespan
)


@mcp.resource("schema://raccoonai_lam_tool")
def get_lam_request_schema() -> str:
    """Get the schema for LAM API requests."""
    schema = {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {
                "type": "string",
                "description": "The input query string for the request. This is typically the main prompt."
            },
            "chat_history": {
                "type": "array",
                "description": "The history of the conversation as a list of messages to give the model context."
            },
            "app_url": {
                "type": "string",
                "description": "The entrypoint URL for the web agent."
            },
            "response_schema": {
                "type": "object",
                "description": "The expected schema for the response, describing fields and their purposes."
            },
            "max_count": {
                "type": "integer",
                "description": "The maximum number of results to extract (default: 1)."
            },
            "stream": {
                "type": "boolean",
                "description": "Whether the response should be streamed back or not (default: false)."
            },
            "mode": {
                "type": "string",
                "enum": ["deepsearch", "default"],
                "description":
                    """
                    Mode of execution (default: 'default'). Can be set to deepsearch if the user task requires gathering 
                    information from multiple sources or requires research.
                    """
            },
            "advanced": {
                "type": "object",
                "description": "Advanced configuration options for the session."
            }
        }
    }
    return json.dumps(schema, indent=2)


@mcp.tool(name="raccoonai_lam_tool")
async def lam_run(
        query: str,
        response_schema: Optional[Dict[str, Any]],
        app_url: Optional[str] = "",
        chat_history: Optional[List[Dict[str, Any]]] = None,
        max_count: Optional[int] = 1,
        stream: Optional[bool] = True,
        mode: Optional[str] = "default",
        advanced: Optional[Dict[str, Any]] = None,
        ctx: Context = None
) -> str:
    """
    The Raccoon LAM Tool enables AI agents to browse and interact with the web to perform tasks like:
    - Executing simple and complex web tasks and workflows across multiple sites
    - Web navigation and form submission
    - Data extraction from websites
    - Online research and information gathering

    ## Key Features
    - **Web Browsing and Web Tasks**: Automated navigation of web pages and completion of user defined tasks
    - **Data Extraction**: Structured data extraction from websites
    - **Schema Definition**: Define the structure of data you want extracted

    ## Capabilities
    - Search and browse websites
    - Fill out forms and navigate UI elements
    - Extract structured data based on defined schemas
    - Handle multistep tasks across websites

    ## Schemas and Deepsearch
    - Schemas are used only when you want to extract information from the web.
    - Deepsearch is only used if answering the query involves gathering data from multiple sources and detailed reports.
    - Schemas can be used alongside deepsearch.
    - Schemas should not be used when the user query is about performing actions/task rather than data collection


    Args:
        query: The input query string for the request
        response_schema: The expected schema for the response (optional)
        app_url: The entrypoint URL for the web agent (optional)
        chat_history: Chat history as list of messages (optional)
        max_count: Maximum number of results (default: 1)
        stream: Whether to stream responses (default: True)
        mode: Mode of execution ("default" or "deepsearch")
        advanced: Advanced configuration options
        ctx: The context

    Returns:
        The LAM results as a formatted string
    """
    raccoon_ctx: RaccoonContext = ctx.request_context.lifespan_context
    client = raccoon_ctx.client
    raccoon_passcode = raccoon_ctx.raccoon_passcode

    if not chat_history:
        chat_history = []

    if not advanced:
        advanced = {
            "block_ads": False,
            "solve_captchas": False,
            "proxy": False,
            "extension_ids": []
        }

    params = {
        "query": query,
        "schema": response_schema,
        "app_url": app_url,
        "chat_history": chat_history,
        "max_count": max_count,
        "stream": stream,
        "mode": mode,
        "advanced": advanced
    }

    try:
        if stream:
            return await _stream_lam_response(client, raccoon_passcode, params, ctx)
        else:
            return await _fetch_lam_response(client, raccoon_passcode, params)
    except Exception as e:
        return f"Error: Failed to connect to Raccoon API: {str(e)}"


async def _fetch_lam_response(client: AsyncRaccoonAI, raccoon_passcode: str, params: Dict[str, Any]) -> str:
    """Fetch a complete LAM response (non-streamed)."""
    response = await client.lam.run(**params, raccoon_passcode=raccoon_passcode)
    return _format_lam_result(response)


async def _stream_lam_response(
        client: AsyncRaccoonAI,
        raccoon_passcode: str,
        params: Dict[str, Any],
        ctx: Context
) -> str:
    """Stream LAM responses and report progress."""
    last_response = None
    stream = await client.lam.run(**params, raccoon_passcode=raccoon_passcode)

    async for response in stream:
        last_response = response

        if response.task_status == "PROCESSING":
            await ctx.info(response.message)
            await ctx.info(
                f"View livestream: {response.livestream_url if hasattr(response, 'livestream_url') else 'Not available'}")

    if last_response:
        return _format_lam_result(last_response)
    return "Error: No data received from streaming response"


def _format_lam_result(response) -> str:
    """Format a LAM result into a readable string."""
    if hasattr(response, "__dict__"):
        result = response.__dict__
    else:
        result = {
            "task_status": response.task_status,
            "message": response.message,
            "properties": response.properties,
            "data": response.data if hasattr(response, "data") else []
        }
        if hasattr(response, "livestream_url"):
            result["livestream_url"] = response.livestream_url

    status = result.get("task_status", "UNKNOWN")
    message = result.get("message", "")
    data = result.get("data", [])
    properties = result.get("properties", {})

    formatted = f"Status: {status}\n\n"

    if message:
        formatted += f"Message: {message}\n\n"

    if properties:
        formatted += f"Properties: {properties}\n\n"

    if data:
        formatted += "Extracted Data:\n"
        for i, item in enumerate(data, 1):
            formatted += f"\n--- Result {i} ---\n"
            formatted += json.dumps(item, indent=2)

    return formatted


@mcp.prompt()
def extract_data_prompt(website_url: str, data_to_extract: str) -> str:
    """
    Create a prompt for extracting specific data from a website.

    Args:
        website_url: URL of the website to extract data from
        data_to_extract: Description of the data to extract
    """
    return f"""
I need to extract the following information from {website_url}:

{data_to_extract}

Please create a Raccoon LAM query that will extract this data in a structured format. Include:
1. The appropriate schema definition
2. Any advanced settings needed (like CAPTCHA solving if applicable)
3. The base app_url
4. A clear query instructing the web agent
5. A value for mode which can be default or deepsearch
"""


@mcp.prompt()
def execute_web_task_prompt(entrypoint_url: str, task_to_execute: str) -> str:
    """
    Create a prompt for executing actions on one or multiple websites.

    Args:
        entrypoint_url: URL of the website to start the execution from
        task_to_execute: Description of the task to execute
    """
    return f"""
I need to do the task: {task_to_execute} starting from the following website: {entrypoint_url}

Please create a Raccoon LAM query that will:
1. Visit the entrypoint url
2. Execute the required task on behalf of the user
3. Share acknowledgement with the user that the task is successful
"""


@mcp.resource("usage://lam")
def get_usage_info() -> str:
    """Get information about LAM API usage."""
    return """
To view your Raccoon API usage:
1. Visit the usage page on Raccoon Platform at https://platform.flyingraccoon.tech/usage
2. View your current usage and billing information
"""


@mcp.tool()
async def sample_lam_query(ctx: Context) -> str:
    """
    Return a sample LAM query to demonstrate the API functionality.
    """
    sample = {
        "query": "Find three YCombinator startups who got funded in W24",
        "app_url": "https://www.ycombinator.com/companies",
        "schema": {
            "name": "Name of the company as a string",
            "funding_season": "The funding season like W24 as a string",
            "address": {
                "city": "What city is the company located in?",
                "country": "Which country is the company located in?"
            }
        },
        "max_count": 3,
        "stream": True,
        "mode": "deepsearch",
        "advanced": {
            "block_ads": True,
            "solve_captchas": False
        }
    }

    return json.dumps(sample, indent=2)
