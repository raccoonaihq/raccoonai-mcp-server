import sys

from mcp.server.fastmcp import FastMCP, Context
import json
from typing import Dict, List, Optional, Any, AsyncIterator
from contextlib import asynccontextmanager
import os
from dataclasses import dataclass
from raccoonai import AsyncRaccoonAI


@dataclass
class RaccoonContext:
    """Context for the Raccoon API client."""
    client: AsyncRaccoonAI


@asynccontextmanager
async def raccoon_lifespan(server: FastMCP) -> AsyncIterator[RaccoonContext]:
    """Initialize the Raccoon AI client."""
    secret_key = os.environ.get("RACCOON_SECRET_KEY")
    if not secret_key:
        print("Warning: RACCOON_SECRET_KEY not found in environment variables", file=sys.stderr)

    client = AsyncRaccoonAI(base_url="http://localhost:3800")

    try:
        yield RaccoonContext(client=client)
    finally:
        pass


mcp = FastMCP(
    "Raccoon LAM",
    description="MCP server for Raccoon AI LAM API",
    dependencies=["raccoonai", "httpx"],
    lifespan=raccoon_lifespan
)


@mcp.resource("schema://lam_request")
def get_lam_request_schema() -> str:
    """Get the schema for LAM API requests."""
    schema = {
        "type": "object",
        "required": ["query", "raccoon_passcode"],
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
                "description": "Mode of execution (default: 'default')."
            },
            "raccoon_passcode": {
                "type": "string",
                "description": "The raccoon passcode associated with the end user."
            },
            "advanced": {
                "type": "object",
                "description": "Advanced configuration options for the session."
            }
        }
    }
    return json.dumps(schema, indent=2)


@mcp.resource("info://raccoon_lam")
def get_raccoon_lam_info() -> str:
    """Get information about the Raccoon LAM API."""
    return """
# Raccoon LAM (Large Action Model) API

The Raccoon LAM API enables AI agents to browse and interact with the web to perform tasks like:
- Data extraction from websites
- Online research and information gathering
- Web navigation and form submission
- Executing complex workflows across multiple sites

## Key Features
- **Web Browsing**: Automated navigation of web pages
- **Data Extraction**: Structured data extraction from websites
- **Schema Definition**: Define the structure of data you want extracted
- **Streaming Responses**: Get real-time updates on the agent's progress
- **Advanced Options**: Use proxies, solve CAPTCHAs, block ads

## Capabilities
- Search and browse websites
- Fill out forms and navigate UI elements
- Extract structured data based on defined schemas
- Handle multi-step processes across websites
- Stream back thoughts and actions in real-time
"""


@mcp.tool()
async def lam_run(
        query: str,
        raccoon_passcode: str,
        response_schema: Dict[str, Any],
        app_url: Optional[str] = "",
        chat_history: Optional[List[Dict[str, Any]]] = None,
        max_count: Optional[int] = 1,
        stream: Optional[bool] = True,
        mode: Optional[str] = "default",
        advanced: Optional[Dict[str, Any]] = None,
        ctx: Context = None
) -> str:
    """
    Run a Raccoon LAM query to extract data from websites.

    Args:
        query: The input query string for the request
        raccoon_passcode: The raccoon passcode for the end user
        response_schema: The expected schema for the response
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

    if not chat_history:
        chat_history = []

    if not advanced:
        advanced = {
            "block_ads": False,
            "solve_captchas": False,
            "proxy": False,
            "extension_ids": []
        }

    # Prepare request parameters
    params = {
        "query": query,
        "raccoon_passcode": raccoon_passcode,
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
            return await _stream_lam_response(client, params, ctx)
        else:
            return await _fetch_lam_response(client, params)
    except Exception as e:
        return f"Error: Failed to connect to Raccoon API: {str(e)}"


async def _fetch_lam_response(client: AsyncRaccoonAI, params: Dict[str, Any]) -> str:
    """Fetch a complete LAM response (non-streamed)."""
    response = await client.lam.run(**params)
    return _format_lam_result(response)


async def _stream_lam_response(
        client: AsyncRaccoonAI,
        params: Dict[str, Any],
        ctx: Context
) -> str:
    """Stream LAM responses and report progress."""
    last_response = None
    stream = await client.lam.run(**params)

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
            "data": response.data if hasattr(response, "data") else []
        }
        if hasattr(response, "livestream_url"):
            result["livestream_url"] = response.livestream_url

    status = result.get("task_status", "UNKNOWN")
    message = result.get("message", "")
    data = result.get("data", [])

    formatted = f"Status: {status}\n\n"

    if message:
        formatted += f"Message: {message}\n\n"

    if status == "DONE" and data:
        formatted += "Extracted Data:\n"
        for i, item in enumerate(data, 1):
            formatted += f"\n--- Result {i} ---\n"
            formatted += json.dumps(item, indent=2)

    if status == "HUMAN_INTERACTION":
        livestream_url = result.get("livestream_url", "Not available")
        formatted += f"Livestream URL: {livestream_url}\n"

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
"""


@mcp.prompt()
def compare_websites_prompt(website_urls: str, comparison_criteria: str) -> str:
    """
    Create a prompt for comparing data from multiple websites.

    Args:
        website_urls: Comma-separated list of website URLs to compare
        comparison_criteria: What to compare between the websites
    """
    return f"""
I need to compare the following websites:
{website_urls}

I want to compare them based on these criteria:
{comparison_criteria}

Please create a Raccoon LAM query that will:
1. Visit each website
2. Extract the relevant data into a consistent schema
3. Present the results in a way that facilitates comparison
"""


@mcp.resource("usage://lam")
def get_usage_info() -> str:
    """Get information about LAM API usage."""
    return """
To view your Raccoon API usage:
1. Visit your Raccoon dashboard at https://flyingraccoon.tech/dashboard
2. Navigate to the "Usage" section
3. View your current billing cycle information

For programmatic access, you can create an API key with usage tracking permissions.
"""


@mcp.tool()
async def sample_lam_query(ctx: Context) -> str:
    """
    Return a sample LAM query to demonstrate the API functionality.
    """
    sample = {
        "query": "Find three YCombinator startups who got funded in W24",
        "raccoon_passcode": "<end-user-raccoon-passcode>",
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

