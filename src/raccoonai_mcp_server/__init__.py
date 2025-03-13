import asyncio

from . import server


def main():
    asyncio.run(server.mcp.run_stdio_async())


__all__ = ["main", "server"]
