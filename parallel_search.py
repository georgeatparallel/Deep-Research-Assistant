"""Optional anonymous Parallel Search MCP provider."""

import asyncio
import json


MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = (
    "Deep-Research-Assistant/parallel-search "
    "(+https://github.com/Harshita1738/Deep-Research-Assistant)"
)
SEARCH_TIMEOUT = 30


async def search_parallel(query: str, num_results: int, session_id: str) -> list[dict]:
    """Return the app's source records; propagate failures rather than fabricating sources."""
    if num_results <= 0:
        return []
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError as exc:
        raise RuntimeError(
            "Parallel search requires: pip install -r requirements-parallel.txt"
        ) from exc

    async def call_search():
        async with streamablehttp_client(
            MCP_URL,
            headers={"User-Agent": USER_AGENT},
            timeout=SEARCH_TIMEOUT,
            sse_read_timeout=SEARCH_TIMEOUT,
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(
                    "web_search",
                    {
                        "objective": query,
                        "search_queries": [query],
                        "session_id": session_id,
                    },
                )

    response = await asyncio.wait_for(call_search(), timeout=SEARCH_TIMEOUT)
    if response.isError:
        message = " ".join(
            block.text for block in response.content if block.type == "text"
        )
        raise RuntimeError(f"Parallel search failed: {message}")

    data = response.structuredContent
    if data is None:
        # Older MCP servers return the same JSON in a text content block.
        text = next((block.text for block in response.content if block.type == "text"), "")
        data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise RuntimeError("Parallel search returned an invalid results payload.")

    return [
        {
            "title": item.get("title") or "",
            "url": item["url"],
            "snippet": "\n".join(item.get("excerpts", [])),
            "source": "web",
        }
        for item in data["results"][:num_results]
    ]
