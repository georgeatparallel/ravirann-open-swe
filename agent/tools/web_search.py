"""Web search with a keyless default and an explicit Exa compatibility path."""

import asyncio
import json
import logging
import os
from typing import Any
from urllib.parse import urlsplit

import httpx
from exa_py import Exa
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

logger = logging.getLogger(__name__)

PARALLEL_MCP_URL = "https://search.parallel.ai/mcp"
SEARCH_USER_AGENT = "Open-SWE/0.1.0 (+https://github.com/ravirann/open-swe)"
SEARCH_TIMEOUT_SECONDS = 30


def web_search(
    query: str,
    num_results: int = 5,
    include_contents: bool = True,
) -> dict[str, Any]:
    """Search the web for documentation, code examples, repositories, news, or papers.

    Uses anonymous Parallel Search MCP unless Exa is configured. Override with
    WEB_SEARCH_PROVIDER=parallel, exa, or none. Parallel returns page excerpts;
    Exa can return full page contents.

    Args:
        query: The search query.
        num_results: Maximum results to return, from 1 to 20 (default: 5).
        include_contents: Include page excerpts (Parallel) or contents (Exa).

    Returns:
        Dictionary with success, results (a string for the agent), and error.
    """
    return asyncio.run(aweb_search(query, num_results, include_contents))


async def aweb_search(
    query: str,
    num_results: int = 5,
    include_contents: bool = True,
) -> dict[str, Any]:
    """Async runtime implementation; cancellation propagates to the MCP client."""
    provider = os.environ.get("WEB_SEARCH_PROVIDER")
    if provider is None:
        provider = "exa" if os.environ.get("EXA_API_KEY") else "parallel"
    provider = provider.strip().lower()
    if provider == "none":
        return {"success": False, "results": None, "error": "Web search is disabled."}
    if provider not in {"parallel", "exa"}:
        return {
            "success": False,
            "results": None,
            "error": "WEB_SEARCH_PROVIDER must be parallel, exa, or none.",
        }
    if not isinstance(query, str) or not query.strip():
        return {"success": False, "results": None, "error": "Search query must not be empty."}
    if type(num_results) is not int or not 1 <= num_results <= 20:
        return {"success": False, "results": None, "error": "num_results must be from 1 to 20."}

    try:
        async with asyncio.timeout(SEARCH_TIMEOUT_SECONDS):
            if provider == "parallel":
                results = await _parallel_search(query, num_results, include_contents)
            else:
                api_key = os.environ.get("EXA_API_KEY")
                if not api_key:
                    return {
                        "success": False,
                        "results": None,
                        "error": "EXA_API_KEY is required when WEB_SEARCH_PROVIDER=exa.",
                    }
                client = Exa(api_key=api_key)
                if include_contents:
                    result = await asyncio.to_thread(
                        client.search_and_contents,
                        query,
                        text=True,
                        num_results=num_results,
                        type="auto",
                    )
                else:
                    result = await asyncio.to_thread(
                        client.search, query, num_results=num_results, type="auto"
                    )
                results = str(result)
        return {"success": True, "results": results, "error": None}
    except TimeoutError:
        return {"success": False, "results": None, "error": "Web search timed out."}
    except Exception as exc:
        logger.exception("web_search failed for %s", provider)
        return {"success": False, "results": None, "error": f"{type(exc).__name__}: {exc}"}


async def _parallel_search(query: str, num_results: int, include_contents: bool) -> str:
    # A fixed endpoint and explicit headers keep this path anonymous even when
    # the deployment has saved Parallel credentials for other integrations.
    async with httpx.AsyncClient(
        headers={"User-Agent": SEARCH_USER_AGENT},
        timeout=SEARCH_TIMEOUT_SECONDS,
        follow_redirects=False,
    ) as http_client:
        async with streamable_http_client(PARALLEL_MCP_URL, http_client=http_client) as (
            read,
            write,
            _,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                response = await session.call_tool(
                    "web_search", {"objective": query, "search_queries": [query]}
                )
    if response.isError:
        message = "\n".join(block.text for block in response.content if block.type == "text")
        raise ValueError(message or "Parallel search failed.")
    payload = response.structuredContent
    if payload is None:
        text = "\n".join(block.text for block in response.content if block.type == "text")
        payload = json.loads(text)
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("Parallel returned an invalid search response.")
    results = []
    for item in payload["results"][:num_results]:
        if not isinstance(item, dict) or not isinstance(item.get("url"), str):
            raise ValueError("Parallel returned a result without a URL.")
        parsed_url = urlsplit(item["url"])
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
            raise ValueError("Parallel returned an invalid result URL.")
        result = {"title": item.get("title"), "url": item["url"]}
        if include_contents:
            result["excerpts"] = item.get("excerpts", [])
        results.append(result)
    return json.dumps(results, ensure_ascii=False)
