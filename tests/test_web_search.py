"""Exercise provider resolution through the agent's async tool dispatcher."""

import asyncio
import importlib
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

search = importlib.import_module("agent.tools.web_search")


@pytest.fixture(autouse=True)
def isolated_search_config(monkeypatch):
    for name in ("WEB_SEARCH_PROVIDER", "EXA_API_KEY", "PARALLEL_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


@pytest.fixture
def mcp_transport(monkeypatch):
    requests = []
    payload = {
        "results": [
            {
                "title": "Documentation",
                "url": "https://example.com/docs",
                "excerpts": ["Use await."],
            },
            {"title": "Example", "url": "https://example.com/code", "excerpts": ["Working code."]},
        ]
    }
    state = {"payload": payload, "error": False, "text_only": False, "status": 200}

    async def respond(request):
        requests.append(request)
        assert str(request.url) == "https://search.parallel.ai/mcp"
        assert request.headers["user-agent"] == search.SEARCH_USER_AGENT
        assert "authorization" not in request.headers
        if request.method != "POST":
            return httpx.Response(405)
        message = json.loads(request.content)
        method = message["method"]
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "initialize":
            result = {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "parallel-fixture", "version": "1"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "web_search",
                        "description": "Fixture search",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "objective": {"type": "string"},
                                "search_queries": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["objective", "search_queries"],
                        },
                    }
                ]
            }
        elif method == "tools/call":
            assert message["params"]["name"] == "web_search"
            assert message["params"]["arguments"] == {
                "objective": "Python async docs",
                "search_queries": ["Python async docs"],
            }
            if state.get("wait"):
                state["started"].set()
                try:
                    await asyncio.Event().wait()
                finally:
                    state["cancelled"].set()
            if state["status"] != 200:
                return httpx.Response(state["status"], text="Provider unavailable")
            result = {
                "content": [{"type": "text", "text": json.dumps(state["payload"])}],
                "isError": state["error"],
            }
            if not state["text_only"]:
                result["structuredContent"] = state["payload"]
        else:
            raise AssertionError(method)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": result})

    original_client = httpx.AsyncClient

    def client(**kwargs):
        if kwargs.get("headers", {}).get("User-Agent") != search.SEARCH_USER_AGENT:
            return original_client(**kwargs)
        assert kwargs["follow_redirects"] is False
        assert kwargs["timeout"] == search.SEARCH_TIMEOUT_SECONDS
        return original_client(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(search.httpx, "AsyncClient", client)
    return requests, state


async def dispatch(**args):
    tool = StructuredTool.from_function(search.web_search, coroutine=search.aweb_search)
    graph = StateGraph(MessagesState)
    graph.add_node("tools", ToolNode([tool]))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    response = await graph.compile().ainvoke(
        {
            "messages": [
                AIMessage(
                    content="", tool_calls=[{"name": "web_search", "args": args, "id": "search-1"}]
                )
            ]
        }
    )
    return json.loads(response["messages"][-1].content)


@pytest.mark.parametrize("text_only", [False, True])
@pytest.mark.parametrize("include_contents", [False, True])
async def test_unconfigured_dispatch_returns_urls_and_contents(
    mcp_transport, text_only, include_contents
):
    requests, state = mcp_transport
    state["text_only"] = text_only
    result = await dispatch(
        query="Python async docs", num_results=1, include_contents=include_contents
    )
    assert result["success"] is True and result["error"] is None
    rows = json.loads(result["results"])
    assert rows == [
        {
            "title": "Documentation",
            "url": "https://example.com/docs",
            **({"excerpts": ["Use await."]} if include_contents else {}),
        }
    ]
    assert {json.loads(r.content)["method"] for r in requests if r.method == "POST"} == {
        "initialize",
        "notifications/initialized",
        "tools/list",
        "tools/call",
    }


@pytest.mark.parametrize("include_contents", [False, True])
async def test_configured_exa_is_preserved(monkeypatch, include_contents):
    monkeypatch.setenv("EXA_API_KEY", "configured-exa-key")
    calls = []

    class Exa:
        def __init__(self, *, api_key):
            assert api_key == "configured-exa-key"

        def search(self, query, **kwargs):
            calls.append(("search", query, kwargs))
            return "Exa result"

        def search_and_contents(self, query, **kwargs):
            calls.append(("contents", query, kwargs))
            return "Exa result"

    monkeypatch.setattr(search, "Exa", Exa)
    monkeypatch.setattr(
        search, "_parallel_search", AsyncMock(side_effect=AssertionError("No fallback"))
    )
    result = await dispatch(query="Python async docs", include_contents=include_contents)
    assert result == {"success": True, "results": "Exa result", "error": None}
    assert calls == [
        (
            "contents" if include_contents else "search",
            "Python async docs",
            {"num_results": 5, "type": "auto", **({"text": True} if include_contents else {})},
        )
    ]


async def test_explicit_parallel_ignores_exa_and_parallel_keys(monkeypatch, mcp_transport):
    monkeypatch.setenv("EXA_API_KEY", "configured-exa-key")
    monkeypatch.setenv("PARALLEL_API_KEY", "unrelated-parallel-key")
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "parallel")
    assert (await dispatch(query="Python async docs"))["success"] is True


@pytest.mark.parametrize(
    "provider,error",
    [("none", "disabled"), ("exa", "EXA_API_KEY"), ("", "must be"), ("unknown", "must be")],
)
async def test_disabled_invalid_or_explicit_unconfigured_provider_never_connects(
    monkeypatch, provider, error
):
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", provider)
    connect = AsyncMock(side_effect=AssertionError("Should not connect"))
    monkeypatch.setattr(search, "_parallel_search", connect)
    result = await dispatch(query="Python async docs")
    assert result["success"] is False and error in result["error"]
    connect.assert_not_awaited()


@pytest.mark.parametrize(
    "args",
    [
        {"query": " "},
        {"query": "Python async docs", "num_results": 0},
        {"query": "Python async docs", "num_results": 21},
    ],
)
async def test_invalid_arguments_do_not_connect(monkeypatch, args):
    connect = AsyncMock(side_effect=AssertionError("Should not connect"))
    monkeypatch.setattr(search, "_parallel_search", connect)
    assert (await dispatch(**args))["success"] is False
    connect.assert_not_awaited()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"results": [None]},
        {"results": [{"url": "file:///etc/passwd"}]},
        {"results": [{"url": "https:///missing-host"}]},
    ],
)
async def test_malformed_results_are_errors(mcp_transport, payload):
    _, state = mcp_transport
    state["payload"] = payload
    result = await dispatch(query="Python async docs")
    assert result["success"] is False and result["results"] is None


@pytest.mark.parametrize("failure", ["tool", "http"])
async def test_provider_errors_do_not_fall_back(monkeypatch, mcp_transport, failure):
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "parallel")
    monkeypatch.setenv("EXA_API_KEY", "configured-exa-key")
    monkeypatch.setattr(search, "Exa", lambda **kwargs: pytest.fail("Must not fall back"))
    _, state = mcp_transport
    if failure == "tool":
        state.update(error=True, payload={"error": "Free anonymous quota exceeded"})
    else:
        state["status"] = 503
    result = await dispatch(query="Python async docs")
    assert result["success"] is False and result["results"] is None


async def test_timeout_closes_request_and_returns_error(monkeypatch, mcp_transport):
    _, state = mcp_transport
    state.update(wait=True, started=asyncio.Event(), cancelled=asyncio.Event())
    monkeypatch.setattr(search, "SEARCH_TIMEOUT_SECONDS", 0.05)
    result = await dispatch(query="Python async docs")
    assert result["success"] is False and "timed out" in result["error"]
    assert state["started"].is_set() and state["cancelled"].is_set()


async def test_cancellation_reaches_transport(mcp_transport):
    _, state = mcp_transport
    state.update(wait=True, started=asyncio.Event(), cancelled=asyncio.Event())
    task = asyncio.create_task(dispatch(query="Python async docs"))
    await asyncio.wait_for(state["started"].wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert state["cancelled"].is_set()


def test_sync_entrypoint_is_preserved(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "none")
    assert search.web_search("Python async docs")["error"] == "Web search is disabled."


class SearchConsumerModel(FakeMessagesListChatModel):
    """A scripted model consumes the actual tool result; no live inference."""

    def bind_tools(self, tools, **kwargs):
        assert any(getattr(t, "name", None) == "web_search" for t in tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        results = [m for m in messages if isinstance(m, ToolMessage) and m.name == "web_search"]
        if results:
            payload = json.loads(results[-1].content)
            row = json.loads(payload["results"])[0]
            message = AIMessage(content=f"{row['url']}: {row['excerpts'][0]}")
            return ChatResult(generations=[ChatGeneration(message=message)])
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


async def test_runtime_factory_dispatch_and_result_consumption(monkeypatch, mcp_transport):
    from deepagents.backends import StateBackend

    from agent import server

    monkeypatch.setattr(server, "_get_cached_sandbox_backend", lambda thread_id: StateBackend())
    monkeypatch.setattr(
        server, "resolve_github_token", AsyncMock(return_value=("token", None, None))
    )
    monkeypatch.setattr(server, "resolve_triggering_user_identity", lambda *args: None)
    monkeypatch.setattr(server, "ensure_sandbox_for_thread", AsyncMock(return_value=object()))
    monkeypatch.setattr(server, "_configure_sandbox_github_auth", AsyncMock())
    monkeypatch.setattr(server, "aresolve_sandbox_work_dir", AsyncMock(return_value="/workspace"))
    monkeypatch.setattr(server, "construct_system_prompt", lambda **kwargs: "Use web_search.")
    monkeypatch.setenv("LLM_FALLBACK_MODEL_ID", "scripted")
    monkeypatch.setenv("LLM_MODEL_ID", "scripted")
    model = SearchConsumerModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "web_search", "args": {"query": "Python async docs"}, "id": "search-1"}
                ],
            )
        ]
    )
    monkeypatch.setattr(server, "make_model", lambda *args, **kwargs: model)
    config = {
        "configurable": {"thread_id": "search-test", "__is_for_execution__": True},
        "metadata": {},
    }
    agent = await server.get_agent(config)
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Find Python async docs"}]}
    )
    assert result["messages"][-1].content == "https://example.com/docs: Use await."


async def test_exa_failure_does_not_fall_back(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "configured-exa-key")

    class FailingExa:
        def __init__(self, **kwargs):
            pass

        def search_and_contents(self, *args, **kwargs):
            raise RuntimeError("Exa unavailable")

    monkeypatch.setattr(search, "Exa", FailingExa)
    parallel = AsyncMock(side_effect=AssertionError("No provider fallback"))
    monkeypatch.setattr(search, "_parallel_search", parallel)
    result = await dispatch(query="Python async docs")
    assert result["success"] is False and "Exa unavailable" in result["error"]
    parallel.assert_not_awaited()


def test_sync_entrypoint_can_search(mcp_transport):
    result = search.web_search("Python async docs")
    assert result["success"] is True
    assert json.loads(result["results"])[0]["url"] == "https://example.com/docs"


async def test_nonexecution_factory_exposes_no_search(monkeypatch, mcp_transport):
    from agent import server

    captured = {}

    class Graph:
        def with_config(self, config):
            return self

    def create(**kwargs):
        captured.update(kwargs)
        return Graph()

    monkeypatch.setattr(server, "create_deep_agent", create)
    await server.get_agent({"configurable": {"thread_id": "not-executing"}, "metadata": {}})
    assert captured["tools"] == []
    assert mcp_transport[0] == []
