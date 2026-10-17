"""Exercise app dispatch and the real MCP HTTP client without a local LLM."""

import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from aiohttp import web

import parallel_search


def load_app(engine=None):
    name = "research_app_under_test"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "app.py")
    app = importlib.util.module_from_spec(spec)
    env = {} if engine is None else {"SEARCH_ENGINE": engine}
    with patch.dict(os.environ, env, clear=True), patch("dotenv.load_dotenv"), patch.dict(
        sys.modules,
        {name: app, "llama_cpp": MagicMock(), "gradio": MagicMock()},
    ):
        spec.loader.exec_module(app)
    return app


class ParallelSearchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.app = load_app()
        self.assistant = self.app.DeepResearchAssistant(
            self.app.ResearchConfig(search_engine="parallel", search_api_key="unused-serper-key")
        )
        self.requests = []
        self.mode = "structured"
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.payload = {"results": [
            {"title": "Research source", "url": "https://example.org/research",
             "excerpts": ["Evidence from the source.", "Further findings."]},
            {"title": None, "url": "https://example.org/second", "excerpts": []},
        ]}
        server = web.Application()
        server.router.add_route("*", "/mcp", self.handle)
        self.runner = web.AppRunner(server)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}/mcp"
        self.endpoint_patch = patch.object(parallel_search, "MCP_URL", self.url)
        self.endpoint_patch.start()

    async def asyncTearDown(self):
        self.release.set()
        self.endpoint_patch.stop()
        await self.runner.cleanup()

    async def handle(self, request):
        if request.method != "POST":
            return web.Response(status=405)
        body = await request.json()
        self.requests.append((str(request.url), dict(request.headers), body))
        if self.mode == "http_error":
            return web.Response(status=503, text="Service unavailable")
        method = body["method"]
        if "id" not in body:
            return web.Response(status=202)
        if method == "initialize":
            result = {"protocolVersion": body["params"]["protocolVersion"],
                      "capabilities": {"tools": {}},
                      "serverInfo": {"name": "fixture", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": [{"name": "web_search", "inputSchema": {"type": "object"}}]}
        elif method == "tools/call":
            self.started.set()
            if self.mode == "delay":
                await self.release.wait()
            if self.mode == "tool_error":
                result = {"isError": True, "content": [{"type": "text", "text": "quota fixture"}]}
            elif self.mode == "malformed":
                result = {"content": [{"type": "text", "text": "{}"}]}
            else:
                result = {"content": [{"type": "text", "text": json.dumps(self.payload)}]}
                if self.mode == "structured":
                    result["structuredContent"] = self.payload
        else:
            return web.Response(status=400)
        return web.json_response({"jsonrpc": "2.0", "id": body["id"], "result": result})

    async def test_app_dispatch_request_and_result_contract(self):
        results = await self.assistant.search_web("climate research findings", 1)
        self.assertEqual(results, [{"title": "Research source", "url": "https://example.org/research",
                                   "snippet": "Evidence from the source.\nFurther findings.", "source": "web"}])
        for url, headers, _ in self.requests:
            self.assertEqual(url, self.url)
            self.assertEqual(headers["User-Agent"], parallel_search.USER_AGENT)
            self.assertNotIn("Authorization", headers)
            self.assertNotIn("X-API-KEY", headers)
        call = next(body for _, _, body in self.requests if body["method"] == "tools/call")
        self.assertEqual(call["params"]["name"], "web_search")
        self.assertEqual(call["params"]["arguments"], {
            "objective": "climate research findings", "search_queries": ["climate research findings"],
            "session_id": self.assistant.parallel_session_id,
        })

    async def test_text_json_response_and_nullable_title(self):
        self.mode = "text"
        results = await self.assistant.search_web("research", 5)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[1]["title"], "")
        self.assertEqual(results[1]["snippet"], "")

    async def test_session_is_stable_across_queries(self):
        await self.assistant.search_web("first")
        await self.assistant.search_web("second")
        sessions = [body["params"]["arguments"]["session_id"] for _, _, body in self.requests
                    if body["method"] == "tools/call"]
        self.assertEqual(sessions, [self.assistant.parallel_session_id] * 2)

    async def test_empty_results(self):
        self.payload = {"results": []}
        self.assertEqual(await self.assistant.search_web("no matches"), [])

    async def test_nonpositive_result_limit_does_not_connect(self):
        self.assertEqual(await self.assistant.search_web("research", 0), [])
        self.assertEqual(await self.assistant.search_web("research", -1), [])
        self.assertEqual(self.requests, [])

    async def test_serper_default_and_missing_key_preserved(self):
        config = self.app.ResearchConfig()
        self.assertEqual(config.search_engine, "serper")
        assistant = self.app.DeepResearchAssistant(config)
        self.assertEqual(await assistant.search_web("research"), [])
        assistant.config.search_api_key = "serper-key"
        assistant.search_serper = AsyncMock(return_value=[{"source": "serper-fixture"}])
        self.assertEqual(await assistant.search_web("research", 3), [{"source": "serper-fixture"}])
        assistant.search_serper.assert_awaited_once_with("research", 3)
        self.assertEqual(self.requests, [])

    async def test_environment_selection(self):
        app = load_app(" Parallel ")
        assistant = app.DeepResearchAssistant(app.ResearchConfig())
        self.assertEqual(assistant.config.search_engine, "parallel")
        self.assertTrue(await assistant.search_web("research"))

    async def test_unknown_provider_is_explicit_error(self):
        self.assistant.config.search_engine = "unknown"
        with self.assertRaisesRegex(ValueError, "SEARCH_ENGINE"):
            await self.assistant.search_web("research")

    async def test_missing_optional_dependency_is_actionable(self):
        with patch.dict(sys.modules, {"mcp": None}):
            with self.assertRaisesRegex(RuntimeError, "requirements-parallel.txt"):
                await self.assistant.search_web("research")

    async def test_tool_error_does_not_fall_back(self):
        self.mode = "tool_error"
        self.assistant.search_serper = AsyncMock()
        with self.assertRaisesRegex(RuntimeError, "quota fixture"):
            await self.assistant.search_web("research")
        self.assistant.search_serper.assert_not_awaited()

    async def test_malformed_payload_is_error(self):
        self.mode = "malformed"
        with self.assertRaisesRegex(RuntimeError, "invalid results payload"):
            await self.assistant.search_web("research")

    async def test_http_error_propagates(self):
        self.mode = "http_error"
        with self.assertRaises(Exception):
            await self.assistant.search_web("research")

    async def test_timeout_bounds_the_whole_search(self):
        self.mode = "delay"
        with patch.object(parallel_search, "SEARCH_TIMEOUT", 1):
            with self.assertRaises(asyncio.TimeoutError):
                await self.assistant.search_web("research")
        self.assertTrue(self.started.is_set())

    async def test_cancellation_propagates(self):
        self.mode = "delay"
        task = asyncio.create_task(self.assistant.search_web("research"))
        await asyncio.wait_for(self.started.wait(), timeout=5)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_research_pipeline_consumes_and_exports_sources(self):
        self.assistant.generate_search_queries = Mock(return_value=["climate research"])
        self.assistant.generate_response = Mock(return_value="<final>INTRODUCTION\nEvidence report.</final>")
        captured = {}
        def save_json(text):
            captured.update(json.loads(text))
            return "report.json"
        with patch.object(self.app, "_assistant", self.assistant), patch.object(
            self.app, "_model_loaded", True
        ), patch.object(self.app, "save_txt", return_value="report.txt"), patch.object(
            self.app, "save_json", side_effect=save_json
        ):
            output = await asyncio.to_thread(
                self.app.run_research, "climate", "surface", "general", "recent", "detailed", Mock()
            )
        self.assertIn("Evidence report.", output[0])
        self.assertIn("https://example.org/research", output[1])
        self.assertIn("Evidence from the source.", self.assistant.generate_response.call_args.args[0])
        self.assertEqual(captured["sources"][0]["title"], "Research source")
        self.assertEqual(output[2:4], ("report.txt", "report.json"))


if __name__ == "__main__":
    unittest.main()
