---
title: Deep Research Assistant
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# Deep Research Assistant

AI-powered research assistant using a complete RAG pipeline and local LLM (Jan-v1-4B).  
The system generates research queries, retrieves web results, and converts them into structured research reports.

## Optional Parallel web search

Use [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
to gather research sources without an account or API key. The anonymous endpoint
uses Fast mode for search and has free-tier rate limits.

In the environment where you installed the app, install the optional MCP client
(Python 3.10 or newer):

```bash
python -m pip install -r requirements-parallel.txt
```

Add this to your `.env` file, then restart the app:

```dotenv
SEARCH_ENGINE=parallel
```

The app calls `web_search` over Streamable HTTP at `https://search.parallel.ai/mcp`.
Result titles, URLs and excerpts flow into the existing research notes, source
cards and JSON export. Each search has a 30-second timeout. Connection failures
and service errors, including rate limits, fail the research run with an error;
the app does not silently switch providers or substitute demo sources.

Serper remains the default. To use it, leave `SEARCH_ENGINE` unset or set it to
`serper`, and provide `SERPER_API_KEY`. Without that key, Serper returns no web
sources. Selecting Parallel ignores `SERPER_API_KEY` and sends no authorization
header. The local GGUF model and the app's other dependencies are still needed
to generate reports; this option changes only web retrieval.

### Search tests

The tests use a local MCP server and stub the model and Gradio UI, so they do not
download a model or call a live search service:

```bash
python -m pip install aiohttp python-dotenv -r requirements-parallel.txt
python -m unittest discover -s tests -v
```
