# Vertex MCP Search (Gemini Deep Research)

[![GitHub Repository](https://img.shields.io/badge/GitHub-vertex--mcp--search-blue?logo=github)](https://github.com/leacvikas0/vertex-mcp-search)

A Model Context Protocol (MCP) server for Kilo Code, Hermes, and other MCP-compatible environments. This server leverages Google Vertex AI (GCP) and the Gemini 3.5 Flash model to perform intelligent deep research with high thinking capability and live Google Search grounding.

It exposes a single, highly capable tool:

```text
deep_research(query: str, file_paths: list[str] | None = None)
```

### Why Vertex MCP Search?

- **Unmatched Quality:** While typical search/research MCPs often struggle to deliver deep, context-aware answers, `vertex-mcp-search` utilizes Vertex Gemini 3.5 Flash with `thinking_level="high"` and native Google Search grounding. This allows the model to search across multiple websites, cross-reference sources, and compile highly accurate, reliable, and grounded answers.
- **Cost-Effective (Free with GCP Credits):** Running advanced search/research workflows can sometimes be costly. However, by running on Google Cloud Vertex AI, you can take advantage of GCP's **$300 free trial credits**, making this enterprise-grade deep research completely free to run.

### Architectural Design
The MCP process itself is intentionally lightweight. `server.py` starts a local background HTTP worker on `127.0.0.1`, and forwards tool calls to it. `worker.py` handles Google Cloud application-default credentials (ADC), Vertex Gemini calls, search grounding, and optional file inputs. This clean separation of concerns avoids the Windows/FastMCP stdio hang that often occurs when running heavy network calls directly inside the stdio-based MCP process.

## What It Does

- Model locked to `gemini-3.5-flash`
- Uses Vertex/GCP auth so usage bills to the Google Cloud project
- Uses Google Search grounding for research/current facts
- Uses `thinking_level="high"`
- Accepts local file paths and `gs://`/HTTP(S) URIs
- Supports PDFs, images, code files, text, JSON, CSV, Markdown, and similar files
- Treats uncommon local text/config files like `.jsonc`, `.env`, `.lock`,
  `.vue`, `.svelte`, and extensionless config files as `text/plain`
- Returns structured JSON with answer, confidence, uncertainty notes, sources,
  search queries, and attached file metadata
- Returns named JSON errors for quota exhaustion, Vertex timeouts, unsupported
  file types, and worker startup problems

## Setup

Install dependencies:

```powershell
cd C:\Users\silen\Downloads\aweffes
pip install -r requirements.txt
```

Set up Google Cloud auth once:

```powershell
gcloud auth application-default login
gcloud config set project funding-vikas
```

The configured project is currently:

```text
funding-vikas
```

The server expects these environment variables from the MCP host:

```text
GEMINI_AUTH_MODE=vertex
GOOGLE_CLOUD_PROJECT=funding-vikas
GOOGLE_CLOUD_LOCATION=global
GOOGLE_APPLICATION_CREDENTIALS=C:\Users\silen\AppData\Roaming\gcloud\application_default_credentials.json
GEMINI_WORKER_REQUEST_TIMEOUT_SECONDS=180
```

Optional tuning:

```text
GEMINI_MODEL_REQUEST_TIMEOUT_MS=170000
GEMINI_HTTP_RETRY_ATTEMPTS=1
```

The default retry count is intentionally low. Vertex quota errors usually do not
recover inside one MCP call, and long hidden retries make the host look frozen.

## MCP Host Config

Kilo/Hermes should point at:

```text
C:\Users\silen\Downloads\aweffes\server.py
```

with:

```text
C:\Python313\python.exe
```

`server.py` automatically starts `worker.py`; you do not need to run the worker
manually.

## Usage

Simple research:

```text
Use deep_research to research the latest MCP Python SDK setup with sources.
```

File coworker mode:

```json
{
  "query": "Inspect this PDF and summarize the key claims. Cross-check anything current with web sources.",
  "file_paths": ["C:\\Users\\silen\\Downloads\\paper.pdf"]
}
```

Code file mode:

```json
{
  "query": "Review this file and explain what the main function does.",
  "file_paths": ["C:\\path\\to\\script.py"]
}
```

For large code reviews, start narrow. A single 30-40 KB file can summarize well,
but full multi-file review prompts may exceed the practical MCP/Vertex timeout,
especially when quota is under pressure.

## Output Shape

```json
{
  "ok": true,
  "result": {
    "answer": "...",
    "confidence_score": 0.82,
    "uncertainty_notes": ["..."],
    "sources": [{"title": "...", "url": "https://..."}],
    "web_search_queries": ["..."],
    "attached_files": [{"path": "...", "mime_type": "application/pdf"}]
  }
}
```

Errors are returned as JSON instead of crashing stdio:

```json
{
  "ok": false,
  "error": {
    "type": "worker_unavailable",
    "message": "Could not reach Gemini worker..."
  }
}
```

Common errors:

```json
{
  "ok": false,
  "error": {
    "type": "quota_exhausted",
    "message": "Vertex AI returned 429 RESOURCE_EXHAUSTED..."
  }
}
```

This means the Google Cloud project/location is out of available Gemini quota
for now. Wait, reduce request frequency/size, switch location/project, or request
more Vertex quota.

```json
{
  "ok": false,
  "error": {
    "type": "worker_timeout",
    "message": "Gemini worker did not return before the MCP timeout..."
  }
}
```

This usually means the prompt was too heavy for one call or Vertex was slow/rate
limited. Try one file at a time, ask a smaller question, or retry after quota
recovers.

Worker logs are written to:

```text
C:\Users\silen\Downloads\aweffes\logs
```
