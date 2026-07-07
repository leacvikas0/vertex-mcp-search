from __future__ import annotations

import json
import logging
import mimetypes
import os
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import google.auth
from google.auth.exceptions import DefaultCredentialsError
from google import genai
from google.genai import types


logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("google_genai._api_client").setLevel(logging.WARNING)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

MODEL_NAME = "gemini-3.5-flash"
DEFAULT_GOOGLE_CLOUD_LOCATION = "global"
MAX_INLINE_BYTES = 100 * 1024 * 1024
MAX_INLINE_PDF_BYTES = 50 * 1024 * 1024
TEXT_SNIFF_BYTES = 8192

TEXT_FILE_EXTENSIONS = {
    ".astro",
    ".bat",
    ".c",
    ".cjs",
    ".cfg",
    ".conf",
    ".cpp",
    ".cs",
    ".css",
    ".csv",
    ".cts",
    ".diff",
    ".dockerfile",
    ".env",
    ".gitattributes",
    ".gitignore",
    ".go",
    ".gql",
    ".gradle",
    ".graphql",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsonc",
    ".jsonl",
    ".jsx",
    ".less",
    ".lock",
    ".log",
    ".md",
    ".mjs",
    ".mts",
    ".patch",
    ".php",
    ".properties",
    ".ps1",
    ".py",
    ".rb",
    ".rst",
    ".rs",
    ".sass",
    ".scss",
    ".sh",
    ".sql",
    ".svelte",
    ".svg",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".vue",
    ".xml",
    ".yaml",
    ".yml",
}

TEXT_FILE_NAMES = {
    ".babelrc",
    ".dockerignore",
    ".editorconfig",
    ".env",
    ".env.local",
    ".env.sample",
    ".eslintrc",
    ".gitattributes",
    ".gitignore",
    ".npmrc",
    ".prettierignore",
    ".prettierrc",
    ".stylelintrc",
    ".yarnrc",
    "dockerfile",
    "makefile",
    "procfile",
}

SYSTEM_PROMPT = """
You are a versatile, high-quality assistant powered by Vertex Gemini. You have live Google Search grounding available, plus attached files as first-party context when provided.

When to search — use your judgment, but default to searching for facts:
- ALWAYS search for: current events, news, releases, prices, dates, public stats, anything external, and anything that could have changed since your training cutoff. You cannot reliably know what is "current" from memory, so when recency matters, search rather than guess.
- DO NOT search for: subjective opinions or judgments (e.g. "does this design look good?", "which poem is better?"), pure reasoning, logic, math, code review, writing/editing help, and anything fully contained in the attached files.
- Rule of thumb: "Could this fact be wrong or outdated in the real world?" If yes, search. If it's purely a matter of reasoning, taste, or the attached files, answer directly. Searching a self-contained question wastes the user's quota for nothing.

Match your response to the user's intent:
- Quick factual lookup: give a direct, accurate answer. Search if it's current/external. Keep it tight.
- Advice / recommendations: give clear, well-reasoned guidance. Lay out options and trade-offs, then a concrete recommendation. Search when it adds real value (current options, prices, best practices); otherwise rely on reasoning and attached files.
- Deep research: investigate thoroughly. Use Google Search for current facts and external claims, cross-reference independent sources, prefer primary sources, distinguish confirmed facts from inference, and flag conflicts. Do not answer external research from memory alone — search and ground it.

Universal rules:
- If files are attached, inspect them directly and treat them as first-party context. Refer to them by filename when the answer depends on them.
- Be direct and genuinely useful. Lead with the answer, then add structure only when it helps.
- Never invent facts, dates, source titles, or URLs. Do not put URLs in your JSON; the system attaches real grounded source URLs.
- If search results are unavailable or insufficient, say so and reflect that in uncertainty_notes and/or a lower confidence_score.
- Admit uncertainty clearly rather than guessing.

Output rules:
- Return only JSON matching the requested schema.
- "answer" is required and always present.
- Use the optional fields only when they add real value; omit them when they would be noise. A simple lookup often needs just "answer". Advice usually benefits from "recommendations". Research usually benefits from "key_points" and "uncertainty_notes".
- Include "confidence_score" only when a reliability signal is genuinely useful to the user.
""".strip()

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The main response to the user. Always present.",
        },
        "key_points": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional concise key points when a quick-scan list adds value.",
        },
        "recommendations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional actionable recommendations for advice or decision queries.",
        },
        "uncertainty_notes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional caveats, gaps, or conflicting evidence. Omit when not relevant.",
        },
        "confidence_score": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Optional self-scored confidence from 0.0 to 1.0. Include only when a reliability signal is useful.",
        },
    },
    "required": ["answer"],
    "propertyOrdering": [
        "answer",
        "key_points",
        "recommendations",
        "uncertainty_notes",
        "confidence_score",
    ],
}


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.wfile.write(body)
    handler.close_connection = True


def _get_attr(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _finish_reason(candidate: Any) -> str | None:
    reason = _get_attr(candidate, "finish_reason")
    if reason is None:
        return None
    return _get_attr(reason, "name", str(reason))


def _blocked_error(response: Any) -> dict[str, Any] | None:
    prompt_feedback = _get_attr(response, "prompt_feedback")
    block_reason = _get_attr(prompt_feedback, "block_reason")
    if block_reason:
        return {
            "ok": False,
            "error": {
                "type": "blocked",
                "message": "Gemini blocked the prompt before generation.",
                "block_reason": _get_attr(block_reason, "name", str(block_reason)),
            },
        }

    candidates = _get_attr(response, "candidates", []) or []
    if not candidates:
        return {
            "ok": False,
            "error": {
                "type": "empty_response",
                "message": "Gemini returned no candidates.",
            },
        }

    finish_reason = _finish_reason(candidates[0])
    if finish_reason and finish_reason.upper() in {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}:
        return {
            "ok": False,
            "error": {
                "type": "blocked",
                "message": "Gemini blocked the generated response.",
                "finish_reason": finish_reason,
            },
        }

    return None


def _extract_sources(response: Any) -> list[dict[str, str]]:
    candidates = _get_attr(response, "candidates", []) or []
    if not candidates:
        return []

    metadata = _get_attr(candidates[0], "grounding_metadata")
    chunks = _get_attr(metadata, "grounding_chunks", []) or []
    sources: list[dict[str, str]] = []
    seen_uris: set[str] = set()

    for chunk in chunks:
        web = _get_attr(chunk, "web")
        uri = _get_attr(web, "uri")
        if not uri or uri in seen_uris:
            continue
        seen_uris.add(uri)
        sources.append(
            {
                "title": str(_get_attr(web, "title", "") or ""),
                "url": str(uri),
            }
        )

    return sources


def _extract_search_queries(response: Any) -> list[str]:
    candidates = _get_attr(response, "candidates", []) or []
    if not candidates:
        return []
    metadata = _get_attr(candidates[0], "grounding_metadata")
    queries = _get_attr(metadata, "web_search_queries", []) or []
    return [str(query) for query in queries]


def _parse_model_json(response: Any) -> dict[str, Any]:
    text = getattr(response, "text", None) or ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gemini returned invalid JSON: {exc.msg}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Gemini JSON output was not an object.")

    answer = parsed.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Gemini JSON output is missing a non-empty string field 'answer'.")

    cleaned: dict[str, Any] = {"answer": answer}

    for field in ("key_points", "recommendations", "uncertainty_notes"):
        value = parsed.get(field)
        if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
            cleaned[field] = value

    confidence = parsed.get("confidence_score")
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        cleaned["confidence_score"] = max(0.0, min(1.0, float(confidence)))

    return cleaned


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _model_timeout_ms() -> int:
    configured = _env_int("GEMINI_MODEL_REQUEST_TIMEOUT_MS", 0)
    if configured > 0:
        return configured

    worker_timeout_seconds = float(os.environ.get("GEMINI_WORKER_REQUEST_TIMEOUT_SECONDS", "180"))
    return max(1000, int((worker_timeout_seconds - 30) * 1000))


def _http_retry_options() -> types.HttpRetryOptions:
    attempts = _env_int("GEMINI_HTTP_RETRY_ATTEMPTS", 1)
    return types.HttpRetryOptions(
        attempts=max(1, attempts),
        initial_delay=0.5,
        max_delay=2.0,
        http_status_codes=[408, 429, 500, 502, 503, 504],
    )


def _client() -> genai.Client:
    auth_mode = os.environ.get("GEMINI_AUTH_MODE", "").strip().lower()
    wants_vertex = (
        auth_mode in {"google", "gcp", "vertex", "vertexai", "enterprise", "adc"}
        or _env_truthy("GOOGLE_GENAI_USE_ENTERPRISE")
        or _env_truthy("GOOGLE_GENAI_USE_VERTEXAI")
        or bool(os.environ.get("GOOGLE_CLOUD_PROJECT"))
        or bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    )

    if wants_vertex:
        project = (
            os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GOOGLE_PROJECT_ID")
            or os.environ.get("GCP_PROJECT")
        )
        if not project:
            try:
                _, adc_project = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                project = adc_project
            except DefaultCredentialsError as exc:
                raise RuntimeError(
                    "Vertex auth was selected, but Application Default Credentials "
                    "were not found. Run: gcloud auth application-default login"
                ) from exc

        if not project:
            raise RuntimeError(
                "Vertex auth needs a project. Set GOOGLE_CLOUD_PROJECT or run "
                "gcloud config set project YOUR_PROJECT_ID."
            )

        location = (
            os.environ.get("GOOGLE_CLOUD_LOCATION")
            or os.environ.get("GOOGLE_CLOUD_REGION")
            or DEFAULT_GOOGLE_CLOUD_LOCATION
        )
        return genai.Client(
            enterprise=True,
            project=project,
            location=location,
            http_options=types.HttpOptions(
                api_version="v1",
                timeout=_model_timeout_ms(),
                retry_options=_http_retry_options(),
            ),
        )

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        return genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=_model_timeout_ms(),
                retry_options=_http_retry_options(),
            ),
        )

    raise RuntimeError(
        "No Gemini auth configured. For Vertex, set GEMINI_AUTH_MODE=vertex, "
        "GOOGLE_CLOUD_PROJECT, and GOOGLE_APPLICATION_CREDENTIALS."
    )


def _looks_like_text(path: Path) -> bool:
    try:
        with path.open("rb") as file:
            chunk = file.read(TEXT_SNIFF_BYTES)
    except OSError:
        return False

    if b"\x00" in chunk:
        return False
    if not chunk:
        return True

    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            chunk.decode(encoding)
            return True
        except UnicodeDecodeError:
            continue
    return False


def _guess_mime_type(path_or_uri: str, local_path: Path | None = None) -> str:
    guessed_path = Path(path_or_uri.split("?", 1)[0])
    suffix = guessed_path.suffix.lower()
    name = guessed_path.name.lower()
    if suffix in TEXT_FILE_EXTENSIONS or name in TEXT_FILE_NAMES:
        return "text/plain"

    mime_type, _ = mimetypes.guess_type(path_or_uri)
    if mime_type:
        return mime_type

    if local_path is not None and _looks_like_text(local_path):
        return "text/plain"

    return "application/octet-stream"


def _normalize_file_paths(file_paths: Any) -> list[str]:
    if file_paths is None:
        return []
    if isinstance(file_paths, str):
        return [file_paths]
    if not isinstance(file_paths, list):
        raise ValueError("file_paths must be a list of file path strings.")
    normalized = []
    for item in file_paths:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("file_paths must only contain non-empty strings.")
        normalized.append(item.strip())
    return normalized


def _file_size_limit(mime_type: str) -> int:
    if mime_type == "application/pdf":
        return MAX_INLINE_PDF_BYTES
    return MAX_INLINE_BYTES


def _build_file_parts(file_paths: Any) -> tuple[list[Any], list[dict[str, Any]]]:
    paths = _normalize_file_paths(file_paths)
    parts: list[Any] = []
    metadata: list[dict[str, Any]] = []
    total_bytes = 0

    for index, raw_path in enumerate(paths, start=1):
        if raw_path.startswith("gs://") or raw_path.startswith(("http://", "https://")):
            mime_type = _guess_mime_type(raw_path)
            parts.append(f"Attached file {index}: {raw_path} ({mime_type})")
            parts.append(types.Part.from_uri(file_uri=raw_path, mime_type=mime_type))
            metadata.append(
                {
                    "path": raw_path,
                    "mime_type": mime_type,
                    "transport": "uri",
                }
            )
            continue

        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Attached file not found: {raw_path}")
        if not path.is_file():
            raise ValueError(f"Attached path is not a file: {raw_path}")

        mime_type = _guess_mime_type(str(path), path)
        size = path.stat().st_size
        limit = _file_size_limit(mime_type)
        if size > limit:
            limit_mb = limit // (1024 * 1024)
            raise ValueError(
                f"Attached file is too large for inline upload ({path.name}, "
                f"{size} bytes). Limit for {mime_type} is {limit_mb} MB."
            )

        total_bytes += size
        if total_bytes > MAX_INLINE_BYTES:
            raise ValueError(
                "Attached files exceed the total inline payload limit of 100 MB."
            )

        parts.append(f"Attached file {index}: {path.name} ({mime_type}, {size} bytes)")
        parts.append(types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type))
        metadata.append(
            {
                "path": str(path),
                "name": path.name,
                "mime_type": mime_type,
                "size_bytes": size,
                "transport": "inline",
            }
        )

    return parts, metadata


def _build_contents(query: str, file_parts: list[Any]) -> list[Any]:
    if not file_parts:
        return [query]
    return [
        query,
        "Use the following attached files as user-provided context. Inspect them before answering.",
        *file_parts,
    ]


def _generate_content(client: genai.Client, contents: list[Any]) -> Any:
    return client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=1.0,
            thinking_config=types.ThinkingConfig(thinking_level="high"),
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
        ),
    )


def _error_payload(exc: Exception) -> dict[str, Any]:
    message = str(exc)
    class_name = exc.__class__.__name__
    upper_message = message.upper()

    if "RESOURCE_EXHAUSTED" in upper_message or "429" in message:
        return {
            "ok": False,
            "error": {
                "type": "quota_exhausted",
                "message": (
                    "Vertex AI returned 429 RESOURCE_EXHAUSTED. The Google Cloud "
                    "project/location is out of available Gemini quota right now."
                ),
                "details": message,
            },
        }

    if isinstance(exc, (TimeoutError, socket.timeout)) or "TIMEOUT" in class_name.upper():
        return {
            "ok": False,
            "error": {
                "type": "vertex_timeout",
                "message": (
                    "The Vertex Gemini request took too long and was stopped before "
                    "the MCP host timeout."
                ),
                "details": message,
            },
        }

    if "MIMETYPE" in upper_message and "NOT SUPPORTED" in upper_message:
        return {
            "ok": False,
            "error": {
                "type": "unsupported_file_type",
                "message": (
                    "Gemini rejected one attached file's MIME type. Use a common "
                    "text/PDF/image extension or paste the file text into the query."
                ),
                "details": message,
            },
        }

    return {
        "ok": False,
        "error": {
            "type": class_name,
            "message": message,
        },
    }


def deep_research_payload(query: str, file_paths: list[str] | None = None) -> dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        return {
            "ok": False,
            "error": {
                "type": "invalid_request",
                "message": "query must be a non-empty string.",
            },
        }

    try:
        client = _client()
        file_parts, attached_files = _build_file_parts(file_paths)
        response = _generate_content(client, _build_contents(query.strip(), file_parts))

        blocked = _blocked_error(response)
        if blocked:
            return blocked

        payload = _parse_model_json(response)
        sources = _extract_sources(response)
        search_queries = _extract_search_queries(response)

        # Optional second pass to force grounding for research-style queries.
        # Off by default: it doubles Vertex quota usage on every non-grounded call.
        # Enable with GEMINI_FORCE_GROUNDING=1 when you want maximum research rigor.
        if _env_truthy("GEMINI_FORCE_GROUNDING") and not sources and not file_parts:
            retry_query = (
                "You must use Google Search grounding before answering this research query. "
                "Do not answer from memory alone. If no reliable sources are available, say so.\n\n"
                f"Research query: {query.strip()}"
            )
            retry_response = _generate_content(client, _build_contents(retry_query, file_parts))
            retry_blocked = _blocked_error(retry_response)
            if retry_blocked:
                return retry_blocked

            retry_sources = _extract_sources(retry_response)
            if retry_sources:
                payload = _parse_model_json(retry_response)
                sources = retry_sources
                search_queries = _extract_search_queries(retry_response)

        # Light caveats only when grounding was expected but missing.
        if not sources and not file_parts:
            notes = payload.setdefault("uncertainty_notes", [])
            if search_queries:
                caveat = "Google Search was used but returned no grounded source URLs for this response."
            else:
                caveat = "No web search was performed; this response relies on the model's knowledge and may be outdated."
            if caveat not in notes:
                notes.append(caveat)
        elif not sources and file_parts:
            notes = payload.setdefault("uncertainty_notes", [])
            caveat = "No external grounding metadata was returned; the answer relies on attached files."
            if caveat not in notes:
                notes.append(caveat)

        payload["sources"] = sources
        payload["web_search_queries"] = search_queries
        payload["attached_files"] = attached_files
        return {"ok": True, "result": payload}

    except Exception as exc:
        return _error_payload(exc)


_DISCONNECTED = (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)


class WorkerHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 30

    def log_message(self, format: str, *args: Any) -> None:
        return

    def handle(self) -> None:
        try:
            super().handle()
        except _DISCONNECTED:
            pass

    def do_GET(self) -> None:
        if urlparse(self.path).path != "/health":
            _json_response(self, 404, {"ok": False, "error": {"type": "not_found"}})
            return
        _json_response(
            self,
            200,
            {
                "ok": True,
                "service": "gemini-deep-research-worker",
                "model": MODEL_NAME,
                "pid": os.getpid(),
            },
        )

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/deep_research":
            _json_response(self, 404, {"ok": False, "error": {"type": "not_found"}})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            request = json.loads(body.decode("utf-8"))
            if not isinstance(request, dict):
                raise ValueError("Request body must be a JSON object.")
            payload = deep_research_payload(
                query=request.get("query", ""),
                file_paths=request.get("file_paths"),
            )
        except _DISCONNECTED:
            return
        except Exception as exc:
            payload = _error_payload(exc)

        try:
            _json_response(self, 200, payload)
        except _DISCONNECTED:
            return


class WorkerServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    host = os.environ.get("GEMINI_WORKER_HOST", "127.0.0.1")
    port = int(os.environ.get("GEMINI_WORKER_PORT", "8765"))
    server = WorkerServer((host, port), WorkerHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
