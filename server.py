from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import atexit
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP


WORKER_HOST = os.environ.get("GEMINI_WORKER_HOST", "127.0.0.1")
WORKER_PORT = int(os.environ.get("GEMINI_WORKER_PORT", "8765"))
WORKER_URL = f"http://{WORKER_HOST}:{WORKER_PORT}"
WORKER_START_TIMEOUT_SECONDS = float(os.environ.get("GEMINI_WORKER_START_TIMEOUT_SECONDS", "20"))
WORKER_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("GEMINI_WORKER_REQUEST_TIMEOUT_SECONDS", "180"))
WORKER_PID_FILE = Path(__file__).parent / "logs" / "worker.pid"

mcp = FastMCP("Gemini Deep Research", log_level="WARNING")
WORKER_PROCESS: subprocess.Popen[bytes] | None = None


def _worker_path() -> Path:
    return Path(__file__).with_name("worker.py")


def _log_path(name: str) -> Path:
    log_dir = Path(os.environ.get("GEMINI_WORKER_LOG_DIR", Path(__file__).parent / "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / name


def _http_json(method: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 5) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        f"{WORKER_URL}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError("Worker returned a non-object JSON response.")
    return parsed


def _worker_info() -> dict[str, Any] | None:
    try:
        return _http_json("GET", "/health", timeout=2)
    except Exception:
        return None


def _worker_healthy() -> bool:
    info = _worker_info()
    return bool(info and info.get("ok"))


def _read_pid_file() -> int | None:
    try:
        return int(WORKER_PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return None


def _write_pid_file(pid: int) -> None:
    try:
        WORKER_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        WORKER_PID_FILE.write_text(str(pid))
    except OSError:
        pass


def _clear_pid_file() -> None:
    try:
        WORKER_PID_FILE.unlink()
    except OSError:
        pass


def _kill_pid(pid: int) -> None:
    try:
        subprocess.run(
            ["taskkill", "/F", "/PID", str(pid)],
            capture_output=True,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        pass


def _start_worker() -> None:
    global WORKER_PROCESS

    info = _worker_info()
    our_pid = _read_pid_file()
    if info and info.get("ok"):
        live_pid = info.get("pid")
        if live_pid is not None and our_pid is not None and int(live_pid) == our_pid:
            return
        if live_pid is not None:
            _kill_pid(int(live_pid))
            time.sleep(0.5)
        elif our_pid is not None:
            _kill_pid(our_pid)
            time.sleep(0.5)

    env = os.environ.copy()
    env.setdefault("GEMINI_WORKER_HOST", WORKER_HOST)
    env.setdefault("GEMINI_WORKER_PORT", str(WORKER_PORT))

    stdout_path = _log_path("worker.stdout.log")
    stderr_path = _log_path("worker.stderr.log")
    stdout = stdout_path.open("ab")
    stderr = stderr_path.open("ab")
    try:
        WORKER_PROCESS = subprocess.Popen(
            [sys.executable, str(_worker_path())],
            cwd=str(Path(__file__).parent),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            close_fds=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    finally:
        stdout.close()
        stderr.close()

    if WORKER_PROCESS is not None:
        _write_pid_file(WORKER_PROCESS.pid)

    deadline = time.monotonic() + WORKER_START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if _worker_healthy():
            return
        time.sleep(0.25)

    _stop_worker()
    raise RuntimeError(
        f"Gemini worker did not become healthy at {WORKER_URL}. "
        f"Check {stderr_path}."
    )


def _stop_worker() -> None:
    if WORKER_PROCESS is None or WORKER_PROCESS.poll() is not None:
        _clear_pid_file()
        return
    WORKER_PROCESS.terminate()
    try:
        WORKER_PROCESS.wait(timeout=5)
    except subprocess.TimeoutExpired:
        WORKER_PROCESS.kill()
    _clear_pid_file()


@mcp.tool()
def deep_research(query: str, file_paths: list[str] | None = None) -> dict[str, Any]:
    """Search, advise, or deep-research with Vertex Gemini 3.5 Flash (Google Search grounding) and optional local/GCS file inputs."""
    if not isinstance(query, str) or not query.strip():
        return {
            "ok": False,
            "error": {
                "type": "invalid_request",
                "message": "query must be a non-empty string.",
            },
        }

    try:
        return _http_json(
            "POST",
            "/deep_research",
            {"query": query, "file_paths": file_paths},
            timeout=WORKER_REQUEST_TIMEOUT_SECONDS,
        )
    except (TimeoutError, socket.timeout) as exc:
        return {
            "ok": False,
            "error": {
                "type": "worker_timeout",
                "message": (
                    "Gemini worker did not return before the MCP timeout. This is "
                    "usually Vertex quota pressure, a very large file prompt, or a "
                    "long code-review request."
                ),
                "details": str(exc),
            },
        }
    except urllib.error.URLError as exc:
        return {
            "ok": False,
            "error": {
                "type": "worker_unavailable",
                "message": f"Could not reach Gemini worker at {WORKER_URL}: {exc}",
            },
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
            },
        }


if __name__ == "__main__":
    atexit.register(_stop_worker)
    _start_worker()
    mcp.run(transport="stdio")
