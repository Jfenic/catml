from __future__ import annotations

import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger("catml.web")

from automl import __version__
from automl.application.bootstrap import build_application
from automl.interfaces.web.api import analysis, datasets, experiments, jobs
from automl.interfaces.web.api._helpers import (  # noqa: F401 — public re-exports for tests
    _build_real_activity_feed,
    _effective_experiment_status,
    _get_best_trial_result,
    _is_metric_minimizing,
    _status_value,
)


def _is_loopback_origin(origin: str) -> bool:
    """Return True when Origin is an http(s) loopback URL."""
    try:
        parsed = urlparse(origin)
    except Exception:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}


class AutoMLWebHandler(BaseHTTPRequestHandler):
    """
    HTTP request handler for the CATML interactive web dashboard / workbench.
    Implements Hexagonal Interface Parity by dispatching exclusively
    through CommandBus, QueryBus, and AutoMLWorkspace.
    """

    workspace_dir: str = ".automl/default"
    static_dir: Path = Path(__file__).parent / "static"
    auth_token: str | None = None
    require_auth: bool = False
    MAX_PAYLOAD_SIZE: int = 50 * 1024 * 1024  # 50 MB

    def _is_authenticated(self) -> bool:
        if not self.require_auth or not self.auth_token:
            return True

        auth_header = self.headers.get("Authorization") if hasattr(self, "headers") and self.headers else None
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            if token == self.auth_token:
                return True

        parsed = urlparse(self.path)
        query_params = parse_qs(parsed.query)
        token_list = query_params.get("token")
        if token_list:
            method = getattr(self, "command", "GET")
            if method != "GET":
                logger.warning("Rejected query token on state-mutating HTTP %s request to %s", method, parsed.path)
                return False
            if token_list[0].strip() == self.auth_token:
                logger.warning(
                    "Insecure authentication via URL query string '?token=' on %s. "
                    "Use 'Authorization: Bearer <token>' header or URL hash fragment '#token=' instead.",
                    parsed.path,
                )
                return True

        return False

    def _require_auth_or_reject(self) -> bool:
        if self._is_authenticated():
            return True
        self._send_json(
            {
                "error": "Unauthorized",
                "message": (
                    "Authentication token required for remote workbench access. "
                    "Provide via 'Authorization: Bearer <token>' header."
                ),
            },
            status=HTTPStatus.UNAUTHORIZED,
        )
        return False

    def _send_cors_headers(self) -> None:
        origin = self.headers.get("Origin") if hasattr(self, "headers") and self.headers else None
        if origin and _is_loopback_origin(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _mutable_request_allowed(self) -> bool:
        """Defend against casual cross-origin POSTs to the local Workbench.

        Policy (defense in depth; browsers still control real CSRF surface):
        - Require ``Content-Type: application/json`` (blocks simple form CSRF).
        - Reject ``Sec-Fetch-Site: cross-site`` when the browser sends it.
        - If ``Origin`` is present, require loopback or same Host as the request.
        Missing Origin / Sec-Fetch-Site (CLI, urlopen, older clients) is allowed.
        """
        headers: Any = self.headers if hasattr(self, "headers") and self.headers else {}
        content_type = (headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            self._send_json(
                {
                    "error": "Unsupported Media Type",
                    "message": "Mutable Workbench endpoints require Content-Type: application/json",
                },
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            )
            return False

        sec_fetch_site = (headers.get("Sec-Fetch-Site") or "").strip().lower()
        if sec_fetch_site == "cross-site":
            self._send_json(
                {
                    "error": "Forbidden",
                    "message": "Cross-site mutable requests are not allowed",
                },
                HTTPStatus.FORBIDDEN,
            )
            return False

        origin = (headers.get("Origin") or "").strip()
        if origin:
            host = (headers.get("Host") or "").strip().lower()
            origin_host = (urlparse(origin).netloc or "").lower()
            if not (_is_loopback_origin(origin) or (host and origin_host == host)):
                self._send_json(
                    {
                        "error": "Forbidden",
                        "message": "Origin is not allowed for mutable Workbench requests",
                    },
                    HTTPStatus.FORBIDDEN,
                )
                return False
        return True

    def _send_json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(payload)

    def _send_html(self, html_content: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = html_content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.end_headers()

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        if not self._require_auth_or_reject():
            return

        parsed = urlparse(self.path)
        path = parsed.path
        query_params = parse_qs(parsed.query)

        # Static file serving (SPA frontend, modular JS, and CSS)
        if path in {"/", "/index.html"}:
            index_file = self.static_dir / "index.html"
            if index_file.exists():
                self._send_html(index_file.read_text(encoding="utf-8"))
            else:
                self._send_json({"error": "Dashboard template not found"}, HTTPStatus.NOT_FOUND)
            return

        if path.startswith("/static/"):
            rel_path = path[len("/static/"):].lstrip("/")
            static_root = self.static_dir.resolve()
            target_file = (static_root / rel_path).resolve()
            if not target_file.is_relative_to(static_root):
                self._send_json({"error": "Access denied: path outside static directory"}, HTTPStatus.FORBIDDEN)
                return
            if target_file.is_file():
                mime_types = {
                    ".html": "text/html; charset=utf-8",
                    ".js": "application/javascript; charset=utf-8",
                    ".css": "text/css; charset=utf-8",
                    ".json": "application/json; charset=utf-8",
                    ".svg": "image/svg+xml",
                    ".png": "image/png",
                    ".woff": "font/woff",
                    ".woff2": "font/woff2",
                    ".ttf": "font/ttf",
                    ".otf": "font/otf",
                }
                content_type = mime_types.get(target_file.suffix.lower(), "application/octet-stream")
                payload = target_file.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(payload)
                return
            self._send_json({"error": f"File {rel_path} not found"}, HTTPStatus.NOT_FOUND)
            return

        ws, cmd, qry = build_application(root_dir=self.workspace_dir)

        if jobs.try_get(self, path, query_params, ws, cmd, qry):
            return
        if experiments.try_get(self, path, query_params, ws, cmd, qry):
            return
        if datasets.try_get(self, path, query_params, ws, cmd, qry):
            return
        if analysis.try_get(self, path, query_params, ws, cmd, qry):
            return

        self._send_json({"error": "Not Found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if not self._require_auth_or_reject():
            return
        if not self._mutable_request_allowed():
            return

        parsed = urlparse(self.path)
        path = parsed.path

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > self.MAX_PAYLOAD_SIZE:
                self._send_json(
                    {
                        "error": "Payload Too Large",
                        "message": f"Request size exceeds limit of {self.MAX_PAYLOAD_SIZE // (1024 * 1024)}MB",
                    },
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                )
                return
            body = self.rfile.read(content_length)
            if len(body) > self.MAX_PAYLOAD_SIZE:
                self._send_json(
                    {
                        "error": "Payload Too Large",
                        "message": f"Request size exceeds limit of {self.MAX_PAYLOAD_SIZE // (1024 * 1024)}MB",
                    },
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                )
                return
            payload = json.loads(body.decode("utf-8")) if body else {}
        except Exception as e:
            self._send_json({"error": f"Invalid JSON payload: {e}"}, HTTPStatus.BAD_REQUEST)
            return

        ws, cmd, qry = build_application(root_dir=self.workspace_dir)

        try:
            if jobs.try_post(self, path, payload, ws, cmd, qry):
                return
            if experiments.try_post(self, path, payload, ws, cmd, qry):
                return
            if datasets.try_post(self, path, payload, ws, cmd, qry):
                return
            if analysis.try_post(self, path, payload, ws, cmd, qry):
                return

            self._send_json({"error": "Endpoint not found"}, HTTPStatus.NOT_FOUND)

        except (ValueError, TypeError) as exc:
            import traceback
            traceback.print_exc()
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except KeyError as exc:
            import traceback
            traceback.print_exc()
            self._send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._send_json({"status": "error", "message": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def run_web_dashboard(
    port: int = 8080,
    workspace_dir: str = ".automl/default",
    host: str = "127.0.0.1",
    auth_token: str | None = None,
    insecure_no_auth: bool = False,
) -> None:
    import secrets

    is_local = host in {"127.0.0.1", "localhost", "::1"}
    effective_token = auth_token

    if not is_local and not insecure_no_auth:
        AutoMLWebHandler.require_auth = True
        if not effective_token:
            effective_token = secrets.token_urlsafe(16)
        AutoMLWebHandler.auth_token = effective_token
    elif not is_local and insecure_no_auth:
        import os
        if os.environ.get("CATML_ALLOW_INSECURE") != "1":
            raise PermissionError(
                f"Refusing to expose Workbench on external interface '{host}' without authentication. "
                "Provide an authentication token or set environment variable CATML_ALLOW_INSECURE=1 to override."
            )
        AutoMLWebHandler.require_auth = False
        AutoMLWebHandler.auth_token = None
    else:
        AutoMLWebHandler.require_auth = False
        AutoMLWebHandler.auth_token = None

    AutoMLWebHandler.workspace_dir = workspace_dir
    server = ThreadingHTTPServer((host, port), AutoMLWebHandler)
    from automl.infrastructure.jobs.worker import JobWorker
    worker = JobWorker(workspace_dir)
    try:
        worker.start()
    except Exception:
        server.server_close()
        raise

    print("=" * 65)
    print(f"  CATML AutoML Workbench (Platform V{__version__})")
    if not is_local and not insecure_no_auth:
        print(f"  Bound to external interface: http://{host}:{port}")
        print("  SECURITY NOTICE: Remote binding protected with authentication token.")
        print(f"  Access token: {effective_token}")
        print(f"  Direct secure URL (fragment, not sent over HTTP): http://{host}:{port}/#token={effective_token}")
    elif not is_local and insecure_no_auth:
        print(f"  Bound to external interface: http://{host}:{port}")
        print("  WARNING: Workbench is exposed on non-localhost interface WITHOUT authentication!")
        print("  Ensure this instance is guarded behind a secure reverse proxy or private VPN.")
    else:
        print(f"  Running locally at: http://{host}:{port}")
        print(f"  Connected Workspace: {workspace_dir}")
        print("  Hexagonal UI Adapter • CQRS • Observability & Control")
    print("  Press Ctrl+C to terminate.")
    print("=" * 65)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server...")
    finally:
        server.server_close()
        worker.close(timeout=2)


if __name__ == "__main__":
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    ws = sys.argv[2] if len(sys.argv) > 2 else ".automl/default"
    host = sys.argv[3] if len(sys.argv) > 3 else "127.0.0.1"
    token = sys.argv[4] if len(sys.argv) > 4 else None
    run_web_dashboard(port=port, workspace_dir=ws, host=host, auth_token=token)
