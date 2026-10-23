"""CLI handler for running the Model Context Protocol (MCP) server."""
from __future__ import annotations

import argparse
import sys


def run_mcp_cli(args: argparse.Namespace) -> int:
    """Launch the CATML MCP server (stdio or streamable-http)."""
    workspace_path = getattr(args, "workspace", None)
    transport = getattr(args, "transport", "stdio") or "stdio"
    host = getattr(args, "host", "127.0.0.1") or "127.0.0.1"
    port = getattr(args, "port", 8000) or 8000
    path = getattr(args, "path", "/mcp") or "/mcp"
    token = getattr(args, "token", None)
    insecure = getattr(args, "insecure_no_auth", False)
    allow_mutations = bool(getattr(args, "allow_mutations", False))
    try:
        from automl.interfaces.mcp.server import HAS_MCP, run_mcp_service
        if not HAS_MCP:
            sys.stderr.write(
                "Error: El componente MCP requiere dependencias opcionales.\n"
                "Instálalas con: pip install '.[mcp]'\n"
            )
            return 1
        if allow_mutations:
            sys.stderr.write(
                "SECURITY NOTICE: --allow-mutations disables PENDING_APPROVAL for "
                "mutating MCP tools. Local/dev only.\n"
            )
        kwargs: dict[str, Any] = {
            "root_dir": workspace_path,
            "transport": transport,
            "host": host,
            "port": port,
            "streamable_http_path": path,
            "allow_mutations": allow_mutations,
        }
        if token is not None:
            kwargs["auth_token"] = token
        if insecure:
            kwargs["insecure_no_auth"] = insecure
        run_mcp_service(**kwargs)
        return 0
    except ImportError as e:
        sys.stderr.write(
            f"Error: Dependencia faltante para MCP ({e}).\n"
            "Instálala con: pip install '.[mcp]'\n"
        )
        return 1
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        sys.stderr.write(f"Error al iniciar servidor MCP: {e}\n")
        return 1
