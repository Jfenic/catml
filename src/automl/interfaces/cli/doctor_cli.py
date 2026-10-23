"""System diagnostics and environment inspection CLI command (catml doctor).

Complies with Tech Minimalista Premium standards and provides both human-readable
diagnostic reporting and structured JSON output for AI agents and CI environments.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
from pathlib import Path

from automl.application.services.workspace import PLATFORM_VERSION


def _check_import(module_name: str) -> tuple[bool, str]:
    """Check if a module is importable and return (is_available, version_or_error)."""
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "installed")
        return True, str(version)
    except ImportError:
        return False, "not installed"
    except Exception as e:
        return False, f"error: {e}"


def run_doctor_diagnostics(workspace_dir: str | Path | None = None) -> dict:
    """Run full environment diagnostics and return structured dictionary."""
    results = {
        "platform_version": PLATFORM_VERSION,
        "python": {
            "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "compatible": sys.version_info >= (3, 10),
            "executable": sys.executable,
        },
        "system": {
            "os": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine(),
            "cpu_count": os.cpu_count() or 1,
        },
        "core_stack": {},
        "model_backends": {},
        "vision_stack": {},
        "agent_stack": {},
        "hardware": {},
        "workspace": {},
    }

    # Core Tabular ML Stack
    for pkg, name in [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("scipy", "SciPy"),
        ("sklearn", "Scikit-Learn"),
        ("optuna", "Optuna"),
    ]:
        ok, ver = _check_import(pkg)
        results["core_stack"][pkg] = {"name": name, "installed": ok, "version": ver}

    # Optional Models Extra: catml[models]
    for pkg, name in [
        ("lightgbm", "LightGBM"),
        ("xgboost", "XGBoost"),
        ("catboost", "CatBoost"),
    ]:
        ok, ver = _check_import(pkg)
        results["model_backends"][pkg] = {"name": name, "installed": ok, "version": ver}

    # Optional Vision Extra: catml[vision]
    for pkg, name in [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("timm", "Timm"),
        ("PIL", "Pillow"),
    ]:
        ok, ver = _check_import(pkg)
        results["vision_stack"][pkg] = {"name": name, "installed": ok, "version": ver}

    # Optional Agent/MCP Extra: catml[mcp,agents]
    for pkg, name in [
        ("mcp", "Model Context Protocol SDK"),
        ("langgraph", "LangGraph StateGraph"),
        ("httpx", "HTTPX Dual Transport"),
    ]:
        ok, ver = _check_import(pkg)
        results["agent_stack"][pkg] = {"name": name, "installed": ok, "version": ver}

    # Hardware & Acceleration
    has_cuda = False
    cuda_device = None
    has_mps = False
    if results["vision_stack"]["torch"]["installed"]:
        try:
            import torch
            has_cuda = torch.cuda.is_available()
            if has_cuda:
                cuda_device = torch.cuda.get_device_name(0)
            if hasattr(torch.backends, "mps"):
                has_mps = torch.backends.mps.is_available()
        except Exception:
            pass

    results["hardware"]["cuda_available"] = has_cuda
    results["hardware"]["cuda_device"] = cuda_device
    results["hardware"]["mps_available"] = has_mps

    # Workspace directory check
    ws_path = Path(workspace_dir or ".automl").resolve()
    ws_writable = False
    try:
        ws_path.mkdir(parents=True, exist_ok=True)
        test_file = ws_path / ".doctor_probe"
        test_file.touch()
        test_file.unlink()
        ws_writable = True
    except Exception:
        ws_writable = False

    results["workspace"] = {
        "path": str(ws_path),
        "exists": ws_path.exists(),
        "writable": ws_writable,
    }

    # Overall health score
    all_core_ok = all(item["installed"] for item in results["core_stack"].values())
    results["ready"] = results["python"]["compatible"] and all_core_ok and ws_writable
    return results


def run_doctor_cli(args: argparse.Namespace) -> int:
    """Execute catml doctor CLI command."""
    diag = run_doctor_diagnostics(workspace_dir=getattr(args, "workspace", None))

    if getattr(args, "json", False):
        print(json.dumps(diag, indent=2))
        return 0 if diag["ready"] else 1

    # Formatted Tech Minimalista Output
    print("╭─────────────────────────────────────────────────────────────────╮")
    print(f"│  🐈 CATML Environment & System Doctor (v{diag['platform_version']})          │")
    print("╰─────────────────────────────────────────────────────────────────╯")

    # Python & OS
    py = diag["python"]
    sys_info = diag["system"]
    py_badge = "[PASS]" if py["compatible"] else "[FAIL]"
    print(f"\n  Entorno Base:")
    print(f"  {py_badge} Python {py['version']} ({'compatible >= 3.10' if py['compatible'] else 'requiere >= 3.10'})")
    print(f"  [INFO] SO: {sys_info['os']} {sys_info['release']} ({sys_info['architecture']}, {sys_info['cpu_count']} CPUs)")

    # Core Stack
    print(f"\n  Núcleo Tabular ML (Base):")
    for pkg, info in diag["core_stack"].items():
        badge = "[PASS]" if info["installed"] else "[FAIL]"
        print(f"  {badge} {info['name']:14s} ({info['version']})")

    # Model Backends
    print(f"\n  Extensiones de Modelos (catml[models]):")
    for pkg, info in diag["model_backends"].items():
        if info["installed"]:
            print(f"  [PASS] {info['name']:14s} ({info['version']})")
        else:
            print(f"  [WARN] {info['name']:14s} no instalado (opcional: pip install \"catml[models]\")")

    # Agent Stack
    print(f"\n  Agentes & Protocolo MCP (catml[mcp,agents]):")
    for pkg, info in diag["agent_stack"].items():
        if info["installed"]:
            print(f"  [PASS] {info['name']:24s} ({info['version']})")
        else:
            print(f"  [WARN] {info['name']:24s} no instalado (opcional: pip install \"catml[mcp,agents]\")")

    # Vision Stack
    print(f"\n  Representaciones de Visión (catml[vision]):")
    for pkg, info in diag["vision_stack"].items():
        if info["installed"]:
            print(f"  [PASS] {info['name']:14s} ({info['version']})")
        else:
            print(f"  [WARN] {info['name']:14s} no instalado (opcional: pip install \"catml[vision]\")")

    # Hardware Acceleration
    hw = diag["hardware"]
    print(f"\n  Aceleración de Hardware:")
    if hw["cuda_available"]:
        print(f"  [PASS] CUDA disponible: {hw['cuda_device']}")
    elif hw["mps_available"]:
        print(f"  [PASS] Apple Silicon MPS aceleración disponible")
    else:
        print(f"  [INFO] Sin acelerador CUDA/MPS detectado (utilizando paralelización CPU)")

    # Workspace
    ws = diag["workspace"]
    ws_badge = "[PASS]" if ws["writable"] else "[FAIL]"
    print(f"\n  Espacio de Trabajo:")
    print(f"  {ws_badge} {ws['path']} (escritura {'verificada' if ws['writable'] else 'bloqueada'})")

    print("\n" + "─" * 65)
    if diag["ready"]:
        print("  ✓ CATML está listo para entrenar modelos tabulares y recibir agentes.")
    else:
        print("  ✗ Se detectaron incidencias críticas en el entorno base.")
    print("─" * 65 + "\n")

    return 0 if diag["ready"] else 1
