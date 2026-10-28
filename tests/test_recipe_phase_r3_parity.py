"""Tests for Phase R3: Surface Parity across Python API, CLI, MCP Agent Tools, and REST API.

Verifies that Experiment Recipes can be listed, created from verified evidence, inspected,
selected, submitted, promoted, and archived with identical IDs, configurations,
and lifecycle states across CLI, MCP tools, and REST API endpoints.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer
import pytest
import pandas as pd

from automl.application.analysis.commands import CreateStudyCommand
from automl.application.bootstrap import build_application
from automl.application.recipes.commands import (
    CreateExperimentRecipeFromEvidenceCommand,
)
from automl.application.recipes.queries import (
    GetExperimentRecipeQuery,
)
from automl.domain.analysis.models import (
    AnalysisHypothesis,
    AnalysisRun,
    EvidenceLink,
    StatisticalFinding,
)
from automl.domain.recipes.recipe import RecipeStatus
from automl.interfaces.cli.main import main
from automl.interfaces.mcp.server import HAS_MCP, create_mcp_server
from automl.interfaces.web.server import AutoMLWebHandler


@pytest.fixture
def recipe_env(tmp_path: Path):
    """Sets up an environment with dataset, study, hypothesis, experiments, and accepted evidence."""
    ws_dir = tmp_path / "workspace"
    ws, cmd_bus, qry_bus = build_application(root_dir=str(ws_dir))

    # 1. Dataset & Run
    df = pd.DataFrame({
        "num1": [1.0, 2.0, 3.0, 4.0, 100.0, 200.0],
        "num2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "target": [0, 0, 0, 1, 1, 1],
    })
    ds = ws.register_dataset("rec_parity_ds", df, target="target")
    run = ws.create_run(ds)

    # 2. Explore Study, Finding, and Hypothesis
    study_id = cmd_bus.dispatch(CreateStudyCommand(dataset_id=ds.id, name="Parity Study", target_column="target"))
    study_repo = ws.get_study_repository()

    analysis_run = AnalysisRun.create(study_id=study_id)
    study_repo.save_analysis_run(analysis_run)

    finding = StatisticalFinding.create(
        study_id=study_id,
        analysis_run_id=analysis_run.id,
        finding_type="skewed_distribution",
        summary="num1 is highly skewed",
        column_name="num1",
    )
    study_repo.save_finding(finding)

    hypothesis = AnalysisHypothesis.create(
        study_id=study_id,
        finding_id=finding.id,
        description="Apply yeo_johnson to num1",
        proposed_action="power_transform",
        experiment_delta={
            "column": "num1",
            "method": "yeo_johnson",
            "transform_kind": "yeo_johnson",
            "suggested_action": "power_transform",
        },
    )
    study_repo.save_hypothesis(hypothesis)

    # 3. Baseline and Candidate Experiments
    base_exp = ws.create_experiment(
        run=run,
        name="Baseline",
        model_ids=["random_forest"],
        feature_names=["num1", "num2"],
        validation_strategy="holdout",
    )
    ws.run_experiment(run, base_exp)

    cand_exp = ws.create_experiment(
        run=run,
        name="Candidate",
        model_ids=["random_forest"],
        feature_names=["num1", "num2"],
        validation_strategy="holdout",
        column_transforms=[{"column": "num1", "kind": "yeo_johnson", "params": {}}],
    )
    ws.run_experiment(run, cand_exp)

    # 4. Accepted EvidenceLink
    evidence = EvidenceLink.create(
        hypothesis_id=hypothesis.id,
        experiment_id=cand_exp.id,
        baseline_score=0.80,
        candidate_score=0.92,
        metric="accuracy",
        accepted=True,
        notes="Verified empirical improvement +0.12",
    )
    study_repo.save_evidence_link(evidence)

    # Rejected EvidenceLink (for testing filtering)
    rejected_evidence = EvidenceLink.create(
        hypothesis_id=hypothesis.id,
        experiment_id=base_exp.id,
        baseline_score=0.80,
        candidate_score=0.75,
        metric="accuracy",
        accepted=False,
        notes="Regression -0.05",
    )
    study_repo.save_evidence_link(rejected_evidence)

    return {
        "workspace_dir": str(ws_dir),
        "ws": ws,
        "cmd_bus": cmd_bus,
        "qry_bus": qry_bus,
        "dataset": ds,
        "run": run,
        "study_id": study_id,
        "evidence": evidence,
        "rejected_evidence": rejected_evidence,
    }


@pytest.fixture
def running_web_server(recipe_env):
    """Runs a local AutoML Web server against the recipe test environment."""
    ws_dir = recipe_env["workspace_dir"]
    AutoMLWebHandler.workspace_dir = ws_dir

    server = ThreadingHTTPServer(("127.0.0.1", 0), AutoMLWebHandler)
    host, port = server.server_address

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://{host}:{port}"
    yield {**recipe_env, "base_url": base_url}

    server.shutdown()
    server.server_close()
    thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# 1. CLI Surface Tests
# ---------------------------------------------------------------------------


def test_recipe_cli_surface(recipe_env, capsys):
    ws_dir = recipe_env["workspace_dir"]
    evidence = recipe_env["evidence"]

    # 1. list-verified (JSON)
    rc = main(["recipe", "list-verified", "--workspace", ws_dir, "--json"])
    assert rc == 0
    captured = capsys.readouterr()
    verified_list = json.loads(captured.out)
    assert len(verified_list) == 1
    assert verified_list[0]["evidence_link_id"] == evidence.id
    assert verified_list[0]["accepted"] is True

    # list-verified with --all includes rejected
    rc = main(["recipe", "list-verified", "--workspace", ws_dir, "--all", "--json"])
    assert rc == 0
    all_verified = json.loads(capsys.readouterr().out)
    assert len(all_verified) == 2

    # list-verified human readable
    rc = main(["recipe", "list-verified", "--workspace", ws_dir])
    assert rc == 0
    text_out = capsys.readouterr().out
    assert "Found 1 verified improvement" in text_out
    assert evidence.id in text_out

    # 2. create-from-evidence (JSON)
    rc = main([
        "recipe", "create-from-evidence", evidence.id,
        "--name", "CLI Created Recipe",
        "--notes", "Created in CLI test",
        "--workspace", ws_dir,
        "--json",
    ])
    assert rc == 0
    created_out = json.loads(capsys.readouterr().out)
    recipe_id = created_out["id"]
    assert recipe_id.startswith("rcp_")
    assert created_out["status"] == "draft"
    assert created_out["name"] == "CLI Created Recipe"
    assert created_out["column_transforms"] == [{"column": "num1", "kind": "yeo_johnson", "params": {}}]

    # 3. list (JSON & text)
    rc = main(["recipe", "list", "--workspace", ws_dir, "--json"])
    assert rc == 0
    listed_recipes = json.loads(capsys.readouterr().out)
    assert len(listed_recipes) == 1
    assert listed_recipes[0]["id"] == recipe_id

    rc = main(["recipe", "list", "--workspace", ws_dir])
    assert rc == 0
    assert recipe_id in capsys.readouterr().out

    # 4. show (JSON & text)
    rc = main(["recipe", "show", recipe_id, "--workspace", ws_dir, "--json"])
    assert rc == 0
    show_data = json.loads(capsys.readouterr().out)
    assert show_data["id"] == recipe_id
    assert show_data["status"] == "draft"

    rc = main(["recipe", "show", recipe_id, "--workspace", ws_dir])
    assert rc == 0
    assert f"Experiment Recipe: {recipe_id}" in capsys.readouterr().out

    # 5. select (JSON & text)
    rc = main(["recipe", "select", recipe_id, "--notes", "Selecting via CLI", "--workspace", ws_dir, "--json"])
    assert rc == 0
    select_data = json.loads(capsys.readouterr().out)
    assert select_data["status"] == "selected"

    # 6. submit (JSON)
    rc = main(["recipe", "submit", recipe_id, "--workspace", ws_dir, "--json"])
    assert rc == 0
    submit_data = json.loads(capsys.readouterr().out)
    assert submit_data["recipe_id"] == recipe_id
    assert len(submit_data["experiment_ids"]) > 0

    # 7. promote (JSON & text)
    rc = main(["recipe", "promote", recipe_id, "--reason", "Proven CV gain", "--workspace", ws_dir, "--json"])
    assert rc == 0
    promote_data = json.loads(capsys.readouterr().out)
    assert promote_data["status"] == "promoted"
    assert promote_data["lineage"]["promoted_reason"] == "Proven CV gain"

    # 8. archive (JSON & text)
    rc = main(["recipe", "archive", recipe_id, "--workspace", ws_dir, "--json"])
    assert rc == 0
    archive_data = json.loads(capsys.readouterr().out)
    assert archive_data["status"] == "archived"

    # Verify final status via show
    rc = main(["recipe", "show", recipe_id, "--workspace", ws_dir, "--json"])
    assert rc == 0
    final_data = json.loads(capsys.readouterr().out)
    assert final_data["status"] == "archived"

    # Test error cases: non-existent recipe show returns 1
    rc = main(["recipe", "show", "rcp_nonexistent", "--workspace", ws_dir])
    assert rc == 1


# ---------------------------------------------------------------------------
# 2. MCP Agent Tools Surface Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MCP, reason="mcp extra required")
def test_recipe_mcp_surface(recipe_env):
    ws = recipe_env["ws"]
    cmd_bus = recipe_env["cmd_bus"]
    qry_bus = recipe_env["qry_bus"]
    evidence = recipe_env["evidence"]
    dataset = recipe_env["dataset"]

    server = create_mcp_server(
        workspace=ws,
        command_bus=cmd_bus,
        query_bus=qry_bus,
        require_approval_for_mutations=False,
    )

    # Verify tool listing includes all 8 recipe tools
    tools = asyncio.run(server.list_tools())
    tool_names = {t.name for t in tools}
    expected_tools = {
        "recipe_list_verified_improvements",
        "recipe_create_from_evidence",
        "recipe_get",
        "recipe_list",
        "recipe_select",
        "recipe_submit",
        "recipe_promote",
        "recipe_archive",
    }
    assert expected_tools.issubset(tool_names)

    # 1. recipe_list_verified_improvements
    res = asyncio.run(server.call_tool("recipe_list_verified_improvements", {"dataset_id": dataset.id}))
    assert res.is_error is False
    verified_data = json.loads(res.content[0].text)
    assert verified_data["total"] >= 1
    assert any(i["evidence_link_id"] == evidence.id for i in verified_data["items"])

    # 2. recipe_create_from_evidence
    res = asyncio.run(
        server.call_tool(
            "recipe_create_from_evidence",
            {
                "evidence_link_id": evidence.id,
                "recipe_name": "MCP Created Recipe",
                "notes": "Created via MCP agent",
            },
        )
    )
    assert res.is_error is False
    created_recipe = json.loads(res.content[0].text)
    recipe_id = created_recipe["id"]
    assert recipe_id.startswith("rcp_")
    assert created_recipe["status"] == "draft"
    assert created_recipe["name"] == "MCP Created Recipe"

    # 3. recipe_get
    res = asyncio.run(server.call_tool("recipe_get", {"recipe_id": recipe_id}))
    assert res.is_error is False
    get_data = json.loads(res.content[0].text)
    assert get_data["id"] == recipe_id
    assert get_data["status"] == "draft"

    # recipe_get with unknown ID returns error
    res_err = asyncio.run(server.call_tool("recipe_get", {"recipe_id": "rcp_unknown"}))
    assert res_err.is_error is True
    assert "NOT_FOUND" in res_err.content[0].text

    # 4. recipe_list
    res = asyncio.run(server.call_tool("recipe_list", {"status": "draft"}))
    assert res.is_error is False
    list_data = json.loads(res.content[0].text)
    assert list_data["total"] >= 1
    assert any(r["id"] == recipe_id for r in list_data["items"])

    # 5. recipe_select
    res = asyncio.run(server.call_tool("recipe_select", {"recipe_id": recipe_id, "notes": "Selecting via MCP"}))
    assert res.is_error is False
    select_data = json.loads(res.content[0].text)
    assert select_data["status"] == "selected"

    # 6. recipe_submit
    res = asyncio.run(server.call_tool("recipe_submit", {"recipe_id": recipe_id}))
    assert res.is_error is False
    submit_data = json.loads(res.content[0].text)
    assert submit_data["status"] == "submitted"
    assert len(submit_data["executed_experiments"]) > 0

    # 7. recipe_promote
    res = asyncio.run(server.call_tool("recipe_promote", {"recipe_id": recipe_id, "reason": "Promoted via MCP"}))
    assert res.is_error is False
    promote_data = json.loads(res.content[0].text)
    assert promote_data["status"] == "promoted"
    assert promote_data["lineage"]["promoted_reason"] == "Promoted via MCP"

    # 8. recipe_archive
    res = asyncio.run(server.call_tool("recipe_archive", {"recipe_id": recipe_id}))
    assert res.is_error is False
    archive_data = json.loads(res.content[0].text)
    assert archive_data["status"] == "archived"


# ---------------------------------------------------------------------------
# 3. REST API Surface Tests
# ---------------------------------------------------------------------------


def test_recipe_rest_surface(running_web_server):
    base = running_web_server["base_url"]
    evidence = running_web_server["evidence"]
    dataset = running_web_server["dataset"]

    # 1. GET /api/recipes/verified-improvements
    with urlopen(f"{base}/api/recipes/verified-improvements?dataset_id={dataset.id}") as resp:
        assert resp.status == 200
        improvements = json.loads(resp.read().decode("utf-8"))
        assert len(improvements) == 1
        assert improvements[0]["evidence_link_id"] == evidence.id
        assert improvements[0]["accepted"] is True

    # 2. POST /api/recipes/create-from-evidence
    create_payload = json.dumps({
        "evidence_link_id": evidence.id,
        "name": "REST Created Recipe",
        "notes": "Created via REST endpoint",
    }).encode("utf-8")
    req = Request(f"{base}/api/recipes/create-from-evidence", data=create_payload, headers={"Content-Type": "application/json"})
    with urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "success"
        recipe_id = data["recipe_id"]
        assert recipe_id.startswith("rcp_")
        assert data["recipe"]["status"] == "draft"

    # POST missing evidence_link_id returns 400
    bad_req = Request(f"{base}/api/recipes/create-from-evidence", data=b"{}", headers={"Content-Type": "application/json"})
    with pytest.raises(HTTPError) as exc_info:
        urlopen(bad_req)
    assert exc_info.value.code == 400

    # 3. GET /api/recipes
    with urlopen(f"{base}/api/recipes") as resp:
        assert resp.status == 200
        recipes = json.loads(resp.read().decode("utf-8"))
        assert len(recipes) >= 1
        assert any(r["id"] == recipe_id for r in recipes)

    # 4. GET /api/recipes/{recipe_id}
    with urlopen(f"{base}/api/recipes/{recipe_id}") as resp:
        assert resp.status == 200
        single_recipe = json.loads(resp.read().decode("utf-8"))
        assert single_recipe["id"] == recipe_id
        assert single_recipe["status"] == "draft"
        assert single_recipe["name"] == "REST Created Recipe"

    # GET non-existent recipe returns 404
    with pytest.raises(HTTPError) as exc_info:
        urlopen(f"{base}/api/recipes/rcp_unknown_id")
    assert exc_info.value.code == 404

    # 5. POST /api/recipes/select
    select_payload = json.dumps({"recipe_id": recipe_id, "notes": "Selected via REST"}).encode("utf-8")
    req = Request(f"{base}/api/recipes/select", data=select_payload, headers={"Content-Type": "application/json"})
    with urlopen(req) as resp:
        assert resp.status == 200
        sel_data = json.loads(resp.read().decode("utf-8"))
        assert sel_data["status"] == "success"
        assert sel_data["recipe"]["status"] == "selected"

    # 6. POST /api/recipes/submit
    submit_payload = json.dumps({"recipe_id": recipe_id, "notes": "Submitted via REST"}).encode("utf-8")
    req = Request(f"{base}/api/recipes/submit", data=submit_payload, headers={"Content-Type": "application/json"})
    with urlopen(req) as resp:
        assert resp.status == 200
        sub_data = json.loads(resp.read().decode("utf-8"))
        assert sub_data["status"] == "submitted"
        assert len(sub_data["experiment_ids"]) > 0
        assert sub_data["recipe"]["status"] == "submitted"

    # 7. POST /api/recipes/promote
    promote_payload = json.dumps({"recipe_id": recipe_id, "reason": "Promoted via REST"}).encode("utf-8")
    req = Request(f"{base}/api/recipes/promote", data=promote_payload, headers={"Content-Type": "application/json"})
    with urlopen(req) as resp:
        assert resp.status == 200
        pro_data = json.loads(resp.read().decode("utf-8"))
        assert pro_data["status"] == "promoted"
        assert pro_data["recipe"]["status"] == "promoted"
        assert pro_data["recipe"]["lineage"]["promoted_reason"] == "Promoted via REST"

    # 8. POST /api/recipes/archive
    archive_payload = json.dumps({"recipe_id": recipe_id}).encode("utf-8")
    req = Request(f"{base}/api/recipes/archive", data=archive_payload, headers={"Content-Type": "application/json"})
    with urlopen(req) as resp:
        assert resp.status == 200
        arc_data = json.loads(resp.read().decode("utf-8"))
        assert arc_data["status"] == "archived"


# ---------------------------------------------------------------------------
# 4. Cross-Surface Interoperability & Exact Parity
# ---------------------------------------------------------------------------


def test_recipe_cross_surface_full_parity(running_web_server, capsys):
    """Proves end-to-end parity:

    1. Inspect evidence via CLI.
    2. Create recipe via CLI -> status is DRAFT.
    3. Read recipe via MCP tool -> exact same ID, transforms, and DRAFT status.
    4. Select recipe via MCP tool -> status becomes SELECTED.
    5. Submit and run experiment via REST API -> status becomes SUBMITTED.
    6. Promote recipe via CLI with audit reason -> status becomes PROMOTED.
    7. Query recipe via REST API, CLI, and MCP -> all 3 report identical PROMOTED status,
       identical lineage reason, and identical executed experiment IDs.
    """
    ws_dir = running_web_server["workspace_dir"]
    base_url = running_web_server["base_url"]
    ws = running_web_server["ws"]
    cmd_bus = running_web_server["cmd_bus"]
    qry_bus = running_web_server["qry_bus"]
    evidence = running_web_server["evidence"]

    # --- Step 1 & 2: CLI Create ---
    rc = main([
        "recipe", "create-from-evidence", evidence.id,
        "--name", "Cross-Surface Parity Recipe",
        "--notes", "Origin: CLI",
        "--workspace", ws_dir,
        "--json",
    ])
    assert rc == 0
    cli_created = json.loads(capsys.readouterr().out)
    recipe_id = cli_created["id"]
    assert cli_created["status"] == "draft"

    # --- Step 3: MCP Read ---
    if HAS_MCP:
        server = create_mcp_server(
        workspace=ws,
        command_bus=cmd_bus,
        query_bus=qry_bus,
        require_approval_for_mutations=False,
    )
        mcp_res = asyncio.run(server.call_tool("recipe_get", {"recipe_id": recipe_id}))
        assert mcp_res.is_error is False
        mcp_recipe = json.loads(mcp_res.content[0].text)
        assert mcp_recipe["id"] == recipe_id
        assert mcp_recipe["name"] == "Cross-Surface Parity Recipe"
        assert mcp_recipe["status"] == "draft"
        assert mcp_recipe["column_transforms"] == cli_created["column_transforms"]

        # --- Step 4: MCP Select ---
        sel_res = asyncio.run(server.call_tool("recipe_select", {"recipe_id": recipe_id, "notes": "Selected via MCP Agent"}))
        assert sel_res.is_error is False
        selected_mcp = json.loads(sel_res.content[0].text)
        assert selected_mcp["status"] == "selected"

    # --- Step 5: REST Submit ---
    sub_payload = json.dumps({"recipe_id": recipe_id, "notes": "Submitted via REST API"}).encode("utf-8")
    req = Request(f"{base_url}/api/recipes/submit", data=sub_payload, headers={"Content-Type": "application/json"})
    with urlopen(req) as resp:
        assert resp.status == 200
        rest_submit = json.loads(resp.read().decode("utf-8"))
        assert rest_submit["status"] == "submitted"
        assert len(rest_submit["experiment_ids"]) > 0
        executed_exp_id = rest_submit["experiment_ids"][0]

    # --- Step 6: CLI Promote ---
    rc = main([
        "recipe", "promote", recipe_id,
        "--reason", "Passed cross-surface test suite",
        "--workspace", ws_dir,
        "--json",
    ])
    assert rc == 0
    cli_promote = json.loads(capsys.readouterr().out)
    assert cli_promote["status"] == "promoted"

    # --- Step 7: Parity Verification across all surfaces ---
    # Surface A: REST GET
    with urlopen(f"{base_url}/api/recipes/{recipe_id}") as resp:
        rest_final = json.loads(resp.read().decode("utf-8"))

    # Surface B: CLI Show
    rc = main(["recipe", "show", recipe_id, "--workspace", ws_dir, "--json"])
    assert rc == 0
    cli_final = json.loads(capsys.readouterr().out)

    # Assert exact equivalence
    assert rest_final["id"] == cli_final["id"] == recipe_id
    assert rest_final["status"] == cli_final["status"] == "promoted"
    assert rest_final["dataset_id"] == cli_final["dataset_id"]
    assert rest_final["lineage"]["promoted_reason"] == cli_final["lineage"]["promoted_reason"] == "Passed cross-surface test suite"
    assert executed_exp_id in rest_final["experiment_ids"]
    assert executed_exp_id in cli_final["experiment_ids"]

    # Surface C: MCP Get (if MCP available)
    if HAS_MCP:
        mcp_final_res = asyncio.run(server.call_tool("recipe_get", {"recipe_id": recipe_id}))
        assert mcp_final_res.is_error is False
        mcp_final = json.loads(mcp_final_res.content[0].text)
        assert mcp_final["id"] == recipe_id
        assert mcp_final["status"] == "promoted"
        assert mcp_final["lineage"]["promoted_reason"] == "Passed cross-surface test suite"
        assert executed_exp_id in mcp_final["experiment_ids"]
