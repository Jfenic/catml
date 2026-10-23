from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import (
    CreateExperimentCommand,
    ExcludeFeatureCommand,
    ExcludeModelCommand,
    OptimizeExperimentCommand,
    PlanExperimentsCommand,
    PrioritizeFeatureCommand,
    RunExperimentCommand,
    RunScheduledExperimentsCommand,
    SelectFeaturesCommand,
    PlanAblationExperimentsCommand,
    GenerateSubmissionCommand,
    GenerateOOFSubmissionCommand,
    GenerateTemporalFeaturesCommand,
)
from automl.application.queries.workspace_queries import (
    GetDatasetProfileQuery,
    GetFeatureRankingQuery,
    GetLeaderboardQuery,
    GetTaskPlanQuery,
    ListCandidatesQuery,
    ListTaskTypesQuery,
    ListPluginsQuery,
    GetOOFResultQuery,
    DetectTemporalStructureQuery,
    GetMetaKnowledgeQuery,
)
from automl.domain.features.selection_strategy import FeatureSelectionStrategy
from automl.application.services.workspace import PLATFORM_VERSION
from automl.benchmarks.runner import BenchmarkRunner



def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_dataset() -> Path:
    return _project_root() / "examples" / "data" / "customers_churn.csv"


def _print_table(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for row in rows:
        delta = row.get("delta_vs_baseline")
        delta_str = f"  Δ={delta:+.4f}" if delta is not None else ""
        print(
            f"  {row['scenario_id']:28s}  {row['metric']}={row['best_score']:.4f}  "
            f"model={row.get('best_model') or '-':22s}  "
            f"{row['total_time_s']:.2f}s{delta_str}"
        )


def fit_cli(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset '{dataset_path}' does not exist.", file=sys.stderr)
        return 1

    from automl.engine.profiling.dataset_profiler import load_dataframe

    try:
        df = load_dataframe(dataset_path)
    except Exception as exc:
        print(f"Error loading dataset: {exc}", file=sys.stderr)
        return 1

    if args.target not in df.columns:
        print(f"Error: Target column '{args.target}' not found in dataset.", file=sys.stderr)
        return 1

    models = [m.strip() for m in args.models.split(",")] if args.models else None

    from automl.facade import AutoML

    automl = AutoML(
        task=args.task,
        models=models,
        workspace_dir=args.workspace,
    )

    if not getattr(args, "json", False):
        print("=" * 65)
        print(f"  CATML Automated Machine Learning (Platform V{PLATFORM_VERSION})")
        print("=" * 65)
        print(f"  Dataset:     {dataset_path}")
        print(f"  Target:      {args.target}")
        if models:
            print(f"  Candidates:  {', '.join(models)}")
        print("\n  Evaluating candidate models and tuning...")

    try:
        result = automl.fit(df, target=args.target)
    except Exception as exc:
        print(f"Error during training: {exc}", file=sys.stderr)
        return 1

    # Save artifact
    output_path = Path(args.output_model)
    result.save_model(output_path)

    if getattr(args, "json", False):
        lb = result.leaderboard().to_dict(orient="records")
        payload = {
            "task_type": result.task_type,
            "best_model_id": result.best_model_id,
            "best_score": result.best_score,
            "metric": result.metric,
            "artifact_path": str(output_path),
            "leaderboard": lb,
        }
        print(json.dumps(payload, indent=2))
        return 0

    lb = result.leaderboard()
    print("\n  Leaderboard:")
    print("  " + "-" * 60)
    print(f"  {'Rank':<6}{'Model':<24}{'Metric':<12}{'Score':<10}{'Time (s)':<10}")
    print("  " + "-" * 60)
    for _, row in lb.iterrows():
        print(
            f"  {int(row.get('rank', 0)):<6}{str(row.get('model_id', '')):<24}"
            f"{str(row.get('metric', '')):<12}{float(row.get('score', 0.0)):<10.4f}"
            f"{float(row.get('training_time_s', 0.0)):<10.2f}"
        )
    print("  " + "-" * 60)
    print(f"\n  ✓ Best Model:   {result.best_model_id} ({result.metric}: {result.best_score:.4f})")
    print(f"  ✓ Model Saved:  {output_path.resolve()}")
    print(f"  ✓ Run in Prod:  ModelArtifact.load(\"{output_path}\")")
    print("=" * 65)
    return 0


def run_demo(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset) if args.dataset else _default_dataset()
    workspace_dir = Path(args.workspace) if args.workspace else _project_root() / ".automl" / "demo"

    ws, cmd, qry = build_application(root_dir=str(workspace_dir))
    dataset = ws.register_dataset(
        name="customers_churn",
        path=dataset_path,
        target="churn",
        task_type="binary_classification",
    )
    profile = qry.dispatch(GetDatasetProfileQuery(dataset.id))
    print(f"Workspace: {workspace_dir}")
    print(f"Dataset:   {dataset_path}")
    print(f"Profile:   {profile.row_count} rows, {profile.column_count} columns")

    run = ws.create_run(dataset, metric="roc_auc")
    cmd.dispatch(ExcludeFeatureCommand(dataset.id, "customer_id", run_id=run.id))
    cmd.dispatch(PrioritizeFeatureCommand(dataset.id, "salary", run_id=run.id))
    cmd.dispatch(PrioritizeFeatureCommand(dataset.id, "debt", run_id=run.id))
    cmd.dispatch(ExcludeModelCommand(run.id, "svc"))

    if getattr(args, "auto", False):
        print(f"Run:         {run.id}")
        print("\nPlanning experiments automatically via RuleBasedExperimentPlanner...\n")
        candidates = cmd.dispatch(PlanExperimentsCommand(run_id=run.id))
        for cand in candidates:
            prio_info = (
                f"effective={cand.priority.effective_score:.2f} [{cand.priority.level.value}]"
                if cand.priority
                else ""
            )
            print(f"  Candidate: {cand.name:28s} models={','.join(cand.model_ids):24s} {prio_info}")
        print("\nRunning scheduled experiments via Priority Scheduler...\n")
        cmd.dispatch(RunScheduledExperimentsCommand(run_id=run.id, max_experiments=3))
    else:
        features = ws.get_feature_registry(dataset.id).active_feature_names(dataset.target_column)
        experiment = cmd.dispatch(
            CreateExperimentCommand(
                run_id=run.id,
                name="financial_baseline",
                feature_names=features,
                model_ids=["logistic_regression", "random_forest"],
                hypothesis="Salary and debt features predict churn",
                priority="high",
            )
        )

        print(f"Run:         {run.id}")
        print(f"Experiment:  {experiment.id}")
        print("\nRunning trials...\n")
        results = cmd.dispatch(RunExperimentCommand(run.id, experiment.id))
        for result in results:
            status = "OK" if result.succeeded else f"FAIL ({result.failure_reason})"
            print(
                f"  {result.model_id:22s} {result.primary_metric}={result.primary_score:.4f}  "
                f"[{status}]  {result.training_time_seconds:.2f}s"
            )

    leaderboard = qry.dispatch(GetLeaderboardQuery(run.id))
    print("\nLeaderboard:")
    for row in leaderboard:
        print(f"  {row['model_id']:22s} {row['metric']}={row['score']:.4f}")

    if args.json:
        print(json.dumps(leaderboard, indent=2))
    return 0


def plan_experiments_cli(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset) if args.dataset else _default_dataset()
    workspace_dir = Path(args.workspace) if args.workspace else _project_root() / ".automl" / "demo"
    ws, cmd, qry = build_application(root_dir=str(workspace_dir))
    dataset = ws.register_dataset(
        name="plan_dataset",
        path=dataset_path,
        target=args.target,
    )
    run = ws.create_run(dataset)
    candidates = cmd.dispatch(PlanExperimentsCommand(run_id=run.id))
    print(f"\nPlanned {len(candidates)} experiment candidates for run {run.id}:\n")
    for c in candidates:
        prio = c.priority
        eff = f"{prio.effective_score:.2f} [{prio.level.value}]" if prio else "-"
        print(f"  {c.name:28s} models={','.join(c.model_ids):24s} prio={eff}")
        if prio and prio.breakdown.explanation:
            print(f"    Reasoning: {prio.breakdown.explanation}")

    if getattr(args, "auto_run", False):
        print("\nExecuting scheduled experiments...\n")
        results = cmd.dispatch(
            RunScheduledExperimentsCommand(
                run_id=run.id,
                max_experiments=args.max_experiments,
            )
        )
        for r in results:
            print(f"  Experiment {r['name']:24s} trials={r['trials_count']} best_score={r['best_score']:.4f}")
        leaderboard = qry.dispatch(GetLeaderboardQuery(run.id))
        print("\nLeaderboard:")
        for row in leaderboard:
            print(f"  {row['model_id']:22s} {row['metric']}={row['score']:.4f}")

    if args.json:
        queued = qry.dispatch(ListCandidatesQuery(run.id))
        print(json.dumps(queued, indent=2))
    return 0


def optimize_cli(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset) if args.dataset else _default_dataset()
    workspace_dir = Path(args.workspace) if args.workspace else _project_root() / ".automl" / "demo"
    ws, cmd, qry = build_application(root_dir=str(workspace_dir))

    if args.experiment_id and args.run_id:
        run_id = args.run_id
        experiment_id = args.experiment_id
    else:
        dataset = ws.register_dataset(
            name="optimize_dataset",
            path=dataset_path,
            target=args.target,
        )
        run = ws.create_run(dataset)
        run_id = run.id
        features = ws.get_feature_registry(dataset.id).active_feature_names(dataset.target_column)
        target_model = args.model or "logistic_regression"
        exp = cmd.dispatch(
            CreateExperimentCommand(
                run_id=run.id,
                name=f"opt_{target_model}",
                feature_names=features,
                model_ids=[target_model],
            )
        )
        experiment_id = exp.id

    json_mode = bool(getattr(args, "json", False))

    if not json_mode:
        print(f"\nStarting Hyperparameter Optimization (V{PLATFORM_VERSION})...")
        print(f"  Run ID:        {run_id}")
        print(f"  Experiment ID: {experiment_id}")
        print(f"  Optimizer:     {args.optimizer}")
        print(f"  Trials:        {args.trials}")
        if args.model:
            print(f"  Model:         {args.model}")

    result = cmd.dispatch(
        OptimizeExperimentCommand(
            run_id=run_id,
            experiment_id=experiment_id,
            model_id=args.model,
            optimizer=args.optimizer,
            n_trials=args.trials,
            timeout_seconds=args.timeout,
            patience=args.patience,
        )
    )

    if json_mode:
        print(json.dumps(result, indent=2))
        return 0

    print("\nOptimization Finished:")
    print(f"  Best Score:    {result['best_score']:.4f}")
    print(f"  Best Params:   {json.dumps(result['best_params'], indent=4)}")
    print(f"  Trials Run:    {result['trials_executed']}")
    print("\nTrial History:")
    for t in result["trials"]:
        print(
            f"  Trial {t['trial_id'][:8]}  score={t['score']:.4f}  "
            f"time={t['training_time_s']:.2f}s  succeeded={t['succeeded']}"
        )
    return 0


def run_benchmark(args: argparse.Namespace) -> int:
    dataset = Path(args.dataset) if args.dataset else _default_dataset()
    ws_dir = str(Path(args.workspace)) if args.workspace else str(_project_root() / ".automl" / "benchmark")

    runner = BenchmarkRunner(dataset_path=str(dataset), workspace_root=ws_dir)
    results = runner.run_all(scenario_filter=args.scenario)

    if getattr(args, "json", False):
        print(json.dumps(results, indent=2))
        return 0

    _print_table(results, f"CATML Benchmark (platform {PLATFORM_VERSION})")
    return 0


def show_benchmark_history(args: argparse.Namespace) -> int:
    ws_dir = str(Path(args.workspace)) if args.workspace else str(_project_root() / ".automl" / "benchmark")
    runner = BenchmarkRunner(dataset_path=str(_default_dataset()), workspace_root=ws_dir)
    rows = runner.compare_history()
    if not rows:
        print("No benchmark history yet. Run: automl benchmark run")
        return 0
    print(f"\nBenchmark history (platform {PLATFORM_VERSION})")
    for row in rows[: args.limit]:
        print(
            f"  {row['timestamp'][:19]}  {row['scenario_id']:28s}  "
            f"{row['metric']}={row['best_score']:.4f}  model={row.get('best_model') or '-'}"
        )
    return 0


def list_tasks(args: argparse.Namespace) -> int:
    _, _, qry = build_application(root_dir=args.workspace)
    tasks = qry.dispatch(ListTaskTypesQuery())
    print("\nTask types and compatible models\n")
    for task in tasks:
        print(f"  {task['task_type']}")
        print(f"    {task['label']} — {task['description']}")
        print(f"    metrics: {', '.join(task['metrics'])}  (default: {task['default_metric']})")
        print(f"    models:  {', '.join(task['models'])}\n")
    return 0


def plan_task(args: argparse.Namespace) -> int:
    ws_dir = str(Path(args.workspace)) if args.workspace else str(_project_root() / ".automl" / "demo")
    ws, _, qry = build_application(root_dir=ws_dir)

    if args.dataset_id:
        plan = qry.dispatch(GetTaskPlanQuery(args.dataset_id))
    else:
        dataset_path = Path(args.dataset) if args.dataset else _default_dataset()
        dataset = ws.register_dataset(
            name="plan_preview",
            path=dataset_path,
            target=args.target,
            task_type=args.task_type,
        )
        plan = qry.dispatch(GetTaskPlanQuery(dataset.id))

    print(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False))
    return 0


def features_select_cli(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset) if args.dataset else _default_dataset()
    workspace_dir = Path(args.workspace) if args.workspace else _project_root() / ".automl" / "features_demo"

    ws, cmd, qry = build_application(root_dir=str(workspace_dir))
    dataset = ws.register_dataset(
        name="features_dataset",
        path=dataset_path,
        target=args.target,
    )
    run = ws.create_run(dataset)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    top_k = [int(k.strip()) for k in args.top_k.split(",") if k.strip()]
    strat = FeatureSelectionStrategy(
        methods=methods,
        top_k=top_k,
        combine_method=args.combine_method,
    )

    candidates = cmd.dispatch(SelectFeaturesCommand(run_id=run.id, strategy=strat))
    rankings = qry.dispatch(GetFeatureRankingQuery(run_id=run.id))

    if args.json:
        payload = {
            "run_id": run.id,
            "candidates": [c.to_dict() for c in candidates],
            "rankings": rankings,
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(f"\nFeature Selection for run: {run.id}")
    print("Top Feature Rankings:")
    print("---------------------")
    for r in rankings[:10]:
        print(f"  Rank {r['rank']:2d}: {r['feature_name']:20s} score={r['score']:.4f} (method={r['method']})")

    print("\nGenerated FeatureSet Candidates:")
    print("--------------------------------")
    for c in candidates:
        print(f"  Candidate: {c.name:20s} ({c.k} features): {', '.join(c.feature_names)}")

    return 0


def features_ablation_cli(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset) if args.dataset else _default_dataset()
    workspace_dir = Path(args.workspace) if args.workspace else _project_root() / ".automl" / "ablation_demo"

    ws, cmd, qry = build_application(root_dir=str(workspace_dir))
    dataset = ws.register_dataset(
        name="ablation_dataset",
        path=dataset_path,
        target=args.target,
    )
    run = ws.create_run(dataset)

    cmd.dispatch(SelectFeaturesCommand(run_id=run.id))

    candidates = cmd.dispatch(
        PlanAblationExperimentsCommand(
            run_id=run.id,
            max_features=args.max_features,
            auto_enqueue=args.auto_run,
        )
    )

    if args.auto_run:
        print(f"\nRunning {len(candidates)} scheduled ablation experiments...")
        cmd.dispatch(RunScheduledExperimentsCommand(run_id=run.id, max_experiments=len(candidates)))
        lb = qry.dispatch(GetLeaderboardQuery(run.id))
        print("\nAblation Experiment Results:")
        print("----------------------------")
        for entry in lb:
            print(f"  {entry['experiment_id']:20s} {entry['model_id']:18s} {entry['metric']}={entry['score']:.4f}")
    else:
        print(f"\nPlanned {len(candidates)} Ablation Experiments:")
        print("---------------------------------------")
        for c in candidates:
            print(f"  Candidate: {c.name:30s} hypothesis={c.hypothesis}")

    return 0


def features_temporal_cli(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset) if args.dataset else _default_dataset()
    workspace_dir = Path(args.workspace) if args.workspace else _project_root() / ".automl" / "temporal_demo"

    ws, cmd, qry = build_application(root_dir=str(workspace_dir))
    dataset = ws.register_dataset(
        name="temporal_dataset",
        path=dataset_path,
        target=args.target,
    )
    run = ws.create_run(dataset)

    # 1. Query detected temporal structure
    structure = qry.dispatch(DetectTemporalStructureQuery(dataset_id=dataset.id))

    # 2. Command to generate temporal candidate feature sets
    candidate_sets = cmd.dispatch(
        GenerateTemporalFeaturesCommand(
            run_id=run.id,
            dataset_id=dataset.id,
            max_lags=args.max_lags,
            include_lags=not args.no_lags,
            include_deltas=not args.no_deltas,
            include_cyclical=not args.no_cyclical,
        )
    )

    if args.json:
        payload = {
            "run_id": run.id,
            "temporal_structure": structure,
            "candidate_feature_sets": [fs.to_dict() for fs in candidate_sets],
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(f"\nTemporal Dynamics Analysis for dataset: {dataset.id}")
    print(f"Sequential dataset: {structure.get('is_sequential', False)}")
    if structure.get("order_column"):
        print(f"Order Column: {structure['order_column']}")
    if structure.get("detected_periodicities"):
        print("Detected Periodicities:")
        for p in structure["detected_periodicities"]:
            print(f"  - {p['name']} (period={p['period']}) on column '{p['column']}'")

    print("\nGenerated Candidate Feature Sets:")
    print("---------------------------------")
    for fs in candidate_sets:
        print(f"  Set: {fs.name:28s} ({len(fs.feature_names)} features): {fs.lineage}")

    return 0


def list_plugins_cli(args: argparse.Namespace) -> int:
    _, _, qry = build_application(root_dir=args.workspace)
    plugins = qry.dispatch(
        ListPluginsQuery(
            plugin_type=args.type if hasattr(args, "type") else None,
            task_type=args.task_type if hasattr(args, "task_type") else None,
        )
    )
    if getattr(args, "json", False):
        print(json.dumps(plugins, indent=2))
        return 0

    print(f"\nRegistered Plugins (Platform V{PLATFORM_VERSION})\n")
    print(f"  {'ID':22s} {'Name':26s} {'Type':10s} {'Backend':24s} {'Version':8s} {'Supported Tasks'}")
    print("  " + "-" * 110)
    for p in plugins:
        caps = p.get("capabilities", {})
        tasks = ", ".join(caps.get("supported_tasks", [])) or "all"
        backend = p.get("backend_status", "native")
        print(f"  {p['plugin_id']:22s} {p['name']:26s} {p['plugin_type']:10s} {backend:24s} {p['version']:8s} {tasks}")
    print()
    return 0


def meta_priors_cli(args: argparse.Namespace) -> int:
    ws, _, qry = build_application(root_dir=args.workspace)
    dataset_input = args.dataset
    target = getattr(args, "target", "churn")

    if Path(dataset_input).exists():
        dataset = ws.register_dataset(
            name=Path(dataset_input).stem,
            path=Path(dataset_input),
            target=target,
        )
        dataset_id = dataset.id
    else:
        dataset_id = dataset_input

    knowledge = qry.dispatch(GetMetaKnowledgeQuery(dataset_id=dataset_id))
    data = knowledge.to_dict()

    if getattr(args, "json", False):
        print(json.dumps(data, indent=2))
        return 0

    fp = data.get("current_fingerprint", {})
    similar = data.get("similar_datasets", [])
    rankings = data.get("historical_rankings", [])
    warm = data.get("warm_start", {})

    print(f"\nCATML Meta-Learning Knowledge (V{knowledge.version})\n")
    print(f"  Dataset: {data.get('dataset_name', dataset_id)}")
    print(f"  Fingerprint: {fp.get('rows', 0)} rows, {fp.get('features', 0)} features")
    print(f"  Modality balance: {int(fp.get('numerical_ratio', 0)*100)}% numeric / {int(fp.get('categorical_ratio', 0)*100)}% categorical")
    print(f"  Missing ratio: {fp.get('missing_ratio', 0.0):.2%}, Target entropy: {fp.get('target_entropy', 0.0)}")

    print("\n  Top Similar Benchmark Datasets:")
    for s in similar:
        print(f"    - {s['name']} (similarity: {s['similarity']:.2f})")

    print("\n  Historical Model Rankings (Empirical):")
    print(f"    {'Model':18s} {'Experiments':14s} {'Mean Rank':12s} {'Win Rate'}")
    print("    " + "-" * 54)
    for r in rankings:
        print(f"    {r['model']:18s} {r['experiments']:<14d} {r['mean_rank']:<12.1f} {r['win_rate']:.1%}")

    print("\n  Warm Start Recommendation:")
    print(f"    Model:  {warm.get('recommended_model', 'N/A')}")
    print(f"    Params: {json.dumps(warm.get('params', {}))}")
    print(f"    Gain:   {warm.get('expected_search_reduction', 'N/A')} search space reduction")
    if warm.get("reason"):
        print(f"    Reason: {warm.get('reason')}")
    print()
    return 0


def predict_cli(args: argparse.Namespace) -> int:
    ws, cmd, qry = build_application(root_dir=args.workspace)
    run_id = args.run_id

    if not run_id:
        if not args.dataset:
            print("Error: Either --run-id or --dataset must be specified.", file=sys.stderr)
            return 1
        dataset_path = Path(args.dataset)
        dataset = ws.register_dataset(
            name="predict_train_ds",
            path=dataset_path,
            target=args.target,
        )
        runs = ws.list_runs(dataset_id=dataset.id)
        if runs:
            run_id = runs[-1].id
        else:
            print("Error: No existing runs found for dataset. Please run experiments first.", file=sys.stderr)
            return 1

    test_path = Path(args.test_dataset)
    if not test_path.exists():
        print(f"Error: Test dataset '{test_path}' does not exist.", file=sys.stderr)
        return 1

    out_path = args.output or "submission.csv"

    if args.models is not None and args.folds is None:
        print("Error: --models requires --folds.", file=sys.stderr)
        return 1
    if args.folds is not None:
        try:
            experiment_id = cmd.dispatch(GenerateOOFSubmissionCommand(
                run_id=run_id, test_dataset_path=str(test_path), output_path=out_path,
                id_column=args.id_column, template_path=args.template,
                experiment_id=args.experiment_id, predict_proba=args.proba,
                folds=args.folds, model_ids=[m.strip() for m in args.models.split(",")] if args.models is not None else None,
                method=getattr(args, "method", "average") or "average",
                meta_model=getattr(args, "meta_model", "ridge") or "ridge",
                max_seconds=args.oof_timeout,
            ))
            res = qry.dispatch(GetOOFResultQuery(run_id=run_id, experiment_id=experiment_id))
        except (ValueError, KeyError, RuntimeError, TimeoutError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    else:
        res = cmd.dispatch(
            GenerateSubmissionCommand(
                run_id=run_id,
                test_dataset_path=str(test_path),
                output_path=out_path,
                id_column=args.id_column,
                template_path=getattr(args, "template", None),
                experiment_id=args.experiment_id,
                predict_proba=args.proba,
            )
        )

    if getattr(args, "json", False):
        print(json.dumps(res, indent=2))
        return 0

    print("\nSubmission Generated Successfully!")
    print("---------------------------------")
    print(f"  Run ID:        {run_id}")
    print(f"  Test Dataset:  {test_path}")
    print(f"  Output CSV:    {res['output_path']}")
    print(f"  Row Count:     {res['row_count']}")
    print(f"  ID Column:     {res['id_column']}")
    print(f"  Target Column: {res['target_column']}")
    print(f"  Probabilities: {res['predict_proba']}")
    if res.get("folds"):
        print(f"  OOF folds:     {res['folds']}")
        print(f"  OOF method:    {res.get('method', 'average')}")
        print(f"  ROC-AUC:       {res['score']:.6f} (baseline {res['baseline_score']:.6f}; delta {res['delta']:+.6f})")
    if res.get("template_used"):
        print(f"  Template:      {res['template_used']}\n")
    else:
        print()
    return 0


def launch_ui_cli(args: argparse.Namespace) -> int:
    from automl.interfaces.web.server import run_web_dashboard

    if args.workspace:
        ws_dir = str(Path(args.workspace))
    else:
        ws_dir = str(_project_root() / ".automl" / "default")

    port = args.port or 8080
    host = getattr(args, "host", "127.0.0.1")
    auth_token = getattr(args, "auth_token", None)
    insecure_no_auth = getattr(args, "insecure_no_auth", False)
    run_web_dashboard(
        port=port,
        workspace_dir=ws_dir,
        host=host,
        auth_token=auth_token,
        insecure_no_auth=insecure_no_auth,
    )
    return 0


def jobs_cli(args) -> int:
    from automl.application.commands.job_commands import SubmitJobCommand, ControlJobCommand
    from automl.application.queries.job_queries import GetJobQuery, ListJobsQuery
    if args.job_action == "worker":
        from automl.infrastructure.jobs.worker import JobWorker
        worker = JobWorker(args.workspace).start(background=False)
        try:
            while True:
                if not worker.run_once():
                    import time
                    time.sleep(0.25)
        except KeyboardInterrupt:
            pass
        finally:
            worker.close()
        return 0
    _, commands, queries = build_application(args.workspace)
    if args.job_action == "submit":
        result = {"job_id": commands.dispatch(SubmitJobCommand(args.operation, args.run_id,
                  json.loads(args.payload), args.key))}
    elif args.job_action == "list":
        result = queries.dispatch(ListJobsQuery(args.run_id))
    elif args.job_action == "show":
        result = queries.dispatch(GetJobQuery(args.job_id))
    else:
        result = {"job_id": commands.dispatch(ControlJobCommand(args.job_id, args.job_action))}
    print(json.dumps(result, indent=2))
    return 0


def mcp_cli(args: argparse.Namespace) -> int:
    from automl.interfaces.cli.mcp_cli import run_mcp_cli
    return run_mcp_cli(args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="automl", description=f"AutoML Platform CLI (V{PLATFORM_VERSION})")
    sub = parser.add_subparsers(dest="command", required=True)
    fit_p = sub.add_parser("fit", help="Fit candidate AutoML models on a dataset and export winning model artifact")
    fit_p.add_argument("dataset", help="Path to input dataset (CSV)")
    fit_p.add_argument("--target", "-t", required=True, help="Target column name")
    fit_p.add_argument("--task", default=None, choices=["classification", "regression", "binary_classification", "multiclass_classification"], help="Explicit task type (default: auto-detected)")
    fit_p.add_argument("--models", "-m", default=None, help="Comma-separated list of candidate models (e.g. logistic_regression,random_forest,lightgbm)")
    fit_p.add_argument("--output-model", "-o", default="catml-runs/model.pkl", help="Output path for standalone ModelArtifact (default: catml-runs/model.pkl)")
    fit_p.add_argument("--workspace", default=None, help="Optional persistent workspace directory")
    fit_p.add_argument("--json", action="store_true", help="Output results as JSON")
    fit_p.set_defaults(func=fit_cli)

    doctor_p = sub.add_parser("doctor", help="Inspect environment dependencies, hardware acceleration, and workspace health")
    doctor_p.add_argument("--workspace", default=None, help="Path to workspace root to probe")
    doctor_p.add_argument("--json", action="store_true", help="Output diagnostic report as machine-readable JSON")
    from automl.interfaces.cli.doctor_cli import run_doctor_cli
    doctor_p.set_defaults(func=run_doctor_cli)

    demo = sub.add_parser("run-demo", help="Run demo experiment via CommandBus")
    demo.add_argument("--dataset")
    demo.add_argument("--workspace")
    demo.add_argument("--auto", action="store_true", help="Use automated experiment planner and priority scheduler")
    demo.add_argument("--json", action="store_true")
    demo.set_defaults(func=run_demo)

    plan_exp = sub.add_parser("plan-experiments", help="Plan and prioritize experiments for a dataset")
    plan_exp.add_argument("--dataset")
    plan_exp.add_argument("--workspace")
    plan_exp.add_argument("--target", default="churn")
    plan_exp.add_argument("--auto-run", action="store_true", help="Execute planned experiments in priority order")
    plan_exp.add_argument("--max-experiments", type=int, default=3)
    plan_exp.add_argument("--json", action="store_true")
    plan_exp.set_defaults(func=plan_experiments_cli)

    opt_parser = sub.add_parser("optimize", help="Optimize hyperparameters using Optuna or Random Search")
    opt_parser.add_argument("--dataset")
    opt_parser.add_argument("--workspace")
    opt_parser.add_argument("--target", default="churn")
    opt_parser.add_argument("--run-id")
    opt_parser.add_argument("--experiment-id")
    opt_parser.add_argument("--model", default=None, help="Model ID to optimize (e.g. logistic_regression, random_forest)")
    opt_parser.add_argument("--optimizer", default="optuna", choices=["optuna", "random_search"])
    opt_parser.add_argument("--trials", type=int, default=10, help="Number of trials to evaluate")
    opt_parser.add_argument("--timeout", type=float, default=None, help="Max time in seconds")
    opt_parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    opt_parser.add_argument("--json", action="store_true")
    opt_parser.set_defaults(func=optimize_cli)


    bench = sub.add_parser("benchmark", help="Benchmark harness")
    bench_sub = bench.add_subparsers(dest="bench_cmd", required=True)

    bench_run = bench_sub.add_parser("run", help="Run all benchmark scenarios")
    bench_run.add_argument("--dataset")
    bench_run.add_argument("--workspace")
    bench_run.add_argument("--scenario", help="Run a single scenario id")
    bench_run.add_argument("--json", action="store_true")
    bench_run.set_defaults(func=run_benchmark)

    bench_hist = bench_sub.add_parser("history", help="Show stored benchmark results")
    bench_hist.add_argument("--workspace")
    bench_hist.add_argument("--limit", type=int, default=20)
    bench_hist.set_defaults(func=show_benchmark_history)

    task = sub.add_parser("task", help="Task types and problem planning")
    task_sub = task.add_subparsers(dest="task_cmd", required=True)

    task_list = task_sub.add_parser("list", help="List task types and model catalogs")
    task_list.add_argument("--workspace")
    task_list.set_defaults(func=list_tasks)

    task_plan = task_sub.add_parser("plan", help="Plan task from dataset (infer classification/regression/clustering)")
    task_plan.add_argument("--dataset")
    task_plan.add_argument("--dataset-id", help="Use an already registered dataset")
    task_plan.add_argument("--target", default="churn")
    task_plan.add_argument("--task-type", help="Force: binary_classification, multiclass_classification, regression, clustering")
    task_plan.add_argument("--workspace")
    task_plan.set_defaults(func=plan_task)

    feat_parser = sub.add_parser("features", help="Feature discovery, selection, and ablation")
    feat_sub = feat_parser.add_subparsers(dest="feat_cmd", required=True)

    feat_sel = feat_sub.add_parser("select", help="Run statistical and ML feature selection")
    feat_sel.add_argument("--dataset")
    feat_sel.add_argument("--workspace")
    feat_sel.add_argument("--target", default="churn")
    feat_sel.add_argument("--methods", default="mutual_information,importance", help="Comma-separated: mutual_information, importance, correlation, variance")
    feat_sel.add_argument("--top-k", default="3,5", help="Comma-separated top-k values")
    feat_sel.add_argument("--combine-method", default="weighted_rank", choices=["weighted_rank", "borda"])
    feat_sel.add_argument("--json", action="store_true")
    feat_sel.set_defaults(func=features_select_cli)

    feat_abl = feat_sub.add_parser("ablation", help="Plan and execute feature ablation experiments")
    feat_abl.add_argument("--dataset")
    feat_abl.add_argument("--workspace")
    feat_abl.add_argument("--target", default="churn")
    feat_abl.add_argument("--max-features", type=int, default=3)
    feat_abl.add_argument("--auto-run", action="store_true", help="Execute planned ablation experiments immediately")
    feat_abl.add_argument("--json", action="store_true")
    feat_abl.set_defaults(func=features_ablation_cli)

    feat_temp = feat_sub.add_parser("temporal", help="Analyze temporal dynamics and generate candidate sequential features")
    feat_temp.add_argument("--dataset")
    feat_temp.add_argument("--workspace")
    feat_temp.add_argument("--target", default="churn")
    feat_temp.add_argument("--max-lags", type=int, default=1, help="Maximum number of lag and delta steps (default: 1)")
    feat_temp.add_argument("--no-lags", action="store_true", help="Disable autoregressive lag feature generation")
    feat_temp.add_argument("--no-deltas", action="store_true", help="Disable rate-of-change trend delta feature generation")
    feat_temp.add_argument("--no-cyclical", action="store_true", help="Disable cyclical trigonometric sine/cosine feature generation")
    feat_temp.add_argument("--json", action="store_true")
    feat_temp.set_defaults(func=features_temporal_cli)

    plugin_parser = sub.add_parser("plugin", help="Plugin ecosystem management")
    plugin_sub = plugin_parser.add_subparsers(dest="plugin_cmd", required=True)

    plugin_list = plugin_sub.add_parser("list", help="List registered plugins")
    plugin_list.add_argument("--workspace")
    plugin_list.add_argument("--type", choices=["model", "metric", "preprocessor", "optimizer"], default=None, help="Filter by plugin type")
    plugin_list.add_argument("--task-type", default=None, help="Filter by supported task type")
    plugin_list.add_argument("--json", action="store_true")
    plugin_list.set_defaults(func=list_plugins_cli)

    meta_parser = sub.add_parser("meta", help="Meta-learning knowledge, dataset fingerprinting, and warm starts")
    meta_sub = meta_parser.add_subparsers(dest="meta_cmd", required=True)

    meta_priors = meta_sub.add_parser("priors", help="Retrieve meta-learning priors and warm-start recommendations for a dataset")
    meta_priors.add_argument("--dataset", required=True, help="Path to CSV dataset or registered dataset ID")
    meta_priors.add_argument("--target", default="churn", help="Target column name (default: churn)")
    meta_priors.add_argument("--workspace", help="Workspace directory")
    meta_priors.add_argument("--json", action="store_true", help="Output raw JSON")
    meta_priors.set_defaults(func=meta_priors_cli)

    pred_parser = sub.add_parser("predict", help="Generate predictions and Kaggle-ready submission file")
    pred_parser.add_argument("--workspace")
    pred_parser.add_argument("--run-id", help="Run ID to use trained model from")
    pred_parser.add_argument("--dataset", help="Optional path to train dataset to look up runs")
    pred_parser.add_argument("--target", default="churn", help="Target column name")
    pred_parser.add_argument("--test-dataset", required=True, help="Path to test CSV file")
    pred_parser.add_argument("--output", default="submission.csv", help="Output submission CSV path (default: submission.csv)")
    pred_parser.add_argument("--id-column", help="ID column name (e.g. id, PassengerId, customer_id)")
    pred_parser.add_argument("--template", help="Path to sample submission CSV to match column names and row ID ordering exactly")
    pred_parser.add_argument("--experiment-id", help="Optional specific experiment ID to use")
    pred_parser.add_argument("--proba", action="store_true", help="Output probabilities instead of binary labels")
    pred_parser.add_argument("--folds", type=int, help="OOF folds for binary classification; e.g. 5")
    pred_parser.add_argument("--models", help="Comma-separated individual models for OOF (e.g. lightgbm,xgboost,catboost)")
    pred_parser.add_argument("--method", choices=["average", "rank", "simplex", "stacked"], default="average", help="Blending method for OOF (default: average)")
    pred_parser.add_argument("--meta-model", choices=["ridge", "logistic_regression", "lasso"], default="ridge", help="Meta-estimator for Level-2 stacking (default: ridge)")
    pred_parser.add_argument("--oof-timeout", type=float, default=300.0, help="OOF time budget in seconds, checked between fits (default: 300)")
    pred_parser.add_argument("--json", action="store_true")
    pred_parser.set_defaults(func=predict_cli)

    ui_parser = sub.add_parser("ui", help="Launch interactive web dashboard")
    ui_parser.add_argument("--port", type=int, default=8080, help="Web server port (default: 8080)")
    ui_parser.add_argument("--host", default="127.0.0.1", help="Web server host interface (default: 127.0.0.1 for local privacy; use 0.0.0.0 to expose externally)")
    ui_parser.add_argument("--auth-token", default=None, help="Authentication token for remote access (required if bound to non-localhost interface)")
    ui_parser.add_argument("--insecure-no-auth", action="store_true", help="Explicitly allow non-localhost binding without an authentication token")
    ui_parser.add_argument("--workspace", help="Connected workspace directory (default: auto)")
    ui_parser.set_defaults(func=launch_ui_cli)

    job_parser = sub.add_parser("job", help="Persistent background jobs and worker")
    job_sub = job_parser.add_subparsers(dest="job_action", required=True)
    for action in ("submit", "list", "show", "pause", "resume", "cancel", "retry", "worker"):
        action_parser = job_sub.add_parser(action)
        action_parser.add_argument("--workspace", required=True)
        action_parser.set_defaults(func=jobs_cli)
        if action in {"show", "pause", "resume", "cancel", "retry"}:
            action_parser.add_argument("--job-id", required=True)
        if action in {"list", "submit"}:
            action_parser.add_argument("--run-id", required=action == "submit")
        if action == "submit":
            action_parser.add_argument("--operation", choices=("experiment", "oof", "submission"), required=True)
            action_parser.add_argument("--payload", required=True, help="JSON operation arguments")
            action_parser.add_argument("--key", required=True, help="Stable idempotency key for this request")

    mcp_parser = sub.add_parser("mcp", help="Launch Model Context Protocol (MCP) server")
    mcp_parser.add_argument("--workspace", help="Path to workspace root directory (default: auto)")
    mcp_parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio", help="MCP transport protocol (default: stdio)")
    mcp_parser.add_argument("--host", default="127.0.0.1", help="Host interface for streamable-http (default: 127.0.0.1)")
    mcp_parser.add_argument("--port", type=int, default=8000, help="Port for streamable-http (default: 8000)")
    mcp_parser.add_argument("--path", default="/mcp", help="Path prefix for streamable-http (default: /mcp)")
    mcp_parser.add_argument("--token", default=None, help="Authentication token for streamable-http transport")
    mcp_parser.add_argument("--insecure-no-auth", action="store_true", help="Allow binding to external interfaces without authentication token (INSECURE)")
    mcp_parser.add_argument(
        "--allow-mutations",
        action="store_true",
        help="Disable human approval gate for mutating MCP tools (local/dev only; default remains require-approval)",
    )
    mcp_parser.set_defaults(func=mcp_cli)

    agent_parser = sub.add_parser("agent", help="LLM Agent governance, approvals, and audit ledger")
    from automl.interfaces.cli.agent_cli import register_agent_subparser
    from automl.interfaces.cli.agent_session_cli import register_agent_session_subparser
    register_agent_subparser(agent_parser)
    register_agent_session_subparser(agent_parser)

    explore_parser = sub.add_parser("explore", help="Statistical exploratory data analysis and studies")
    explore_sub = explore_parser.add_subparsers(dest="explore_action", required=True)
    from automl.interfaces.cli.explore_cli import run_explore_cli

    exp_create = explore_sub.add_parser("create", help="Create an exploratory study on a dataset")
    exp_create.add_argument(
        "dataset_id",
        nargs="?",
        default=None,
        help="ID of dataset to study (positional; or use --dataset-id)",
    )
    exp_create.add_argument(
        "--dataset-id",
        dest="dataset_id_flag",
        default=None,
        help="ID of dataset to study (alias for positional dataset_id)",
    )
    exp_create.add_argument("--name", "-n", default=None, help="Name of the study")
    exp_create.add_argument("--target", "-t", default=None, help="Optional target column for supervised associations")
    exp_create.add_argument("--workspace", default=None, help="Path to workspace root")
    exp_create.add_argument("--json", action="store_true", help="Output results as JSON")
    exp_create.set_defaults(func=run_explore_cli)

    exp_run = explore_sub.add_parser("run", help="Execute statistical analysis on a study")
    exp_run.add_argument(
        "study_id",
        nargs="?",
        default=None,
        help="ID of study to run (positional; or use --study-id)",
    )
    exp_run.add_argument(
        "--study-id",
        dest="study_id_flag",
        default=None,
        help="ID of study to run (alias for positional study_id)",
    )
    exp_run.add_argument("--workspace", default=None, help="Path to workspace root")
    exp_run.add_argument("--json", action="store_true", help="Output results as JSON")
    exp_run.set_defaults(func=run_explore_cli)

    exp_list = explore_sub.add_parser("list", help="List registered exploratory studies")
    exp_list.add_argument("--dataset", "-d", default=None, help="Filter by dataset ID")
    exp_list.add_argument("--workspace", default=None, help="Path to workspace root")
    exp_list.add_argument("--json", action="store_true", help="Output results as JSON")
    exp_list.set_defaults(func=run_explore_cli)

    exp_show = explore_sub.add_parser("show", help="Show details of a study")
    exp_show.add_argument("study_id", help="ID of study to display")
    exp_show.add_argument("--workspace", default=None, help="Path to workspace root")
    exp_show.add_argument("--json", action="store_true", help="Output results as JSON")
    exp_show.set_defaults(func=run_explore_cli)

    exp_find = explore_sub.add_parser("findings", help="List statistical findings of a study")
    exp_find.add_argument("study_id", help="ID of study to inspect")
    exp_find.add_argument("--type", "-t", default=None, help="Filter findings by type (e.g. correlation, skewness)")
    exp_find.add_argument("--min-sig", type=float, default=None, help="Filter findings by significance threshold (e.g. 0.05)")
    exp_find.add_argument("--limit", "-l", type=int, default=None, help="Maximum number of findings to display")
    exp_find.add_argument("--workspace", default=None, help="Path to workspace root")
    exp_find.add_argument("--json", action="store_true", help="Output results as JSON")
    exp_find.set_defaults(func=run_explore_cli)

    exp_export = explore_sub.add_parser("export", help="Export study technical report (markdown or json)")
    exp_export.add_argument("study_id", help="ID of study to export")
    exp_export.add_argument("--output", "-o", default=None, help="Output file path (prints to stdout if omitted)")
    exp_export.add_argument("--format", "-f", choices=["markdown", "md", "json"], default="markdown", help="Report format")
    exp_export.add_argument("--workspace", default=None, help="Path to workspace root")
    exp_export.set_defaults(func=run_explore_cli)

    exp_verify = explore_sub.add_parser("verify", help="Empirically verify a hypothesis against AutoML baseline (Propose ≠ Accept)")
    exp_verify.add_argument("hypothesis_id", help="ID of hypothesis to verify")
    exp_verify.add_argument("--run-id", "-r", default=None, help="Optional existing AutoML run ID")
    exp_verify.add_argument("--min-improvement", "-m", type=float, default=0.0, help="Minimum delta required for acceptance")
    exp_verify.add_argument("--workspace", default=None, help="Path to workspace root")
    exp_verify.add_argument("--json", action="store_true", help="Output result as JSON")
    exp_verify.set_defaults(func=run_explore_cli)

    # Recipe subparsers
    recipe_parser = sub.add_parser("recipe", help="Manage reproducible Experiment Recipes born from verified evidence")
    recipe_sub = recipe_parser.add_subparsers(dest="recipe_action", required=True)
    from automl.interfaces.cli.recipe_cli import run_recipe_cli

    rcp_list_v = recipe_sub.add_parser("list-verified", help="List verified improvements from explore studies")
    rcp_list_v.add_argument("--study", "-s", default=None, help="Filter by study ID")
    rcp_list_v.add_argument("--dataset", "-d", default=None, help="Filter by dataset ID")
    rcp_list_v.add_argument("--all", "-a", action="store_true", help="Include rejected hypotheses")
    rcp_list_v.add_argument("--workspace", default=None, help="Path to workspace root")
    rcp_list_v.add_argument("--json", action="store_true", help="Output results as JSON")
    rcp_list_v.set_defaults(func=run_recipe_cli)

    rcp_create = recipe_sub.add_parser("create-from-evidence", help="Create reproducible recipe from verified evidence link")
    rcp_create.add_argument("evidence_id", help="ID of verified evidence link")
    rcp_create.add_argument("--name", "-n", default=None, help="Descriptive recipe name")
    rcp_create.add_argument("--notes", default=None, help="Audit notes")
    rcp_create.add_argument("--model", "-m", action="append", default=None, help="Model(s) to associate")
    rcp_create.add_argument("--workspace", default=None, help="Path to workspace root")
    rcp_create.add_argument("--json", action="store_true", help="Output result as JSON")
    rcp_create.set_defaults(func=run_recipe_cli)

    rcp_list = recipe_sub.add_parser("list", help="List experiment recipes")
    rcp_list.add_argument("--dataset", "-d", default=None, help="Filter by dataset ID")
    rcp_list.add_argument("--status", choices=["draft", "selected", "submitted", "promoted", "archived"], default=None, help="Filter by status")
    rcp_list.add_argument("--workspace", default=None, help="Path to workspace root")
    rcp_list.add_argument("--json", action="store_true", help="Output results as JSON")
    rcp_list.set_defaults(func=run_recipe_cli)

    rcp_show = recipe_sub.add_parser("show", help="Show full details of a recipe")
    rcp_show.add_argument("recipe_id", help="Recipe ID")
    rcp_show.add_argument("--workspace", default=None, help="Path to workspace root")
    rcp_show.add_argument("--json", action="store_true", help="Output details as JSON")
    rcp_show.set_defaults(func=run_recipe_cli)

    rcp_select = recipe_sub.add_parser("select", help="Select recipe for evaluation/experimentation")
    rcp_select.add_argument("recipe_id", help="Recipe ID to select")
    rcp_select.add_argument("--notes", default=None, help="Selection notes")
    rcp_select.add_argument("--workspace", default=None, help="Path to workspace root")
    rcp_select.add_argument("--json", action="store_true", help="Output result as JSON")
    rcp_select.set_defaults(func=run_recipe_cli)

    rcp_submit = recipe_sub.add_parser("submit", help="Submit and execute experiments using recipe")
    rcp_submit.add_argument("recipe_id", help="Recipe ID to submit")
    rcp_submit.add_argument("--run", "-r", default=None, help="Target run ID (defaults to active run)")
    rcp_submit.add_argument("--model", "-m", action="append", default=None, help="Optional model override(s)")
    rcp_submit.add_argument("--notes", default=None, help="Execution notes")
    rcp_submit.add_argument("--workspace", default=None, help="Path to workspace root")
    rcp_submit.add_argument("--json", action="store_true", help="Output result as JSON")
    rcp_submit.set_defaults(func=run_recipe_cli)

    rcp_promote = recipe_sub.add_parser("promote", help="Explicitly promote recipe as verified production candidate")
    rcp_promote.add_argument("recipe_id", help="Recipe ID to promote")
    rcp_promote.add_argument("--reason", default=None, help="Promotion reason")
    rcp_promote.add_argument("--notes", default=None, help="Promotion audit notes")
    rcp_promote.add_argument("--workspace", default=None, help="Path to workspace root")
    rcp_promote.add_argument("--json", action="store_true", help="Output result as JSON")
    rcp_promote.set_defaults(func=run_recipe_cli)

    rcp_archive = recipe_sub.add_parser("archive", help="Archive recipe")
    rcp_archive.add_argument("recipe_id", help="Recipe ID to archive")
    rcp_archive.add_argument("--workspace", default=None, help="Path to workspace root")
    rcp_archive.add_argument("--json", action="store_true", help="Output result as JSON")
    rcp_archive.set_defaults(func=run_recipe_cli)

    # Data Preparation subparsers
    data_parser = sub.add_parser("data", help="Inspect data quality and manage reproducible preparation plans")
    data_sub = data_parser.add_subparsers(dest="data_action", required=True)
    from automl.interfaces.cli.data_cli import run_data_cli

    # data inspect
    data_inspect = data_sub.add_parser("inspect", help="Inspect dataset quality, missing values, constants, and schema anomalies")
    data_inspect.add_argument("dataset_id", nargs="?", default=None, help="Dataset ID to inspect")
    data_inspect.add_argument("--dataset", "-d", default=None, help="Dataset ID alias")
    data_inspect.add_argument("--workspace", default=None, help="Path to workspace root")
    data_inspect.add_argument("--json", action="store_true", help="Output quality report as JSON")
    data_inspect.set_defaults(func=run_data_cli)

    # data prepare
    data_prep = data_sub.add_parser("prepare", help="Manage reproducible data preparation plans and versions")
    data_prep_sub = data_prep.add_subparsers(dest="prepare_action", required=True)

    # data prepare suggest
    prep_suggest = data_prep_sub.add_parser("suggest", help="Suggest a typed preparation plan based on diagnostics")
    prep_suggest.add_argument("dataset_id", nargs="?", default=None, help="Target dataset ID")
    prep_suggest.add_argument("--dataset", "-d", default=None, help="Target dataset ID alias")
    prep_suggest.add_argument("--name", "-n", default=None, help="Optional plan name")
    prep_suggest.add_argument("--no-pipeline-marks", action="store_true", help="Exclude pipeline-scoped imputation/scaling steps")
    prep_suggest.add_argument("--workspace", default=None, help="Path to workspace root")
    prep_suggest.add_argument("--json", action="store_true", help="Output plan as JSON")
    prep_suggest.set_defaults(func=run_data_cli)

    # data prepare preview
    prep_preview = data_prep_sub.add_parser("preview", help="Preview plan effect without mutating stored dataset")
    prep_preview.add_argument("plan_id", help="Plan ID to preview")
    prep_preview.add_argument("--workspace", default=None, help="Path to workspace root")
    prep_preview.add_argument("--json", action="store_true", help="Output preview as JSON")
    prep_preview.set_defaults(func=run_data_cli)

    # data prepare validate
    prep_val = data_prep_sub.add_parser("validate", help="Validate plan scopes and integrity")
    prep_val.add_argument("plan_id", help="Plan ID to validate")
    prep_val.add_argument("--workspace", default=None, help="Path to workspace root")
    prep_val.add_argument("--json", action="store_true", help="Output validation as JSON")
    prep_val.set_defaults(func=run_data_cli)

    # data prepare apply
    prep_apply = data_prep_sub.add_parser("apply", help="Apply plan to produce a new immutable dataset version")
    prep_apply.add_argument("plan_id", help="Plan ID to apply")
    prep_apply.add_argument("--approve", "--approved", "-y", "--yes", action="store_true", help="Approve destructive or semantic steps")
    prep_apply.add_argument("--notes", default=None, help="Audit notes")
    prep_apply.add_argument("--workspace", default=None, help="Path to workspace root")
    prep_apply.add_argument("--json", action="store_true", help="Output applied result as JSON")
    prep_apply.set_defaults(func=run_data_cli)

    # data prepare create
    prep_create = data_prep_sub.add_parser("create", help="Create an explicit preparation plan from step definitions")
    prep_create.add_argument("dataset_id", nargs="?", default=None, help="Target dataset ID")
    prep_create.add_argument("--dataset", "-d", default=None, help="Target dataset ID alias")
    prep_create.add_argument("--steps", required=True, help="JSON array of step objects or path to JSON file")
    prep_create.add_argument("--name", "-n", default=None, help="Plan name")
    prep_create.add_argument("--notes", default=None, help="Plan notes")
    prep_create.add_argument("--workspace", default=None, help="Path to workspace root")
    prep_create.add_argument("--json", action="store_true", help="Output result as JSON")
    prep_create.set_defaults(func=run_data_cli)

    # data prepare plans
    prep_plans = data_prep_sub.add_parser("plans", help="List preparation plans")
    prep_plans.add_argument("dataset_id", nargs="?", default=None, help="Filter by dataset ID")
    prep_plans.add_argument("--dataset", "-d", default=None, help="Filter by dataset ID alias")
    prep_plans.add_argument("--workspace", default=None, help="Path to workspace root")
    prep_plans.add_argument("--json", action="store_true", help="Output plans as JSON")
    prep_plans.set_defaults(func=run_data_cli)

    # data prepare show-plan
    prep_show = data_prep_sub.add_parser("show-plan", help="Show full details of a preparation plan")
    prep_show.add_argument("plan_id", help="Plan ID to show")
    prep_show.add_argument("--workspace", default=None, help="Path to workspace root")
    prep_show.add_argument("--json", action="store_true", help="Output plan details as JSON")
    prep_show.set_defaults(func=run_data_cli)

    # data prepare versions
    prep_vers = data_prep_sub.add_parser("versions", help="List immutable dataset versions produced from parent dataset")
    prep_vers.add_argument("parent_dataset_id", nargs="?", default=None, help="Parent dataset ID")
    prep_vers.add_argument("--parent", default=None, help="Parent dataset ID alias")
    prep_vers.add_argument("--workspace", default=None, help="Path to workspace root")
    prep_vers.add_argument("--json", action="store_true", help="Output versions as JSON")
    prep_vers.set_defaults(func=run_data_cli)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
