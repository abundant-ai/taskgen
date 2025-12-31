from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harbor.models.trial.result import TrialResult
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from .models import (
    BaselineValidation,
    Classification,
    TaskVerdict,
    TrialClassification,
)
from .classifier import (
    TrialClassifier,
    classify_baseline_result,
    compute_task_verdict,
)
from taskgen.tools.harbor_runner import harbor_cmd_base


def _setup_claude_auth_preference(console: Console) -> None:
    """Setup Claude Code to prefer OAuth token over API key.
    
    For Claude Code trials and classification, we prefer OAuth token:
    1. CLAUDE_CODE_OAUTH_TOKEN (preferred - run 'claude setup-token')
    2. ANTHROPIC_API_KEY (fallback)
    
    Displays which authentication method is being used.
    """
    has_oauth = bool(os.getenv("CLAUDE_CODE_OAUTH_TOKEN"))
    has_api_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    
    if has_oauth:
        # Prefer OAuth - unset API key to ensure OAuth is used
        if "ANTHROPIC_API_KEY" in os.environ:
            os.environ.pop("ANTHROPIC_API_KEY")
        console.print("[dim]🔐 Claude Code authentication: OAuth token (preferred)[/dim]")
    elif has_api_key:
        # Use API key - unset OAuth to ensure API key is used
        if "CLAUDE_CODE_OAUTH_TOKEN" in os.environ:
            os.environ.pop("CLAUDE_CODE_OAUTH_TOKEN")
        console.print("[dim]🔐 Claude Code authentication: API key (fallback)[/dim]")
        console.print("[dim]   Tip: For better security, use OAuth token ('claude setup-token')[/dim]")
    else:
        console.print("[yellow]⚠️  No Claude Code authentication configured[/yellow]")
        console.print("[yellow]   Set CLAUDE_CODE_OAUTH_TOKEN (preferred) or ANTHROPIC_API_KEY[/yellow]")


@dataclass
class TrialOutcome:
    """Result of a single trial (basic info before classification)."""

    trial_name: str
    trial_dir: Path
    reward: float | None
    exception_type: str | None
    exception_message: str | None


@dataclass
class QualityCheckResult:
    """Result of static quality check."""

    passed: bool
    issues: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Complete analysis result for a task."""

    task_id: str
    task_path: Path
    
    # Quality check
    quality_check: QualityCheckResult | None
    
    # Baseline validation
    baseline: BaselineValidation | None
    
    # Trial results
    trials_run: int
    success_rate: float
    trial_outcomes: list[TrialOutcome]
    
    # Classifications (NEW)
    classifications: list[TrialClassification]
    
    # Task verdict (NEW)
    verdict: TaskVerdict
    
    # Job directory
    job_dir: Path | None


@dataclass
class AnalyzeArgs:
    """Arguments for the analyze command."""

    task_path: Path
    agent: str = "claude-code"
    model: str = "anthropic/claude-sonnet-4-20250514"
    n_trials: int = 3
    n_concurrent: int = 1  # Number of concurrent trials (matches Harbor's -n flag)
    jobs_dir: Path = Path(".state/analyze-jobs")
    skip_quality_check: bool = False
    skip_baseline: bool = False  # Skip baseline validation (nop/oracle)
    skip_classify: bool = False  # Skip Claude Code classification
    analysis_model: str = "claude-sonnet-4-20250514"  # Model for Claude Code classification
    environment: str = "docker"  # Environment type (docker|daytona|e2b|modal|runloop|gke)
    verbose: bool = False
    timeout_multiplier: float = 1.0


def run_analyze(args: AnalyzeArgs) -> AnalysisResult:
    """Main entry point for task analysis."""
    console = Console()

    # Resolve task path
    task_path = args.task_path.resolve()
    if not task_path.is_dir():
        console.print(f"[red]Error: Task path does not exist: {task_path}[/red]")
        raise SystemExit(1)

    task_id = task_path.name
    dataset_path = task_path.parent

    # Check task structure
    if not (task_path / "tests" / "test.sh").exists():
        console.print(f"[red]Error: Not a valid task (missing tests/test.sh): {task_path}[/red]")
        raise SystemExit(1)

    # Setup and display Claude authentication for Claude Code agent
    if args.agent == "claude-code":
        _setup_claude_auth_preference(console)

    console.print(
        Panel.fit(
            f"Agent: {args.agent} | Model: {args.model} | Trials: {args.n_trials}",
            title=task_id,
        )
    )

    # Run analysis steps
    result = _run_analysis(args, task_id, task_path, dataset_path, console)

    # Print final report
    _print_report(result, console)

    return result


def _run_analysis(
    args: AnalyzeArgs,
    task_id: str,
    task_path: Path,
    dataset_path: Path,
    console: Console,
) -> AnalysisResult:
    """Run all analysis steps."""

    # Step 1: Static quality check
    quality_check = None
    if not args.skip_quality_check:
        console.print("\n[bold blue]Step 1/4: Static Quality Check[/bold blue]")
        quality_check = _run_quality_check(task_path, args.analysis_model, console)
    else:
        console.print("\n[dim]Step 1/4: Static Quality Check (skipped)[/dim]")

    # Step 2: Baseline validation (NEW)
    baseline = None
    if not args.skip_baseline:
        console.print("\n[bold blue]Step 2/4: Baseline Validation (nop/oracle)[/bold blue]")
        baseline = _run_baseline_validation(args, task_id, dataset_path, console)
    else:
        console.print("\n[dim]Step 2/4: Baseline Validation (skipped)[/dim]")

    # Step 3: Run agent trials
    console.print(f"\n[bold blue]Step 3/4: Running {args.n_trials} Agent Trials[/bold blue]")
    job_dir, trial_outcomes = _run_agent_trials(args, task_id, dataset_path, console)

    successes = sum(1 for t in trial_outcomes if t.reward == 1)
    failures = sum(1 for t in trial_outcomes if t.reward is not None and t.reward != 1)
    errors = sum(1 for t in trial_outcomes if t.exception_type is not None)
    success_rate = successes / len(trial_outcomes) if trial_outcomes else 0.0

    console.print(f"  Results: {successes} passed, {failures} failed, {errors} errors")
    console.print(f"  Success rate: {success_rate:.1%}")

    # Step 4: Classify trials (NEW - replaces summarize)
    classifications: list[TrialClassification] = []
    if not args.skip_classify and trial_outcomes:
        console.print("\n[bold blue]Step 4/4: Classifying Trial Outcomes[/bold blue]")
        
        # Get trial directories for failed trials (and optionally all)
        trial_dirs = [t.trial_dir for t in trial_outcomes if t.trial_dir.exists()]
        
        if trial_dirs:
            classifier = TrialClassifier(model=args.analysis_model)
            classifications = classifier.classify_trials_sync(trial_dirs, task_path, console)
            
            # Show classification summary
            task_problems = sum(1 for c in classifications if c.is_task_problem)
            agent_problems = sum(1 for c in classifications if c.classification == Classification.GOOD_FAILURE)
            
            if task_problems > 0:
                console.print(f"  [yellow]⚠ {task_problems} trial(s) indicate task problems[/yellow]")
            if agent_problems > 0:
                console.print(f"  [green]✓ {agent_problems} trial(s) are normal agent failures[/green]")
        else:
            console.print("  [dim]No trial directories found to classify[/dim]")
    else:
        console.print("\n[dim]Step 4/4: Classifying Trial Outcomes (skipped)[/dim]")

    # Compute task verdict
    quality_passed = quality_check is None or quality_check.passed
    verdict = compute_task_verdict(classifications, baseline, quality_passed)

    return AnalysisResult(
        task_id=task_id,
        task_path=task_path,
        quality_check=quality_check,
        baseline=baseline,
        trials_run=len(trial_outcomes),
        success_rate=success_rate,
        trial_outcomes=trial_outcomes,
        classifications=classifications,
        verdict=verdict,
        job_dir=job_dir,
    )


def _run_quality_check(
    task_path: Path,
    model: str,
    console: Console,
) -> QualityCheckResult:
    """Run Harbor's static quality check on the task."""
    cmd = harbor_cmd_base() + [
        "tasks",
        "check",
        str(task_path),
        "-m",
        model,
    ]

    with console.status("[cyan]Running quality check..."):
        proc = subprocess.run(cmd, capture_output=True, text=True)

    # Parse output to extract issues
    issues = []
    details: dict[str, Any] = {}

    output = proc.stdout + proc.stderr

    # Look for failed checks in output
    fail_keywords = ["fail", "FAIL", "❌"]
    for line in output.split("\n"):
        for keyword in fail_keywords:
            if keyword in line and "passed" not in line.lower():
                clean_line = line.strip()
                if clean_line and "│" in clean_line:
                    parts = [p.strip() for p in clean_line.split("│")]
                    if len(parts) >= 2 and any(k in parts[1].lower() for k in ["fail"]):
                        issues.append(parts[0])

    passed = proc.returncode == 0 and len(issues) == 0

    if passed:
        console.print("  [green]✓ Quality check passed[/green]")
    else:
        console.print("  [yellow]⚠ Quality check found issues:[/yellow]")
        for issue in issues[:5]:
            console.print(f"    - {issue}")

    return QualityCheckResult(passed=passed, issues=issues, details=details)


def _run_baseline_validation(
    args: AnalyzeArgs,
    task_id: str,
    dataset_path: Path,
    console: Console,
) -> BaselineValidation:
    """Run nop and oracle baseline agents to validate task correctness."""
    
    jobs_parent = args.jobs_dir.resolve()
    jobs_parent.mkdir(parents=True, exist_ok=True)
    
    baseline = BaselineValidation()
    
    # Run nop agent (should fail - reward=0)
    console.print("  Running nop agent (should fail)...")
    nop_reward, nop_error = _run_single_agent(
        task_id, dataset_path, "nop", jobs_parent, args.timeout_multiplier, args.environment
    )
    baseline.nop = classify_baseline_result("nop", nop_reward, nop_error)
    
    if baseline.nop.is_expected:
        console.print("    [green]✓ nop failed as expected[/green]")
    else:
        console.print("    [red]✗ CRITICAL: nop passed - task may be pre-solved![/red]")
    
    # Run oracle agent (should pass - reward=1)
    console.print("  Running oracle agent (should pass)...")
    oracle_reward, oracle_error = _run_single_agent(
        task_id, dataset_path, "oracle", jobs_parent, args.timeout_multiplier, args.environment
    )
    baseline.oracle = classify_baseline_result("oracle", oracle_reward, oracle_error)
    
    if baseline.oracle.is_expected:
        console.print("    [green]✓ oracle passed as expected[/green]")
    else:
        console.print("    [red]✗ CRITICAL: oracle failed - reference solution broken![/red]")
    
    return baseline


def _run_single_agent(
    task_id: str,
    dataset_path: Path,
    agent: str,
    jobs_dir: Path,
    timeout_multiplier: float,
    environment: str,
) -> tuple[float | None, str | None]:
    """Run a single agent and return (reward, error)."""
    cmd = harbor_cmd_base() + [
        "run",
        "-p", str(dataset_path),
        "-t", task_id,
        "-a", agent,
        "-k", "1",
        "-n", "1",
        "-e", environment,
        "--jobs-dir", str(jobs_dir),
        "--timeout-multiplier", str(timeout_multiplier),
    ]
    
    proc = subprocess.run(cmd, capture_output=True, text=True)
    
    # Find the job directory that was created
    job_dirs = sorted(
        [d for d in jobs_dir.iterdir() if d.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    
    if not job_dirs:
        return None, "No job directory created"
    
    # Find trial result
    job_dir = job_dirs[0]
    for trial_dir in job_dir.iterdir():
        if not trial_dir.is_dir():
            continue
        result_path = trial_dir / "result.json"
        if result_path.exists():
            try:
                result = TrialResult.model_validate_json(result_path.read_text())
                reward = None
                if result.verifier_result and result.verifier_result.rewards:
                    reward = result.verifier_result.rewards.get("reward")
                error = None
                if result.exception_info:
                    error = result.exception_info.exception_message
                return reward, error
            except Exception as e:
                return None, str(e)
    
    return None, "No trial result found"


def _run_agent_trials(
    args: AnalyzeArgs,
    task_id: str,
    dataset_path: Path,
    console: Console,
) -> tuple[Path | None, list[TrialOutcome]]:
    """Run multiple agent trials on the task."""

    _timestamp = int(time.time())
    jobs_parent = args.jobs_dir.resolve()
    jobs_parent.mkdir(parents=True, exist_ok=True)

    cmd = harbor_cmd_base() + [
        "run",
        "-p", str(dataset_path),
        "-t", task_id,
        "-a", args.agent,
        "-m", args.model,
        "-k", str(args.n_trials),
        "-n", str(args.n_concurrent),  # Matches Harbor's -n flag
        "-e", args.environment,
        "--jobs-dir", str(jobs_parent),
        "--timeout-multiplier", str(args.timeout_multiplier),
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        concurrent_msg = f" ({args.n_concurrent} concurrent)" if args.n_concurrent > 1 else ""
        task = progress.add_task(
            f"[cyan]Running {args.n_trials} trials with {args.agent}{concurrent_msg}...", total=None
        )

        _proc = subprocess.run(cmd, capture_output=True, text=True)
        progress.update(task, completed=True)

    # Find the job directory that was created
    job_dirs = sorted(
        [d for d in jobs_parent.iterdir() if d.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    job_dir = job_dirs[0] if job_dirs else None

    # Parse trial results
    trial_outcomes = []
    if job_dir:
        trial_outcomes = _parse_trial_results(job_dir)

    return job_dir, trial_outcomes


def _parse_trial_results(job_dir: Path) -> list[TrialOutcome]:
    """Parse trial results from a job directory."""
    outcomes = []

    for trial_dir in job_dir.iterdir():
        if not trial_dir.is_dir():
            continue

        result_path = trial_dir / "result.json"
        if not result_path.exists():
            continue

        try:
            result = TrialResult.model_validate_json(result_path.read_text())

            reward = None
            if result.verifier_result and result.verifier_result.rewards:
                reward = result.verifier_result.rewards.get("reward")

            exception_type = None
            exception_message = None
            if result.exception_info:
                exception_type = result.exception_info.exception_type
                exception_message = result.exception_info.exception_message

            outcomes.append(
                TrialOutcome(
                    trial_name=result.trial_name,
                    trial_dir=trial_dir,
                    reward=reward,
                    exception_type=exception_type,
                    exception_message=exception_message,
                )
            )
        except Exception as e:
            console = Console()
            console.print(f"[dim]Warning: Could not parse {result_path}: {e}[/dim]")

    return outcomes


def _print_report(result: AnalysisResult, console: Console) -> None:
    """Print the final analysis report."""
    console.print("\n")

    # Overall verdict
    verdict = result.verdict
    if verdict.is_good:
        verdict_style = "bold green"
        verdict_icon = "✅"
        verdict_text = f"GOOD TASK (confidence: {verdict.confidence})"
    else:
        verdict_style = "bold red"
        verdict_icon = "❌"
        verdict_text = "NEEDS REVIEW"

    console.print(
        Panel.fit(
            f"[{verdict_style}]{verdict_icon} {verdict_text}[/{verdict_style}]",
            title=f"Task Verdict: {result.task_id}",
        )
    )

    # Summary table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Check", style="cyan")
    table.add_column("Result")
    table.add_column("Details")

    # Quality check row
    if result.quality_check:
        qc_status = (
            "✅ Passed"
            if result.quality_check.passed
            else f"⚠️ {len(result.quality_check.issues)} issues"
        )
        qc_style = "green" if result.quality_check.passed else "yellow"
        table.add_row(
            "Quality Check",
            f"[{qc_style}]{qc_status}[/{qc_style}]",
            ", ".join(result.quality_check.issues[:3])
            if result.quality_check.issues
            else "All checks passed",
        )

    # Baseline validation row
    if result.baseline:
        baseline_ok = result.baseline.is_valid
        if baseline_ok:
            baseline_status = "✅ Valid"
            baseline_style = "green"
            baseline_details = "nop fails, oracle passes"
        else:
            baseline_status = "❌ Invalid"
            baseline_style = "red"
            baseline_details = "; ".join(result.baseline.issues)
        table.add_row(
            "Baseline (nop/oracle)",
            f"[{baseline_style}]{baseline_status}[/{baseline_style}]",
            baseline_details,
        )

    # Trials row
    trials_status = f"{result.success_rate:.0%} success rate"
    if result.success_rate >= 0.67:
        trials_style = "green"
        trials_icon = "✅"
    elif result.success_rate >= 0.33:
        trials_style = "yellow"
        trials_icon = "⚠️"
    else:
        trials_style = "red"
        trials_icon = "❌"

    successes = sum(1 for t in result.trial_outcomes if t.reward == 1)
    failures = sum(1 for t in result.trial_outcomes if t.reward is not None and t.reward != 1)
    errors = sum(1 for t in result.trial_outcomes if t.exception_type)

    table.add_row(
        f"Agent Trials ({result.trials_run})",
        f"[{trials_style}]{trials_icon} {trials_status}[/{trials_style}]",
        f"{successes} passed, {failures} failed, {errors} errors",
    )

    # Classification summary row
    if result.classifications:
        task_problems = verdict.task_problem_count
        agent_problems = verdict.agent_problem_count
        
        if task_problems > 0:
            class_status = f"⚠️ {task_problems} task problem(s)"
            class_style = "yellow"
        else:
            class_status = f"✅ {agent_problems} agent failure(s)"
            class_style = "green"
        
        table.add_row(
            "Classification",
            f"[{class_style}]{class_status}[/{class_style}]",
            f"{verdict.success_count} success, {task_problems} task issue, {agent_problems} agent issue",
        )

    console.print(table)

    # Show classification details
    if result.classifications:
        console.print("\n[bold]Trial Classifications:[/bold]")
        
        for c in result.classifications:
            # Color based on classification
            if c.classification == Classification.GOOD_SUCCESS:
                icon = "✅"
                style = "green"
            elif c.classification == Classification.GOOD_FAILURE:
                icon = "⚪"
                style = "dim"
            elif c.classification == Classification.BAD_SUCCESS:
                icon = "🔴"
                style = "red"
            elif c.classification == Classification.BAD_FAILURE:
                icon = "🟡"
                style = "yellow"
            else:
                icon = "⚫"
                style = "dim"
            
            console.print(f"\n  [{style}]{icon} {c.trial_name}: {c.classification.value} - {c.subtype}[/{style}]")
            console.print(f"     [dim]Evidence:[/dim] {c.evidence[:100]}...")
            console.print(f"     [dim]Root cause:[/dim] {c.root_cause}")
            if c.is_task_problem and c.recommendation != "N/A - task is fine":
                console.print(f"     [yellow]Recommendation:[/yellow] {c.recommendation}")

    # Show recommendations
    if verdict.recommendations:
        console.print("\n[bold yellow]Recommendations to Fix Task:[/bold yellow]")
        for i, rec in enumerate(verdict.recommendations, 1):
            console.print(f"  {i}. {rec}")

    # Primary issue
    if verdict.primary_issue:
        console.print(f"\n[bold]Primary Issue:[/bold] {verdict.primary_issue}")

    # Job directory
    if result.job_dir:
        console.print(f"\n[dim]Job artifacts: {result.job_dir}[/dim]")
