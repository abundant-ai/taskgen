from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    TextBlock,
)
from harbor.models.trial.result import TrialResult

from .models import (
    BaselineResult,
    BaselineValidation,
    Classification,
    TaskVerdict,
    TrialClassification,
)

from rich.console import Console


# Load prompt template
_PROMPT_PATH = Path(__file__).parent / "classify_prompt.txt"
_CLASSIFY_PROMPT = _PROMPT_PATH.read_text()


class TrialClassifier:
    """Classifies trial outcomes using Claude Code to identify task quality issues.
    
    Uses Claude Agent SDK with file access to explore trial artifacts
    and classify whether outcomes reveal task problems.
    
    Authentication (in priority order):
    1. CLAUDE_CODE_OAUTH_TOKEN environment variable (recommended)
       - Generate with: claude setup-token (requires Claude Pro/Max)
    2. ANTHROPIC_API_KEY environment variable (fallback)
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
    ):
        """Initialize the classifier.
        
        Args:
            model: Model name for Claude Code (default: claude-sonnet-4-20250514)
        """
        self._model = model
        self._setup_authentication()
    
    def _setup_authentication(self) -> None:
        """Setup authentication for Claude Code.
        
        Prefers OAuth token over API key. If OAuth token is available,
        unset API key to ensure OAuth is used.
        """
        has_oauth = bool(os.getenv("CLAUDE_CODE_OAUTH_TOKEN"))
        has_api_key = bool(os.getenv("ANTHROPIC_API_KEY"))
        
        if has_oauth:
            # Prefer OAuth - unset API key to ensure OAuth is used
            if "ANTHROPIC_API_KEY" in os.environ:
                os.environ.pop("ANTHROPIC_API_KEY")
            # No action needed - Claude SDK will use CLAUDE_CODE_OAUTH_TOKEN
        elif has_api_key:
            # Use API key - unset OAuth to ensure API key is used
            if "CLAUDE_CODE_OAUTH_TOKEN" in os.environ:
                os.environ.pop("CLAUDE_CODE_OAUTH_TOKEN")
            # No action needed - Claude SDK will use ANTHROPIC_API_KEY
        else:
            # No authentication available - will fail when trying to classify
            # We'll handle this gracefully in classify_trial
            pass
    
    async def classify_trial(
        self,
        trial_dir: Path,
        task_dir: Path,
    ) -> TrialClassification:
        """Classify a single trial outcome using Claude Code.
        
        Args:
            trial_dir: Path to trial directory (contains result.json, agent/, verifier/)
            task_dir: Path to task directory (contains instruction.md, solution/, tests/)
            
        Returns:
            TrialClassification with classification, evidence, and recommendations
        """
        # Read trial result to get the verified outcome
        result_path = trial_dir / "result.json"
        if not result_path.exists():
            return TrialClassification(
                trial_name=trial_dir.name,
                classification=Classification.HARNESS_ERROR,
                subtype="Missing Result",
                evidence="result.json not found in trial directory",
                root_cause="Trial did not complete - no result.json file",
                recommendation="Check Harbor logs for infrastructure issues",
                reward=None,
            )
        
        try:
            result = TrialResult.model_validate_json(result_path.read_text())
        except Exception as e:
            return TrialClassification(
                trial_name=trial_dir.name,
                classification=Classification.HARNESS_ERROR,
                subtype="Invalid Result",
                evidence=f"Could not parse result.json: {e}",
                root_cause="Trial result file is corrupted or malformed",
                recommendation="Check Harbor logs for what went wrong",
                reward=None,
            )
        
        # Extract reward
        reward = None
        if result.verifier_result and result.verifier_result.rewards:
            reward = result.verifier_result.rewards.get("reward")
        
        # Determine result string for prompt
        if reward == 1.0:
            result_str = "pass"
        elif reward == 0.0:
            result_str = "fail"
        else:
            result_str = f"unknown (reward={reward})"
        
        # Build prompt with paths for Claude to explore
        prompt = _CLASSIFY_PROMPT.format(
            result=result_str,
            task_dir=str(task_dir),
            trial_dir=str(trial_dir),
        )
        
        # Run Claude Code with file access
        options = ClaudeAgentOptions(
            permission_mode="bypassPermissions",
            allowed_tools=["Read", "Glob"],
            cwd=str(trial_dir),
            add_dirs=[str(task_dir)],
            model=self._model,
        )
        
        response_parts = []
        try:
            # Check for authentication before attempting to classify
            has_auth = bool(os.getenv("CLAUDE_CODE_OAUTH_TOKEN") or os.getenv("ANTHROPIC_API_KEY"))
            if not has_auth:
                raise RuntimeError(
                    "No authentication configured. Set either CLAUDE_CODE_OAUTH_TOKEN "
                    "(preferred, run 'claude setup-token') or ANTHROPIC_API_KEY"
                )
            
            async with ClaudeSDKClient(options=options) as client:
                await client.query(prompt)
                
                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                response_parts.append(block.text)
            
            response_text = "\n".join(response_parts)
            
            # Parse JSON from response
            classification = self._parse_response(response_text, trial_dir.name, reward)
            return classification
            
        except Exception as e:
            # Fallback classification based on reward
            if reward == 1.0:
                classification = Classification.GOOD_SUCCESS
                subtype = "Presumed Correct"
            elif reward == 0.0:
                classification = Classification.GOOD_FAILURE
                subtype = "Presumed Agent Error"
            else:
                classification = Classification.HARNESS_ERROR
                subtype = "Classification Failed"
            
            return TrialClassification(
                trial_name=trial_dir.name,
                classification=classification,
                subtype=subtype,
                evidence=f"Claude Code classification failed: {e}",
                root_cause="Could not analyze trial with Claude Code",
                recommendation="Review trial manually",
                reward=reward,
            )
    
    def _parse_response(
        self,
        response_text: str,
        trial_name: str,
        reward: float | None,
    ) -> TrialClassification:
        """Parse JSON classification from Claude's response."""
        # Try to extract JSON from response
        # Claude might include markdown or explanation around the JSON
        json_str = response_text.strip()
        
        # Try to find JSON object in response
        start_idx = json_str.find("{")
        end_idx = json_str.rfind("}") + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = json_str[start_idx:end_idx]
        
        try:
            data = json.loads(json_str)
            
            # Handle nested response formats
            if "structured_output" in data:
                data = data["structured_output"]
            elif "result" in data and data["result"]:
                data = data["result"]
            
            # Normalize classification string (handle spaces)
            classification_str = data.get("classification", "").replace(" ", "_").upper()
            
            # Map to enum
            classification_map = {
                "HARNESS_ERROR": Classification.HARNESS_ERROR,
                "GOOD_FAILURE": Classification.GOOD_FAILURE,
                "BAD_FAILURE": Classification.BAD_FAILURE,
                "GOOD_SUCCESS": Classification.GOOD_SUCCESS,
                "BAD_SUCCESS": Classification.BAD_SUCCESS,
            }
            
            classification = classification_map.get(
                classification_str,
                Classification.HARNESS_ERROR
            )
            
            return TrialClassification(
                trial_name=trial_name,
                classification=classification,
                subtype=data.get("subtype", "Unknown"),
                evidence=data.get("evidence", ""),
                root_cause=data.get("root_cause", ""),
                recommendation=data.get("recommendation", ""),
                reward=reward,
            )
            
        except json.JSONDecodeError:
            # Couldn't parse JSON - create error classification
            return TrialClassification(
                trial_name=trial_name,
                classification=Classification.HARNESS_ERROR,
                subtype="Parse Error",
                evidence=f"Could not parse JSON from response: {response_text[:500]}",
                root_cause="Claude's response was not valid JSON",
                recommendation="Review trial manually",
                reward=reward,
            )
    
    def classify_trial_sync(
        self,
        trial_dir: Path,
        task_dir: Path,
    ) -> TrialClassification:
        """Synchronous wrapper for classify_trial."""
        return asyncio.run(self.classify_trial(trial_dir, task_dir))
    
    async def classify_trials(
        self,
        trial_dirs: list[Path],
        task_dir: Path,
        console: "Console | None" = None,
    ) -> list[TrialClassification]:
        """Classify multiple trials.
        
        Note: Runs sequentially to avoid overwhelming Claude Code.
        
        Args:
            trial_dirs: List of trial directories to classify
            task_dir: Path to task directory
            console: Optional console for progress output
            
        Returns:
            List of TrialClassification results
        """
        if console:
            console.print(f"  Classifying {len(trial_dirs)} trial(s) with Claude Code...")
        
        classifications = []
        for i, trial_dir in enumerate(trial_dirs):
            if console:
                console.print(f"    [{i+1}/{len(trial_dirs)}] {trial_dir.name}...")
            
            try:
                classification = await self.classify_trial(trial_dir, task_dir)
                classifications.append(classification)
            except Exception as e:
                classifications.append(TrialClassification(
                    trial_name=trial_dir.name,
                    classification=Classification.HARNESS_ERROR,
                    subtype="Classification Error",
                    evidence=str(e),
                    root_cause="Exception during classification",
                    recommendation="Review trial manually",
                    reward=None,
                ))
        
        return classifications
    
    def classify_trials_sync(
        self,
        trial_dirs: list[Path],
        task_dir: Path,
        console: "Console | None" = None,
    ) -> list[TrialClassification]:
        """Synchronous wrapper for classify_trials."""
        return asyncio.run(self.classify_trials(trial_dirs, task_dir, console))


def compute_task_verdict(
    classifications: list[TrialClassification],
    baseline: BaselineValidation | None = None,
    quality_check_passed: bool = True,
) -> TaskVerdict:
    """Compute overall task verdict from trial classifications.
    
    Args:
        classifications: List of trial classifications
        baseline: Optional baseline validation results
        quality_check_passed: Whether static quality check passed
        
    Returns:
        TaskVerdict with is_good, confidence, and recommendations
    """
    if not classifications:
        return TaskVerdict(
            is_good=False,
            confidence="low",
            primary_issue="No trials to analyze",
            recommendations=["Run agent trials first"],
        )
    
    # Count by category
    task_problems = [c for c in classifications if c.is_task_problem]
    agent_problems = [c for c in classifications if c.classification == Classification.GOOD_FAILURE]
    successes = [c for c in classifications if c.classification == Classification.GOOD_SUCCESS]
    bad_successes = [c for c in classifications if c.classification == Classification.BAD_SUCCESS]
    harness_errors = [c for c in classifications if c.classification == Classification.HARNESS_ERROR]
    
    # Collect unique recommendations from task problems
    recommendations = []
    seen_recs = set()
    for c in task_problems:
        if c.recommendation and c.recommendation != "N/A - task is fine":
            rec = c.recommendation.strip()
            if rec not in seen_recs:
                recommendations.append(rec)
                seen_recs.add(rec)
    
    # Determine verdict
    is_good = True
    primary_issue = None
    confidence = "high"
    
    # Check baseline first
    if baseline and not baseline.is_valid:
        is_good = False
        primary_issue = baseline.issues[0] if baseline.issues else "Baseline validation failed"
        recommendations = baseline.issues + recommendations
    
    # Check for task problems
    elif len(task_problems) > 0:
        is_good = False
        # Find most common subtype
        subtype_counts: dict[str, int] = {}
        for c in task_problems:
            subtype_counts[c.subtype] = subtype_counts.get(c.subtype, 0) + 1
        most_common = max(subtype_counts, key=lambda k: subtype_counts[k])
        primary_issue = f"{len(task_problems)}/{len(classifications)} trials indicate task problem: {most_common}"
        
        # Confidence based on consistency
        if len(task_problems) == len(classifications):
            confidence = "high"
        elif len(task_problems) > len(classifications) / 2:
            confidence = "medium"
        else:
            confidence = "low"
    
    # Check for bad successes (cheating)
    elif len(bad_successes) > 0:
        is_good = False
        primary_issue = f"{len(bad_successes)} trial(s) show potential cheating or over-permissive tests"
        confidence = "medium"
    
    # Check success rate
    elif len(successes) == 0 and len(agent_problems) > 0:
        # All failures but classified as agent problems
        is_good = True  # Task might be fine, just hard
        primary_issue = None
        confidence = "medium"  # Lower confidence since no successes
        recommendations.append("Consider if task difficulty is appropriate - no successes in trials")
    
    # Static quality check
    if not quality_check_passed and is_good:
        is_good = False
        primary_issue = primary_issue or "Static quality check failed"
        confidence = "medium"
    
    return TaskVerdict(
        is_good=is_good,
        confidence=confidence,
        primary_issue=primary_issue,
        recommendations=recommendations[:5],  # Top 5 recommendations
        task_problem_count=len(task_problems),
        agent_problem_count=len(agent_problems),
        success_count=len(successes) + len(bad_successes),
        harness_error_count=len(harness_errors),
        classifications=classifications,
        baseline=baseline,
    )


def classify_baseline_result(
    agent: str,
    reward: float | None,
    error: str | None = None,
) -> BaselineResult:
    """Create a BaselineResult from agent run outcome.
    
    Args:
        agent: "nop" or "oracle"
        reward: Reward value (1.0 = pass, 0.0 = fail)
        error: Optional error message if agent failed to run
        
    Returns:
        BaselineResult with pass/fail status
    """
    passed = reward == 1.0 if reward is not None else False
    return BaselineResult(
        agent=agent,  # type: ignore
        passed=passed,
        reward=reward,
        error=error,
    )
