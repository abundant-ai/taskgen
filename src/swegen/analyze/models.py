from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Classification(str, Enum):
    """Top-level classification of a trial outcome.
    
    The classification indicates whether the outcome reveals a problem
    with the task (BAD_*) or is expected behavior (GOOD_*, HARNESS_ERROR).
    """
    
    # Infrastructure problem - agent never ran
    HARNESS_ERROR = "HARNESS_ERROR"
    
    # Agent ran but failed - task is fine, agent couldn't solve it
    GOOD_FAILURE = "GOOD_FAILURE"
    
    # Agent failed due to task issues - task needs fixing
    BAD_FAILURE = "BAD_FAILURE"
    
    # Agent solved it legitimately - task is working
    GOOD_SUCCESS = "GOOD_SUCCESS"
    
    # Agent "solved" it by cheating or task is broken - task needs fixing
    BAD_SUCCESS = "BAD_SUCCESS"
    
    @property
    def is_task_problem(self) -> bool:
        """Returns True if this classification indicates a task issue."""
        return self in (Classification.BAD_FAILURE, Classification.BAD_SUCCESS)
    
    @property
    def is_success(self) -> bool:
        """Returns True if tests passed."""
        return self in (Classification.GOOD_SUCCESS, Classification.BAD_SUCCESS)


class Subtype(str, Enum):
    """Detailed subtype explaining the classification.
    
    These provide actionable information about what specifically
    caused the outcome.
    """
    
    # HARNESS_ERROR subtypes
    AGENT_NOT_FOUND = "Agent Not Found"
    CONTAINER_FAILURE = "Container/Docker Failure"
    MISSING_DEPENDENCIES = "Missing Dependencies"
    EMPTY_TRAJECTORY = "Empty Trajectory"
    INFRASTRUCTURE_ERROR = "Infrastructure Error"
    
    # GOOD_FAILURE subtypes (agent's fault)
    TIMEOUT = "Timeout"
    WRONG_APPROACH = "Wrong Approach"
    IMPLEMENTATION_BUGS = "Implementation Bugs"
    CONTEXT_LOSS = "Context Loss"
    PREMATURE_STOP = "Premature Stop"
    COMPLEXITY_OVERWHELM = "Complexity Overwhelm"
    INCOMPLETE_SOLUTION = "Incomplete Solution"
    LOGIC_ERROR = "Logic Error"
    
    # BAD_FAILURE subtypes (task's fault)
    UNDERSPECIFIED_INSTRUCTION = "Underspecified Instruction"
    RIGID_BRITTLE_TESTS = "Rigid/Brittle Tests"
    NONDETERMINISTIC_TESTS = "Non-deterministic Tests"
    ENVIRONMENT_ISSUES = "Environment Issues"
    MISSING_FILE_REFERENCE = "Missing File Reference"
    AMBIGUOUS_REQUIREMENTS = "Ambiguous Requirements"
    IMPLEMENTATION_DETAILS_REQUIRED = "Implementation Details Required"
    EDGE_CASES_NOT_SPECIFIED = "Edge Cases Not Specified"
    TEST_EXPECTS_SPECIFIC_FORMAT = "Test Expects Specific Format"
    
    # GOOD_SUCCESS subtypes
    CORRECT_SOLUTION = "Correct Solution"
    ALTERNATIVE_VALID_SOLUTION = "Alternative Valid Solution"
    
    # BAD_SUCCESS subtypes (cheating/gaming)
    HARDCODING = "Hardcoding"
    TEST_INSPECTION = "Test Inspection"
    ORACLE_COPYING = "Oracle Copying"
    MINIMAL_COMPLIANCE = "Minimal Compliance"
    TESTS_TOO_PERMISSIVE = "Tests Too Permissive"
    TASK_PRE_SOLVED = "Task Pre-solved"


class TrialClassificationModel(BaseModel):
    """Pydantic model for LLM structured output."""
    
    classification: Literal[
        "HARNESS_ERROR", "GOOD_FAILURE", "BAD_FAILURE", "GOOD_SUCCESS", "BAD_SUCCESS"
    ] = Field(description="Top-level classification")
    
    subtype: str = Field(
        description="Specific subtype from the taxonomy (e.g., 'Timeout', 'Underspecified Instruction')"
    )
    
    evidence: str = Field(
        description="Specific evidence from files: test names, error messages, code snippets"
    )
    
    root_cause: str = Field(
        description="1-2 sentence explanation of what caused this outcome"
    )
    
    recommendation: str = Field(
        description="How to fix the task (if BAD_FAILURE or BAD_SUCCESS), or 'N/A' if task is fine"
    )


class TaskVerdictModel(BaseModel):
    """Pydantic model for LLM structured output for the overall task verdict."""

    is_good: bool = Field(description="Whether the task is good (true) or needs review (false)")
    confidence: Literal["high", "medium", "low"] = Field(description="Confidence level")
    primary_issue: str | None = Field(
        default=None, description="Primary issue if task needs review, else null"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Actionable recommendations (3-5 for bad tasks)"
    )
    reasoning: str | None = Field(
        default=None, description="1-2 sentence explanation of the verdict (optional)"
    )


@dataclass
class TrialClassification:
    """Classification result for a single trial.
    
    This captures why a trial succeeded or failed, and whether
    the outcome indicates a task problem that needs fixing.
    """
    
    trial_name: str
    classification: Classification
    subtype: str
    evidence: str
    root_cause: str
    recommendation: str
    
    # Derived from verifier
    reward: float | None = None
    
    @property
    def is_task_problem(self) -> bool:
        """Returns True if this trial reveals a task issue."""
        return self.classification.is_task_problem
    
    @classmethod
    def from_model(cls, trial_name: str, model: TrialClassificationModel, reward: float | None = None) -> "TrialClassification":
        """Create from Pydantic model response."""
        return cls(
            trial_name=trial_name,
            classification=Classification(model.classification),
            subtype=model.subtype,
            evidence=model.evidence,
            root_cause=model.root_cause,
            recommendation=model.recommendation,
            reward=reward,
        )


@dataclass
class BaselineResult:
    """Result from running a baseline agent (nop or oracle)."""
    
    agent: Literal["nop", "oracle"]
    passed: bool  # reward == 1
    reward: float | None
    error: str | None = None
    
    @property
    def is_expected(self) -> bool:
        """Returns True if the result is what we expect for a good task."""
        if self.agent == "nop":
            # nop should FAIL (reward=0) - tests should require changes
            return not self.passed
        else:
            # oracle should PASS (reward=1) - reference solution should work
            return self.passed


@dataclass
class BaselineValidation:
    """Results from baseline validation (nop and oracle runs).
    
    For a well-formed task:
    - nop should FAIL (tests require actual work)
    - oracle should PASS (reference solution works)
    """
    
    nop: BaselineResult | None = None
    oracle: BaselineResult | None = None
    
    @property
    def is_valid(self) -> bool:
        """Returns True if baseline validation passes."""
        nop_ok = self.nop is None or self.nop.is_expected
        oracle_ok = self.oracle is None or self.oracle.is_expected
        return nop_ok and oracle_ok
    
    @property
    def issues(self) -> list[str]:
        """Returns list of baseline validation issues."""
        issues = []
        if self.nop and not self.nop.is_expected:
            issues.append(
                "CRITICAL: nop agent passed - task may be pre-solved or tests are broken"
            )
        if self.oracle and not self.oracle.is_expected:
            issues.append(
                "CRITICAL: oracle agent failed - reference solution doesn't work"
            )
        return issues


@dataclass
class TaskVerdict:
    """Final verdict on task quality based on all analysis.
    
    This aggregates results from:
    - Static quality checks
    - Baseline validation (nop/oracle)
    - Agent trial classifications
    """
    
    is_good: bool
    confidence: Literal["high", "medium", "low"]
    primary_issue: str | None
    recommendations: list[str] = field(default_factory=list)
    
    # Breakdown
    task_problem_count: int = 0
    agent_problem_count: int = 0
    success_count: int = 0
    harness_error_count: int = 0
    
    # From classifications
    classifications: list[TrialClassification] = field(default_factory=list)
    baseline: BaselineValidation | None = None
    
    def summary(self) -> str:
        """Return a one-line summary of the verdict."""
        if self.is_good:
            return f"✅ GOOD TASK (confidence: {self.confidence})"
        else:
            return f"❌ NEEDS REVIEW: {self.primary_issue}"

