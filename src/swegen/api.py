"""Public API for SWE-gen.

This module defines the stable, user-facing imports for programmatic use.
"""

from swegen.analyze import (
    Classification,
    Subtype,
    TaskVerdict,
    TrialClassification,
    classify_trial,
    compute_task_verdict,
)

__all__ = [
    "Classification",
    "Subtype",
    "TaskVerdict",
    "TrialClassification",
    "classify_trial",
    "compute_task_verdict",
]
