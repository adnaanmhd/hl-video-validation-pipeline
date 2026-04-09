"""Shared types for check results."""

from dataclasses import dataclass


@dataclass
class CheckResult:
    """Result of a single acceptance criteria check.

    Returned by every check function.
    """
    status: str  # "pass", "fail", "review", or "skipped"
    metric_value: float  # Raw measurement (e.g., % of frames passing)
    confidence: float  # 0.0-1.0 confidence in the result
    details: dict | None = None  # Optional extra info for debugging
