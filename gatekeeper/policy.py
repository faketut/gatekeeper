"""ALLOW / REVIEW / BLOCK policy.

Defaults are starting points from the demo scorer. Retune on the
IEEE-CIS precision-recall curve (see eval/evaluate.py) and update
README.md with the tuned values — do not ship tuned-looking numbers
that were never measured.
"""

from dataclasses import dataclass

REVIEW_THRESHOLD = 0.35
BLOCK_THRESHOLD = 0.75


@dataclass
class Decision:
    verdict: str  # ALLOW | REVIEW | BLOCK
    probability: float
    reasons: list
    contributions: dict


def _reasons(probability: float, contributions: dict) -> list:
    reasons = []
    if contributions.get("is_night"):
        reasons.append("purchase at night (23:00-05:00 UTC)")
    if contributions.get("new_device"):
        reasons.append("new device")
    if contributions.get("new_merchant"):
        reasons.append("new merchant for this buyer")
    if contributions.get("amount_tier", 0) >= 0.7:
        reasons.append("amount well above buyer's normal range")
    if contributions.get("velocity"):
        reasons.append("unusual purchase velocity")
    if not reasons:
        reasons.append("matches buyer's normal pattern")
    return reasons


def decide(probability: float, contributions: dict | None = None) -> Decision:
    contributions = contributions or {}
    if probability >= BLOCK_THRESHOLD:
        verdict = "BLOCK"
    elif probability >= REVIEW_THRESHOLD:
        verdict = "REVIEW"
    else:
        verdict = "ALLOW"
    return Decision(
        verdict=verdict,
        probability=probability,
        reasons=_reasons(probability, contributions),
        contributions=contributions,
    )
