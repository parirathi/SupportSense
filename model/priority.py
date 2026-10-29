"""
Priority Evaluation Module for SupportSense.

IMPORTANT TRANSPARENCY NOTICE:
This module uses an explicit, transparent rule-based heuristic to flag
potentially urgent messages for human review. It is NOT a trained machine
learning urgency model. Every priority decision is explained with exact
trigger reasons for full auditability and operator oversight.
"""

import re
from typing import Dict, Any, List


# Tier 1: Severe security, legal, billing fraud, or major outage terms.
# These trigger Urgent status unconditionally.
CRITICAL_KEYWORDS = [
    r"\blawyer\b",
    r"\blegal\b",
    r"\blawsuit\b",
    r"\bsue\b",
    r"\bfraud\b",
    r"\bunauthorized\b",
    r"\bchargeback\b",
    r"\bdata breach\b",
    r"\bbreach\b",
    r"\bhacked\b",
    r"\bstolen\b",
    r"\bcompromised\b",
    r"\boutage\b",
    r"\bsystem down\b",
    r"\bserver down\b",
    r"\bproduction down\b",
    r"\bproduction broken\b",
    r"\bdata loss\b",
    r"\bcorrupted\b",
    r"\bemergency\b",
]

# Tier 2: Explicit escalation or time-sensitive demands.
ESCALATION_KEYWORDS = [
    r"\burgent\b",
    r"\burgently\b",
    r"\basap\b",
    r"\bimmediate\b",
    r"\bimmediately\b",
    r"\bescalate\b",
    r"\bescalation\b",
    r"\bsupervisor\b",
    r"\bmanager right away\b",
    r"\bcritical\b",
    r"\bdeadline\b",
    r"\btime sensitive\b",
]

# Tier 3: Churn risks and high-friction customer dissatisfaction terms.
CHURN_RISK_KEYWORDS = [
    r"\bcancel subscription\b",
    r"\bcancel my renewal\b",
    r"\bcancelling\b",
    r"\bcancel account\b",
    r"\bleaving for a competitor\b",
    r"\bfull refund\b",
    r"\bdemanding refund\b",
    r"\breport this fraud\b",
    r"\bunacceptable\b",
    r"\bfurious\b",
    r"\bdisaster\b",
]


def evaluate_priority(text: str, sentiment: str, confidence: float) -> Dict[str, Any]:
    """
    Evaluates customer feedback text and sentiment confidence against transparent rules.

    Args:
        text (str): The customer feedback message.
        sentiment (str): Predicted sentiment ('negative', 'neutral', 'positive').
        confidence (float): Model confidence for the predicted sentiment (0.0 to 1.0).

    Returns:
        dict: Detailed priority assessment containing:
            - priority: 'Urgent' or 'Normal'
            - is_urgent: bool
            - reasons: list of human-readable trigger explanations
            - matched_keywords: list of matched keyword patterns
            - rule_type: Explicit descriptor confirming this is a heuristic rule
            - suggested_action: Actionable recommendation for customer support staff
    """
    reasons: List[str] = []
    matched_keywords: List[str] = []
    lowered = text.lower()

    # Rule 1: Severe / Critical Safety & Legal terms
    for pattern in CRITICAL_KEYWORDS:
        found = re.findall(pattern, lowered)
        if found:
            term = found[0].strip()
            if term not in matched_keywords:
                matched_keywords.append(term)
    if matched_keywords:
        reasons.append(
            f"Contains severe risk/downtime keyword(s): {', '.join(repr(k) for k in matched_keywords)}"
        )

    # Rule 2: Escalation & time-sensitive terms
    escalation_matches = []
    for pattern in ESCALATION_KEYWORDS:
        found = re.findall(pattern, lowered)
        if found:
            term = found[0].strip()
            if term not in escalation_matches:
                escalation_matches.append(term)
    if escalation_matches:
        reasons.append(
            f"Contains explicit escalation/time-sensitive keyword(s): {', '.join(repr(k) for k in escalation_matches)}"
        )

    # Rule 3: Churn & strong frustration terms
    churn_matches = []
    for pattern in CHURN_RISK_KEYWORDS:
        found = re.findall(pattern, lowered)
        if found:
            term = found[0].strip()
            if term not in churn_matches:
                churn_matches.append(term)
    if churn_matches:
        reasons.append(
            f"Contains customer churn risk / complaint keyword(s): {', '.join(repr(k) for k in churn_matches)}"
        )

    # Rule 4: High-confidence negative sentiment combination
    is_strong_negative = (sentiment.lower() == "negative" and confidence >= 0.70)
    if is_strong_negative:
        reasons.append(
            f"High-confidence negative sentiment ({confidence * 100:.1f}%) detected by classifier"
        )

    # Urgency decision logic:
    # - Any Critical Keyword (Rule 1) => URGENT
    # - Any Escalation Keyword (Rule 2) with negative or neutral sentiment => URGENT
    # - Any Churn Keyword (Rule 3) with negative sentiment => URGENT
    # - Strong Negative (Rule 4) AND (Escalation Keyword OR Churn Keyword) => URGENT
    is_urgent = False

    if matched_keywords:
        is_urgent = True
    elif escalation_matches and sentiment.lower() in ("negative", "neutral"):
        is_urgent = True
    elif churn_matches and sentiment.lower() == "negative":
        is_urgent = True
    elif is_strong_negative and (escalation_matches or churn_matches):
        is_urgent = True

    if is_urgent:
        priority_label = "Urgent"
        action = "Escalate to Human Agent: Route to Tier-2 priority queue for immediate review within 15 minutes."
    else:
        priority_label = "Normal"
        reasons.append("No critical urgency keywords or escalation triggers met standard threshold.")
        action = "Standard Queue: Route to general support triage according to standard SLA."

    return {
        "priority": priority_label,
        "is_urgent": is_urgent,
        "reasons": reasons,
        "matched_keywords": sorted(list(set(matched_keywords + escalation_matches + churn_matches))),
        "rule_type": "Transparent Heuristic Rules (Not an ML Urgency Model)",
        "suggested_action": action,
    }
