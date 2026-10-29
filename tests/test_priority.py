"""
Tests for Transparent Priority Rules (Heuristic Triage).
"""

from model.priority import evaluate_priority


def test_critical_outage_keyword_triggers_urgent():
    """Major system outages should unconditionally trigger Urgent priority."""
    feedback = "Our production server is down and our system is suffering a total outage!"
    assessment = evaluate_priority(feedback, sentiment="negative", confidence=0.85)

    assert assessment["priority"] == "Urgent"
    assert assessment["is_urgent"] is True
    assert any("outage" in kw for kw in assessment["matched_keywords"])
    assert "Transparent Heuristic" in assessment["rule_type"]


def test_legal_and_fraud_keywords_trigger_urgent():
    """Legal threats and fraudulent chargebacks must trigger Urgent priority."""
    feedback = "I will contact my lawyer and initiate a chargeback for this fraud."
    assessment = evaluate_priority(feedback, sentiment="negative", confidence=0.75)

    assert assessment["priority"] == "Urgent"
    assert assessment["is_urgent"] is True
    assert "chargeback" in assessment["matched_keywords"] or "lawyer" in assessment["matched_keywords"]
    assert len(assessment["reasons"]) > 0


def test_escalation_keyword_with_neutral_triggers_urgent():
    """Urgent time-sensitive requests should escalate to human review even if neutral."""
    feedback = "We have a critical deadline this afternoon, please reply ASAP."
    assessment = evaluate_priority(feedback, sentiment="neutral", confidence=0.60)

    assert assessment["priority"] == "Urgent"
    assert assessment["is_urgent"] is True
    assert "asap" in assessment["matched_keywords"] or "critical" in assessment["matched_keywords"]


def test_normal_support_inquiry_triggers_normal():
    """Routine how-to questions should be classified as Normal priority."""
    feedback = "How can I update my billing email and download last month's invoice?"
    assessment = evaluate_priority(feedback, sentiment="neutral", confidence=0.80)

    assert assessment["priority"] == "Normal"
    assert assessment["is_urgent"] is False
    assert "Standard Queue" in assessment["suggested_action"]


def test_positive_praise_triggers_normal():
    """Compliments and positive feedback should not be flagged as urgent."""
    feedback = "Great tool, everything works seamlessly! Keep up the good work."
    assessment = evaluate_priority(feedback, sentiment="positive", confidence=0.92)

    assert assessment["priority"] == "Normal"
    assert assessment["is_urgent"] is False


def test_rule_transparency_metadata():
    """Ensure rule_type explicitly states this is a heuristic rule and NOT an ML model."""
    feedback = "Please help me cancel subscription."
    assessment = evaluate_priority(feedback, sentiment="negative", confidence=0.75)

    assert assessment["priority"] == "Urgent"
    assert assessment["rule_type"] == "Transparent Heuristic Rules (Not an ML Urgency Model)"
    assert "Escalate to Human Agent" in assessment["suggested_action"]
