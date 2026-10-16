import pytest
from pydantic import ValidationError

from baseline01 import TicketDecision

def test_valid_decision_is_accepted():
    decision = TicketDecision(
        department="billing",
        urgency="high",
        requires_human_review=True,
    )

    assert decision.department == "billing"
    assert decision.urgency == "high"
    assert decision.requires_human_review is True


def test_unknown_is_a_valid_fallback():
    decision = TicketDecision(
        department="unknown",
        urgency="unknown",
        requires_human_review=True,
    )

    assert decision.department == "unknown"


def test_invalid_department_is_rejected():
    with pytest.raises(ValidationError):
        TicketDecision(
            department="cryptocurrency",
            urgency="high",
            requires_human_review=True,
        )


def test_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        TicketDecision.model_validate(
            {
                "department": "billing",
                "urgency": "high",
                "requires_human_review": True,
                "execute_refund": True,
            }
        )


def test_empty_ticket_is_rejected():
    from baseline01 import classify_ticket

    with pytest.raises(ValueError):
        classify_ticket("   ")


def test_calculate_percentile():
    from baseline01 import calculate_percentile

    assert calculate_percentile([], 50) == 0.0
    assert calculate_percentile([100.0], 50) == 100.0

    vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert calculate_percentile(vals, 50) == 30.0
    assert calculate_percentile(vals, 90) == 46.0
    assert calculate_percentile(vals, 95) == 48.0


def test_calculate_classification_metrics():
    from baseline01 import calculate_classification_metrics

    y_true = ["billing", "billing", "sales", "technical"]
    y_pred = ["billing", "technical", "sales", "technical"]

    metrics = calculate_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.75
    assert metrics["per_class"]["sales"]["precision"] == 1.0
    assert metrics["per_class"]["sales"]["recall"] == 1.0
    assert metrics["per_class"]["sales"]["f1"] == 1.0
    assert metrics["per_class"]["billing"]["f1"] == 0.6667
    assert metrics["f1_macro"] == pytest.approx(0.7778, abs=1e-4)


def test_evaluate_dataset_with_mock_model():
    from baseline01 import evaluate_dataset

    samples = [
        {
            "id": 1,
            "ticket": "Billing inquiry",
            "department": "billing",
            "urgency": "low",
            "requires_human_review": False,
        },
        {
            "id": 2,
            "ticket": "System crash",
            "department": "technical",
            "urgency": "high",
            "requires_human_review": True,
        },
    ]

    def mock_classify(ticket: str):
        if "Billing" in ticket:
            return TicketDecision(department="billing", urgency="low", requires_human_review=False), 50.0
        return TicketDecision(department="technical", urgency="high", requires_human_review=True), 150.0

    records, benchmark = evaluate_dataset(samples, model_fn=mock_classify)

    assert len(records) == 2
    assert records[0]["matches"]["all"] is True
    assert records[1]["matches"]["all"] is True
    assert benchmark["accuracy"] == 1.0
    assert benchmark["f1"] == 1.0
    assert "p50" in benchmark["latency"]
    assert "p90" in benchmark["latency"]
    assert "p95" in benchmark["latency"]
    assert benchmark["latency"]["p50"] == 100.0


