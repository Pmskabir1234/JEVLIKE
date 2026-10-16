import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from baseline.baseline01 import TicketDecision
from data.sample_data import (
    RAW_DATA,
    TicketExample,
    get_dataset,
    load_dataset_from_json,
    save_dataset_to_json,
)


def test_dataset_size():
    assert len(RAW_DATA) == 200
    dataset = get_dataset()
    assert len(dataset) == 200


def test_unique_sequential_ids():
    ids = [item["id"] for item in RAW_DATA]
    assert ids == list(range(1, 201))
    assert len(set(ids)) == 200


def test_all_tickets_non_empty_and_strip():
    for item in RAW_DATA:
        ticket = item["ticket"]
        assert isinstance(ticket, str)
        assert len(ticket.strip()) > 0, f"Ticket ID {item['id']} is empty after strip."


def test_all_examples_conform_to_ticket_decision():
    for item in RAW_DATA:
        # Validate that TicketDecision strictly accepts the fields
        decision = TicketDecision(
            department=item["department"],
            urgency=item["urgency"],
            requires_human_review=item["requires_human_review"],
        )
        assert decision.department in ["billing", "technical", "sales", "general", "unknown"]
        assert decision.urgency in ["low", "medium", "high", "unknown"]
        assert isinstance(decision.requires_human_review, bool)


def test_json_file_matches_raw_data():
    json_path = Path(__file__).parent / "sample_data.json"
    assert json_path.exists(), "sample_data.json does not exist."

    with open(json_path, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)

    assert len(loaded_json) == 200
    assert loaded_json == RAW_DATA


def test_load_dataset_from_json():
    loaded = load_dataset_from_json()
    assert len(loaded) == 200
    for ex in loaded:
        assert isinstance(ex, TicketExample)
        decision = ex.to_decision()
        assert isinstance(decision, TicketDecision)


def test_edge_cases_included():
    # Prompt injections
    prompt_injections = [ex for ex in RAW_DATA if "SYSTEM OVERRIDE" in ex["ticket"] or "system prompt" in ex["ticket"]]
    assert len(prompt_injections) >= 2
    for pi in prompt_injections:
        assert pi["requires_human_review"] is True
        assert pi["department"] == "unknown"

    # Outages
    outages = [ex for ex in RAW_DATA if "PRODUCTION OUTAGE" in ex["ticket"] or "cluster is down" in ex["ticket"]]
    assert len(outages) >= 1
    assert outages[0]["department"] == "technical"
    assert outages[0]["urgency"] == "high"

    # Financial disputes
    disputes = [ex for ex in RAW_DATA if "chargeback" in ex["ticket"].lower() or "unauthorized charge" in ex["ticket"].lower()]
    assert len(disputes) >= 2
    for d in disputes:
        assert d["department"] == "billing"
        assert d["requires_human_review"] is True


def test_department_distribution():
    from collections import Counter
    counts = Counter(item["department"] for item in RAW_DATA)
    assert counts["billing"] == 45
    assert counts["technical"] == 50
    assert counts["sales"] == 35
    assert counts["general"] == 35
    assert counts["unknown"] == 35
