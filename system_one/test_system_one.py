import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from system_one_local import (
    DEMO_REQUEST,
    ChoiceQuestion,
    DecisionRequest,
    NoulQuestion,
    ScoreQuestion,
    build_question_prompt,
    evaluate_request,
    output_model_for,
)


class FakeClient:
    """Returns deterministic answers and tracks concurrent model calls."""

    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.prompts = []

    async def chat(self, *, model, messages, format, options):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.prompts.append(messages[1]["content"])
        try:
            await asyncio.sleep(0.03)
            keys = format["properties"].keys()
            if "choice" in keys:
                payload = {"choice": "technical"}
            elif "level_index" in keys:
                payload = {"level_index": 3}
            else:
                payload = {"noul": 0.91}
            return SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(payload))
            )
        finally:
            self.active -= 1


def test_choice_schema_only_accepts_declared_options():
    question = ChoiceQuestion(
        type="choice",
        instructions="Choose a team.",
        criteria={"billing": "Payment issues", "technical": "Outages"},
    )
    schema = output_model_for(question)
    assert schema.model_validate({"choice": "billing"}).choice == "billing"
    with pytest.raises(ValidationError):
        schema.model_validate({"choice": "finance"})


def test_score_schema_enforces_valid_level_index():
    question = ScoreQuestion(
        type="score",
        instructions="Rate impact.",
        criteria=["low", "medium", "high"],
    )
    schema = output_model_for(question)
    assert schema.model_validate({"level_index": 2}).level_index == 2
    with pytest.raises(ValidationError):
        schema.model_validate({"level_index": 3})


def test_noul_schema_requires_value_between_zero_and_one():
    question = NoulQuestion(
        type="noul", instructions="Is human intervention needed?"
    )
    schema = output_model_for(question)
    assert schema.model_validate({"noul": 0.7}).noul == 0.7
    with pytest.raises(ValidationError):
        schema.model_validate({"noul": 1.2})


def test_request_contract_validates_discriminated_question_types():
    request = DecisionRequest.model_validate(DEMO_REQUEST)
    assert isinstance(request.questions["team"], ChoiceQuestion)
    assert isinstance(request.questions["severity"], ScoreQuestion)
    assert isinstance(request.questions["needs_human"], NoulQuestion)


def test_same_state_is_in_each_independent_question_prompt():
    request = DecisionRequest.model_validate(DEMO_REQUEST)
    state_token = "SUP-1042"
    prompts = [
        build_question_prompt(request.state, question)
        for question in request.questions.values()
    ]
    assert all(state_token in prompt for prompt in prompts)


def test_questions_run_concurrently_and_return_typed_answers():
    async def run_test():
        fake = FakeClient()
        result = await evaluate_request(DEMO_REQUEST, client=fake, max_concurrency=3)
        assert fake.max_active == 3
        assert result["answers"]["team"] == {
            "type": "choice", "choice": "technical"
        }
        assert result["answers"]["severity"]["level_index"] == 3
        assert result["answers"]["severity"]["level"].startswith("Critical")
        assert result["answers"]["needs_human"]["noul_estimate"] == 0.91
        assert result["answers"]["needs_human"]["calibration"] == (
            "uncalibrated_model_reported_estimate"
        )
        assert result["metadata"]["shared_state"] is True

    asyncio.run(run_test())


def test_bad_concurrency_is_rejected():
    with pytest.raises(ValueError):
        asyncio.run(evaluate_request(DEMO_REQUEST, max_concurrency=0, client=FakeClient()))
