import pytest
from model_evaluation.evalaution import (
    calculate_percentile,
    calculate_classification_metrics,
    evaluate_dataset,
    save_evaluation_results,
    print_benchmark_summary,
)
from model_evaluation.evaluation import (
    calculate_percentile as calc_pct_alias,
    calculate_classification_metrics as calc_metrics_alias,
    evaluate_dataset as eval_dataset_alias,
)


def test_percentile_calculation():
    assert calculate_percentile([], 50) == 0.0
    assert calculate_percentile([42.0], 50) == 42.0
    assert calculate_percentile([10.0, 20.0, 30.0], 50) == 20.0
    assert calc_pct_alias([10.0, 20.0, 30.0], 50) == 20.0


def test_classification_metrics():
    y_true = ["billing", "billing", "sales", "technical"]
    y_pred = ["billing", "sales", "sales", "technical"]

    res = calculate_classification_metrics(y_true, y_pred)
    assert res["accuracy"] == 0.75
    assert "sales" in res["per_class"]
    assert "f1_macro" in res
    assert "f1_weighted" in res

    res_alias = calc_metrics_alias(y_true, y_pred)
    assert res_alias["accuracy"] == res["accuracy"]


def test_evaluate_dataset_mock(tmp_path):
    class MockDecision:
        def __init__(self, dep, urg, rev):
            self.department = dep
            self.urgency = urg
            self.requires_human_review = rev

    def mock_model(text: str):
        return MockDecision("billing", "low", False), 25.5

    samples = [
        {
            "id": 1,
            "ticket": "Need receipt",
            "department": "billing",
            "urgency": "low",
            "requires_human_review": False,
        }
    ]

    records, benchmark = evaluate_dataset(samples, model_fn=mock_model, model_name="test-model")
    assert len(records) == 1
    assert records[0]["matches"]["all"] is True
    assert benchmark["accuracy"] == 1.0
    assert benchmark["model"] == "test-model"
    assert benchmark["latency"]["p50"] == 25.5

    # Test saving results
    eval_file = tmp_path / "eval.json"
    bench_file = tmp_path / "bench.json"
    sec_bench_file = tmp_path / "sec_bench.json"

    save_evaluation_results(records, benchmark, eval_file, bench_file, sec_bench_file)
    assert eval_file.exists()
    assert bench_file.exists()
    assert sec_bench_file.exists()

    # Test summary printing doesn't crash
    print_benchmark_summary(benchmark)
