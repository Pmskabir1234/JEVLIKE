import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


def calculate_percentile(sorted_values: List[float], percentile: float) -> float:
    """Calculate the percentile of a sorted list of numerical values using linear interpolation."""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (percentile / 100.0)
    floor_k = math.floor(k)
    ceil_k = math.ceil(k)
    if floor_k == ceil_k:
        return float(sorted_values[int(k)])
    d0 = sorted_values[floor_k] * (ceil_k - k)
    d1 = sorted_values[ceil_k] * (k - floor_k)
    return float(d0 + d1)


def calculate_classification_metrics(y_true: List[Any], y_pred: List[Any]) -> Dict[str, Any]:
    """Compute accuracy, macro F1, weighted F1, and per-class precision/recall/F1/support."""
    classes = sorted(list(set(y_true) | set(y_pred)), key=lambda x: str(x))
    f1_per_class: Dict[str, float] = {}
    precision_per_class: Dict[str, float] = {}
    recall_per_class: Dict[str, float] = {}
    support_per_class: Dict[str, int] = {}

    for c in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        key = str(c)
        f1_per_class[key] = round(f1, 4)
        precision_per_class[key] = round(precision, 4)
        recall_per_class[key] = round(recall, 4)
        support_per_class[key] = sum(1 for yt in y_true if yt == c)

    macro_f1 = sum(f1_per_class.values()) / len(classes) if classes else 0.0
    total = len(y_true)
    weighted_f1 = (
        sum(f1_per_class[str(c)] * support_per_class[str(c)] for c in classes) / total
        if total > 0
        else 0.0
    )
    accuracy = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / total if total > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "f1_macro": round(macro_f1, 4),
        "f1_weighted": round(weighted_f1, 4),
        "per_class": {
            k: {
                "precision": precision_per_class[k],
                "recall": recall_per_class[k],
                "f1": f1_per_class[k],
                "support": support_per_class[k],
            }
            for k in f1_per_class
        },
    }


def evaluate_dataset(
    samples: List[Dict[str, Any]],
    model_fn: Optional[Callable[[str], Tuple[Any, float]]] = None,
    model_name: str = "qwen3:0.6b",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Evaluate ticket samples using the model function and calculate metrics."""
    if model_fn is None:
        try:
            from baseline.baseline01 import classify_ticket
            model_fn = classify_ticket
        except ImportError:
            try:
                from baseline.baseline01 import classify_ticket
                model_fn = classify_ticket
            except ImportError:
                raise ValueError("model_fn must be provided when evaluate_dataset is called.")

    evaluation_records: List[Dict[str, Any]] = []
    latencies: List[float] = []

    y_true_dep: List[str] = []
    y_pred_dep: List[str] = []

    y_true_urg: List[str] = []
    y_pred_urg: List[str] = []

    y_true_rev: List[bool] = []
    y_pred_rev: List[bool] = []

    total_samples = len(samples)
    successful_count = 0
    failed_count = 0

    for idx, item in enumerate(samples, start=1):
        ticket_id = item.get("id", idx)
        ticket_text = item.get("ticket", "")
        ground_truth = {
            "department": item.get("department"),
            "urgency": item.get("urgency"),
            "requires_human_review": item.get("requires_human_review"),
        }

        try:
            decision, latency_ms = model_fn(ticket_text)
            if hasattr(decision, "department"):
                dep = decision.department
                urg = decision.urgency
                rev = decision.requires_human_review
            elif isinstance(decision, dict):
                dep = decision.get("department", "unknown")
                urg = decision.get("urgency", "unknown")
                rev = decision.get("requires_human_review", True)
            else:
                dep = getattr(decision, "department", "unknown")
                urg = getattr(decision, "urgency", "unknown")
                rev = getattr(decision, "requires_human_review", True)

            pred_dict = {
                "department": dep,
                "urgency": urg,
                "requires_human_review": rev,
            }
            status = "success"
            err_msg = None
            latencies.append(latency_ms)
            successful_count += 1
            print(
                f"[{idx}/{total_samples}] Ticket #{ticket_id}: "
                f"dep={pred_dict['department']}, urg={pred_dict['urgency']}, "
                f"review={pred_dict['requires_human_review']} ({latency_ms:.1f}ms)"
            )
        except Exception as exc:
            pred_dict = {
                "department": "unknown",
                "urgency": "unknown",
                "requires_human_review": True,
            }
            latency_ms = 0.0
            status = "error"
            err_msg = str(exc)
            failed_count += 1
            print(f"[{idx}/{total_samples}] Ticket #{ticket_id} ERROR: {exc}")

        matches = {
            "department": pred_dict["department"] == ground_truth["department"],
            "urgency": pred_dict["urgency"] == ground_truth["urgency"],
            "requires_human_review": (
                pred_dict["requires_human_review"] == ground_truth["requires_human_review"]
            ),
            "all": (
                pred_dict["department"] == ground_truth["department"]
                and pred_dict["urgency"] == ground_truth["urgency"]
                and pred_dict["requires_human_review"] == ground_truth["requires_human_review"]
            ),
        }

        record = {
            "id": ticket_id,
            "ticket": ticket_text,
            "ground_truth": ground_truth,
            "prediction": pred_dict,
            "latency_ms": round(latency_ms, 2),
            "status": status,
            "error": err_msg,
            "matches": matches,
        }
        evaluation_records.append(record)

        y_true_dep.append(ground_truth["department"])
        y_pred_dep.append(pred_dict["department"])

        y_true_urg.append(ground_truth["urgency"])
        y_pred_urg.append(pred_dict["urgency"])

        y_true_rev.append(ground_truth["requires_human_review"])
        y_pred_rev.append(pred_dict["requires_human_review"])

    # Calculate metrics
    dep_metrics = calculate_classification_metrics(y_true_dep, y_pred_dep)
    urg_metrics = calculate_classification_metrics(y_true_urg, y_pred_urg)
    rev_metrics = calculate_classification_metrics(y_true_rev, y_pred_rev)

    exact_match_accuracy = (
        sum(1 for r in evaluation_records if r["matches"]["all"]) / total_samples
        if total_samples > 0
        else 0.0
    )
    mean_field_accuracy = (
        (dep_metrics["accuracy"] + urg_metrics["accuracy"] + rev_metrics["accuracy"]) / 3.0
        if total_samples > 0
        else 0.0
    )
    overall_f1 = (
        (dep_metrics["f1_macro"] + urg_metrics["f1_macro"] + rev_metrics["f1_macro"]) / 3.0
        if total_samples > 0
        else 0.0
    )

    sorted_latencies = sorted(latencies)
    latency_p50 = round(calculate_percentile(sorted_latencies, 50), 2)
    latency_p90 = round(calculate_percentile(sorted_latencies, 90), 2)
    latency_p95 = round(calculate_percentile(sorted_latencies, 95), 2)
    latency_mean = (
        round(statistics.mean(sorted_latencies), 2) if sorted_latencies else 0.0
    )
    latency_min = round(min(sorted_latencies), 2) if sorted_latencies else 0.0
    latency_max = round(max(sorted_latencies), 2) if sorted_latencies else 0.0

    benchmark_data = {
        "model": model_name,
        "accuracy": round(exact_match_accuracy, 4),
        "f1": round(overall_f1, 4),
        "latency": {
            "p50": latency_p50,
            "p90": latency_p90,
            "p95": latency_p95,
        },
        "exact_match_accuracy": round(exact_match_accuracy, 4),
        "mean_field_accuracy": round(mean_field_accuracy, 4),
        "f1_macro_average": round(overall_f1, 4),
        "latency_metrics": {
            "p50": latency_p50,
            "p90": latency_p90,
            "p95": latency_p95,
            "mean": latency_mean,
            "min": latency_min,
            "max": latency_max,
            "unit": "ms",
        },
        "by_field": {
            "department": dep_metrics,
            "urgency": urg_metrics,
            "requires_human_review": rev_metrics,
        },
        "summary": {
            "total_samples": total_samples,
            "successful_samples": successful_count,
            "failed_samples": failed_count,
        },
    }

    return evaluation_records, benchmark_data


def save_evaluation_results(
    records: List[Dict[str, Any]],
    benchmark_data: Dict[str, Any],
    eval_file: Union[str, Path],
    benchmark_file: Union[str, Path],
    secondary_benchmark_file: Optional[Union[str, Path]] = None,
) -> None:
    """Save evaluation records and benchmark metrics to JSON files."""
    eval_path = Path(eval_file)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Saved predictions to: {eval_path}")

    bench_path = Path(benchmark_file)
    bench_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"Saved benchmark results to: {bench_path}")

    if secondary_benchmark_file:
        sec_path = Path(secondary_benchmark_file)
        sec_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sec_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_data, f, indent=2)


def print_benchmark_summary(benchmark_results: Dict[str, Any]) -> None:
    """Print a clean benchmark summary to the console."""
    print("\n" + "=" * 60)
    print("EVALUATION & BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Model:                   {benchmark_results.get('model', 'N/A')}")
    print(f"Total Samples:           {benchmark_results['summary']['total_samples']}")
    print(f"Successful:              {benchmark_results['summary']['successful_samples']}")
    print(f"Failed:                  {benchmark_results['summary']['failed_samples']}")
    print(f"Exact Match Accuracy:    {benchmark_results['accuracy'] * 100:.2f}%")
    print(f"Macro F1:                {benchmark_results['f1']:.4f}")
    print(f"Latency p50:             {benchmark_results['latency']['p50']:.1f} ms")
    print(f"Latency p90:             {benchmark_results['latency']['p90']:.1f} ms")
    print(f"Latency p95:             {benchmark_results['latency']['p95']:.1f} ms")
    print("=" * 60)
