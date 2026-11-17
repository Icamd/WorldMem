from worldmem.evaluation.metrics import evaluate_results
from worldmem.evaluation.perltqa_trimem import calculate_trimem_metrics


def test_preserved_metric_protocols_have_expected_bounds_and_categories():
    report = evaluate_results(
        {"owner": [{"answer": "red blue", "response": "red red blue", "category": "Profile"}]}
    )
    assert report["overall"]["count"] == 1
    assert report["Profile"]["f1"] == 1.0  # The upstream scorer uses token sets.
    assert report["Profile"]["exact_match"] == 0
    assert 0 <= report["Profile"]["bleu1"] <= 1
    assert 0 <= report["Profile"]["rouge_l_score"] <= 1
    trimem = calculate_trimem_metrics("Alice visited Paris.", "Alice visited Paris.")
    assert trimem["f1"] == 1.0
    assert trimem["rougeL_f"] == 1.0


def test_empty_prediction_scores_zero():
    report = evaluate_results({"owner": [{"answer": "Paris", "response": "", "category": 4}]})
    assert report["overall"]["f1"] == 0
    assert report["overall"]["rouge_l_score"] == 0
