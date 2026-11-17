import re
import statistics
from collections import defaultdict
from typing import Any, Dict, List

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def simple_tokenize(text: str) -> List[str]:
    """
    Simple offline tokenizer.
    Lowercase + split on alphanumeric word boundaries.
    Avoids dependency on nltk punkt.
    """
    text = str(text).lower().strip()
    return re.findall("\\w+", text, flags=re.UNICODE)


def calculate_bleu_scores(prediction: str, reference: str) -> Dict[str, float]:
    """
    Calculate BLEU-1/2/3/4 using a simple offline tokenizer.
    """
    pred_tokens = simple_tokenize(prediction)
    ref_tokens = [simple_tokenize(reference)]
    weights_list = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
    smooth = SmoothingFunction().method1
    scores = {}
    for n, weights in enumerate(weights_list, start=1):
        try:
            score = sentence_bleu(
                ref_tokens, pred_tokens, weights=weights, smoothing_function=smooth
            )
        except Exception:
            score = 0.0
        scores[f"bleu{n}"] = score
    return scores


def calculate_metrics(prediction: str, reference: str) -> Dict[str, float]:
    """
    Calculate the metrics currently used in evals.py:
    - exact_match
    - f1
    - bleu1/2/3/4
    """
    if not prediction or not reference:
        return {"exact_match": 0, "f1": 0.0, "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}
    prediction = str(prediction).strip()
    reference = str(reference).strip()
    exact_match = int(prediction.lower() == reference.lower())
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    common_tokens = pred_tokens & ref_tokens
    if not pred_tokens or not ref_tokens:
        f1 = 0.0
    else:
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    bleu_scores = calculate_bleu_scores(prediction, reference)
    return {"exact_match": exact_match, "f1": f1, **bleu_scores}


_ROUGE_TOKEN_RE = re.compile("[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[\\u4e00-\\u9fff]")


def _rouge_l_tokens(text: Any) -> list[str]:
    """
    Tokenize text for ROUGE-L in a lightweight, dependency-free way.
    English words/numbers are lowercased; CJK characters are treated as individual tokens.
    """
    if text is None:
        return []
    text = str(text).lower()
    return _ROUGE_TOKEN_RE.findall(text)


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Return the longest common subsequence length using O(min(m, n)) memory."""
    if not a or not b:
        return 0
    if len(b) > len(a):
        a, b = (b, a)
    prev = [0] * (len(b) + 1)
    for tok_a in a:
        cur = [0] * (len(b) + 1)
        for j, tok_b in enumerate(b, start=1):
            if tok_a == tok_b:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def calculate_rouge_l(prediction: Any, reference: Any) -> dict[str, float]:
    """
    Compute ROUGE-L using token-level longest common subsequence.

    Returns precision, recall, and F1. The `rouge_l` score reported by the evaluator
    is the F1 score, which is the most common single-number ROUGE-L summary.
    """
    pred_tokens = _rouge_l_tokens(prediction)
    ref_tokens = _rouge_l_tokens(reference)
    if not pred_tokens or not ref_tokens:
        return {"rouge_l_precision": 0.0, "rouge_l_recall": 0.0, "rouge_l_score": 0.0}
    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens) if pred_tokens else 0.0
    recall = lcs / len(ref_tokens) if ref_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"rouge_l_precision": precision, "rouge_l_recall": recall, "rouge_l_score": f1}


def evaluate_results(results):
    groups = defaultdict(list)
    for records in results.values():
        for item in records:
            prediction, reference = (str(item.get("response", "")), str(item.get("answer", "")))
            values = calculate_metrics(prediction, reference)
            values.update(calculate_rouge_l(prediction, reference))
            groups["overall"].append(values)
            groups[str(item.get("category_name") or item.get("category", "Unknown"))].append(values)
    return {
        category: dict(
            count=len(rows), **{key: statistics.mean((row[key] for row in rows)) for key in rows[0]}
        )
        for category, rows in groups.items()
        if rows
    }
