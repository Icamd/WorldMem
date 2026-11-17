"""
Evaluate PerLTQA result files using the same metric style as the local TriMem
`test_locomo10.py`, with optional multi-threaded vLLM/OpenAI-compatible
LLM-as-judge evaluation.

This replaces the previous two-step workflow:
  1) evals_perltqa.py
  2) generate_scores_perltqa.py

Input format is compatible with the user's existing PerLTQA evaluator:
{
  "character_id_1": [
    {
      "question": "...",
      "answer": "ground-truth answer",
      "response": "model prediction",
      "category" or "category_name": "profile/events/...",
      ...
    }
  ],
  ...
}

Default input files are:
  <result-prefix>.full.json
  <result-prefix>.no_mental.json
  <result-prefix>.no_situation.json

Outputs:
  - detailed per-question metrics JSON
  - aggregate summary JSON
  - printed per-category and overall means
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import statistics
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import nltk
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
except Exception as e:
    raise RuntimeError("This evaluator requires nltk for TriMem-style BLEU.") from e
try:
    from rouge_score import rouge_scorer
except Exception as e:
    raise RuntimeError("This evaluator requires rouge_score for TriMem-style ROUGE-L.") from e
try:
    from tqdm import tqdm
except Exception:
    tqdm = None
try:
    from openai import OpenAI
except Exception:
    OpenAI = None
CATEGORY_NAME_MAP = {
    "profile": "Profile",
    "Profile": "Profile",
    "social_relationship": "Social Relationship",
    "social relationship": "Social Relationship",
    "Social Relationship": "Social Relationship",
    "events": "Events",
    "event": "Events",
    "Events": "Events",
    "dialogues": "Dialogues",
    "dialogue": "Dialogues",
    "Dialogues": "Dialogues",
}
CATEGORY_ORDER = ["Profile", "Social Relationship", "Events", "Dialogues"]
DEFAULT_EXPERIMENTS = ["full", "no_mental", "no_situation"]
_PRINT_LOCK = threading.Lock()


def log(msg: str = "") -> None:
    with _PRINT_LOCK:
        print(msg, flush=True)


def normalize_category(value: Any) -> str:
    if value is None:
        return "Unknown"
    text = str(value).strip()
    if not text:
        return "Unknown"
    if text in CATEGORY_NAME_MAP:
        return CATEGORY_NAME_MAP[text]
    low = text.lower().replace("_", " ").replace("-", " ")
    low = " ".join(low.split())
    return CATEGORY_NAME_MAP.get(low, text)


def normalize_category_filter(values: Optional[Any]) -> Optional[set[str]]:
    """Normalize CLI category names to canonical display names.

    Accepts both styles:
      --categories profile events
      --categories profile,events
      --categories "Profile, Events"
      --categories social_relationship "Social Relationship"

    This function is deliberately robust to values being either a list from
    argparse(nargs="*") or a single string. A previous version iterated over a
    string character-by-character when --categories was parsed as one string,
    causing filters like [',', '_', 'a', ...] and therefore zero matches.
    """
    if not values:
        return None
    if isinstance(values, str):
        raw_items = [values]
    else:
        raw_items = list(values)
    pieces: List[str] = []
    for raw in raw_items:
        if raw is None:
            continue
        for part in str(raw).split(","):
            part = part.strip()
            if part:
                pieces.append(part)
    normalized = {normalize_category(v) for v in pieces if str(v).strip()}
    return normalized or None


def item_category(item: Dict[str, Any]) -> str:
    return normalize_category(
        item.get("category_name") or item.get("category") or item.get("memory_type")
    )


def apply_eval_sampling(
    tasks: List[Tuple[str, Dict[str, Any]]],
    selected_categories: Optional[set[str]] = None,
    per_category_limit: int = 0,
    max_questions: int = 0,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Filter evaluation tasks before metric/judge computation.

    Sampling is applied per experiment, after category filtering, in the original
    JSON traversal order. This makes it useful for low-cost judge smoke tests:
      --categories Profile Events --per-category-limit 100
    evaluates at most 100 Profile + 100 Events examples per experiment.
    """
    filtered: List[Tuple[str, Dict[str, Any]]] = []
    per_cat_counts: Dict[str, int] = defaultdict(int)
    per_category_limit = max(0, int(per_category_limit or 0))
    max_questions = max(0, int(max_questions or 0))
    for character_id, item in tasks:
        category = item_category(item)
        if selected_categories and category not in selected_categories:
            continue
        if per_category_limit and per_cat_counts[category] >= per_category_limit:
            continue
        filtered.append((character_id, item))
        per_cat_counts[category] += 1
        if max_questions and len(filtered) >= max_questions:
            break
    return filtered


def count_tasks_by_category(tasks: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for _, item in tasks:
        counts[item_category(item)] += 1
    return dict(
        sorted(
            counts.items(),
            key=lambda kv: (CATEGORY_ORDER.index(kv[0]) if kv[0] in CATEGORY_ORDER else 999, kv[0]),
        )
    )


def get_ground_truth_answer(item: Dict[str, Any]) -> str:
    """Read GT answer from existing PerLTQA evaluator format and common variants."""
    for key in ["answer", "reference", "reference_answer", "gt_answer", "gold_answer"]:
        if key in item and item.get(key) is not None:
            return str(item.get(key, ""))
    return ""


def get_predicted_answer(item: Dict[str, Any]) -> str:
    """Read prediction from existing PerLTQA evaluator format and common variants."""
    for key in ["response", "prediction", "pred_answer", "model_answer", "generated_answer"]:
        if key in item and item.get(key) is not None:
            return str(item.get(key, "")).strip()
    return ""


def trimem_simple_tokenize(text: Any) -> List[str]:
    """TriMem test_locomo10.py tokenization for F1."""
    text = str(text)
    text = text.lower()
    text = re.sub("[\"\\'\\u201c\\u201d\\u2018\\u2019`]", "", text)
    text = re.sub("[.!?,;:()\\[\\]{}/\\\\]", " ", text)
    return text.split()


def calculate_trimem_rouge_scores(prediction: str, reference: str) -> Dict[str, float]:
    """TriMem ROUGE implementation: rouge_score.RougeScorer with stemming."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        "rouge1_f": scores["rouge1"].fmeasure,
        "rouge2_f": scores["rouge2"].fmeasure,
        "rougeL_f": scores["rougeL"].fmeasure,
    }


def _safe_word_tokenize(text: str) -> List[str]:
    text = str(text or "")
    try:
        return nltk.word_tokenize(text, preserve_line=True)
    except Exception:
        return re.findall("\\w+|[^\\w\\s]", text, flags=re.UNICODE)


def calculate_trimem_bleu_scores(prediction: str, reference: str) -> Dict[str, float]:
    """TriMem BLEU implementation: nltk sentence_bleu with method1 smoothing."""
    pred_tokens = _safe_word_tokenize(str(prediction).lower())
    ref_tokens = [_safe_word_tokenize(str(reference).lower())]
    weights_list = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
    smooth = SmoothingFunction().method1
    scores: Dict[str, float] = {}
    for n, weights in enumerate(weights_list, start=1):
        try:
            score = sentence_bleu(
                ref_tokens, pred_tokens, weights=weights, smoothing_function=smooth
            )
        except Exception:
            score = 0.0
        scores[f"bleu{n}"] = float(score)
    return scores


def calculate_trimem_metrics(prediction: str, reference: str) -> Dict[str, float]:
    """TriMem-style exact/F1/ROUGE/BLEU for PerLTQA records."""
    if not prediction or not reference:
        return {
            "exact_match": 0,
            "f1": 0.0,
            "rouge1_f": 0.0,
            "rouge2_f": 0.0,
            "rougeL_f": 0.0,
            "bleu1": 0.0,
            "bleu2": 0.0,
            "bleu3": 0.0,
            "bleu4": 0.0,
        }
    prediction = str(prediction).strip()
    reference = str(reference).strip()
    exact_match = int(prediction.lower() == reference.lower())
    pred_tokens = set(trimem_simple_tokenize(prediction))
    ref_tokens = set(trimem_simple_tokenize(reference))
    common_tokens = pred_tokens & ref_tokens
    if not pred_tokens or not ref_tokens:
        f1 = 0.0
    else:
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    rouge_scores = calculate_trimem_rouge_scores(prediction, reference)
    bleu_scores = calculate_trimem_bleu_scores(prediction, reference)
    return {"exact_match": exact_match, "f1": float(f1), **rouge_scores, **bleu_scores}


TRIMEM_JUDGE_SYSTEM_MESSAGE = "You are an expert evaluator. Always output valid JSON format."


def build_trimem_judge_prompt(question: str, reference: str, prediction: str) -> str:
    return f"""You are an expert Relevance & Accuracy Evaluator. Your task is to determine if the Predicted Answer successfully retrieves the necessary information to answer the Question, based on the Reference Answer.\n\nQuestion: {question}\nReference Answer: {reference}\nPredicted Answer: {prediction}\n\nEvaluation Principle:\nGive credit for semantic usefulness rather than exact wording. Prefer Score 1.0 when the predicted answer captures the main intent, any key factual signal, or a reasonable partial answer that would help the user answer the question.\n\n1. **Responsiveness to Query**: \n   The predicted answer must directly address the specific question asked. It must contain highly relevant information that is topically aligned with the user's intent.\n\n2. **Core Fact Preservation**: \n   The prediction must capture the "Key Signal" or "Core Entity" from the reference. The primary subject (Who), event (What), or outcome must be factually grounded in the reference text.\n\n3. **Informational Utility**: \n   The answer must provide actionable or meaningful value. Even if brief, it must convey the essential message required by the question context.\n\n4. **Acceptable Representational Variances (Robustness Protocol)**:\n   To ensure fair evaluation of semantic meaning over syntactic rigidity, you must accept the following variations as **Valid Matches**:\n   - **Temporal & Numerical Margins**: Accept timestamps within a reasonable proximity (e.g., +/- 1-2 days due to timezone/reporting differences) and rounded numerical approximations.\n   - **Granularity Independence**: Accept answers at different levels of abstraction (e.g., "Afternoon" vs. "14:05", "Late October" vs. "Oct 25th") provided they encompass the truth.\n   - **Information Subsetting**: A valid subset of the reference (e.g., mentioning 1 out of 3 reasons) is acceptable if it answers the core of the question.\n   - **Synonymy**: Recognize domain-specific synonyms and different formats as equivalent.\n\nFail Criteria:\nScore 0.0 only when one of the following is true:\n- The prediction is unrelated to the question.\n- The prediction identifies the wrong person, relationship, event, time, place, attribute, or outcome.\n- The prediction directly contradicts the reference.\n- The prediction says the answer is unknown/not mentioned when the reference contains a clear answer.\n- The prediction is so vague that it provides no usable information for answering the question.\n\nGrading Logic:\n- Score 1.0 (Pass): The prediction is fully correct, mostly correct, or partially correct but useful for the question.\n- Score 0.0 (Fail): The prediction is wrong, contradictory, unrelated, or non-informative.\n\nOutput your evaluation in JSON format:\n{{\n  "score": 1.0, \n  "reasoning": "Brief assessment focusing on information relevance and core match."\n}}\n\nReturn ONLY the JSON, no other text.\n"""


def extract_json_object(text: str) -> Dict[str, Any]:
    text = str(text or "").strip()
    if not text:
        raise ValueError("Empty judge response")
    if text.startswith("```"):
        text = re.sub("^```(?:json)?\\s*", "", text, flags=re.IGNORECASE)
        text = re.sub("\\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search("\\{.*\\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge response: {text[:200]}")
    return json.loads(match.group(0))


class ThreadLocalJudgeClient:
    """One OpenAI-compatible client per worker thread."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.3,
        timeout: float = 120.0,
        max_retries: int = 3,
        use_json_format: bool = False,
    ):
        if OpenAI is None:
            raise RuntimeError("openai package is required for --llm-judge")
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = float(temperature)
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.use_json_format = bool(use_json_format)
        self._local = threading.local()

    def _client(self):
        client = getattr(self._local, "client", None)
        if client is None:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
            self._local.client = client
        return client

    def judge(self, question: str, reference: str, prediction: str) -> Dict[str, Any]:
        if not prediction or not reference:
            return {
                "llm_judge_score": 0.0,
                "llm_reasoning": "Empty prediction or reference",
                "llm_judge_error": None,
            }
        messages = [
            {"role": "system", "content": TRIMEM_JUDGE_SYSTEM_MESSAGE},
            {"role": "user", "content": build_trimem_judge_prompt(question, reference, prediction)},
        ]
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.use_json_format:
            kwargs["response_format"] = {"type": "json_object"}
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client().chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
                parsed = extract_json_object(content)
                score = float(parsed.get("score", 0.0))
                score = 1.0 if score >= 0.5 else 0.0
                return {
                    "llm_judge_score": score,
                    "llm_reasoning": str(parsed.get("reasoning", "No reasoning provided")),
                    "llm_judge_error": None,
                }
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(min(2.0 * attempt, 8.0))
        return {
            "llm_judge_score": 0.0,
            "llm_reasoning": f"Evaluation failed: {last_error}",
            "llm_judge_error": str(last_error),
        }


def load_result_file(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "results" in data and isinstance(data["results"], list):
            grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for item in data["results"]:
                if not isinstance(item, dict):
                    continue
                character_id = str(
                    item.get("owner_id") or item.get("character") or item.get("owner") or "unknown"
                )
                grouped[character_id].append(item)
            return dict(grouped)
        return {str(k): list(v or []) for k, v in data.items() if isinstance(v, list)}
    if isinstance(data, list):
        grouped = defaultdict(list)
        for item in data:
            if not isinstance(item, dict):
                continue
            character_id = str(
                item.get("owner_id") or item.get("character") or item.get("owner") or "unknown"
            )
            grouped[character_id].append(item)
        return dict(grouped)
    raise ValueError(f"Unsupported result format in {path}")


def build_output_record(
    character_id: str, item: Dict[str, Any], judge_client: Optional[ThreadLocalJudgeClient] = None
) -> Dict[str, Any]:
    gt_answer = get_ground_truth_answer(item)
    pred_answer = get_predicted_answer(item)
    category = normalize_category(
        item.get("category_name") or item.get("category") or item.get("memory_type")
    )
    question = str(item.get("question", ""))
    metrics = calculate_trimem_metrics(pred_answer, gt_answer)
    judge_result: Dict[str, Any] = {}
    if judge_client is not None:
        judge_result = judge_client.judge(question, gt_answer, pred_answer)
        metrics["llm_judge_score"] = float(judge_result.get("llm_judge_score", 0.0))
    record = {
        "character": item.get("character"),
        "owner_id": item.get("owner_id", item.get("owner", character_id)),
        "qa_source_id": item.get("qa_source_id", item.get("memory_id", "")),
        "question": question,
        "answer": gt_answer,
        "response": pred_answer,
        "category": category,
        "category_name": category,
        "reference_memory": item.get("reference_memory", ""),
        "bleu_score": metrics.get("bleu1", 0.0),
        "f1_score": metrics.get("f1", 0.0),
        "rouge_l_score": metrics.get("rougeL_f", 0.0),
        "exact_match": metrics.get("exact_match", 0),
        "bleu1": metrics.get("bleu1", 0.0),
        "bleu2": metrics.get("bleu2", 0.0),
        "bleu3": metrics.get("bleu3", 0.0),
        "bleu4": metrics.get("bleu4", 0.0),
        "rouge1_f": metrics.get("rouge1_f", 0.0),
        "rouge2_f": metrics.get("rouge2_f", 0.0),
        "rougeL_f": metrics.get("rougeL_f", 0.0),
    }
    if judge_result:
        record["llm_judge_score"] = judge_result.get("llm_judge_score", 0.0)
        record["llm_reasoning"] = judge_result.get("llm_reasoning", "")
        if judge_result.get("llm_judge_error"):
            record["llm_judge_error"] = judge_result.get("llm_judge_error")
    return record


def evaluate_single_experiment(
    exp_name: str,
    input_path: Path,
    max_workers: int,
    judge_client: Optional[ThreadLocalJudgeClient] = None,
    selected_categories: Optional[set[str]] = None,
    per_category_limit: int = 0,
    max_questions: int = 0,
    list_categories_only: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    if not input_path.exists():
        log(f"[WARN] File not found: {input_path}")
        return {}
    data = load_result_file(input_path)
    all_tasks: List[Tuple[str, Dict[str, Any]]] = []
    for character_id, records in data.items():
        for item in records:
            if isinstance(item, dict):
                all_tasks.append((character_id, item))
    original_counts = count_tasks_by_category(all_tasks)
    log(f"[{exp_name}] Loaded {len(all_tasks)} QA records from {input_path}")
    log(
        f"[{exp_name}] Category counts before filtering: {json.dumps(original_counts, ensure_ascii=False)}"
    )
    tasks = apply_eval_sampling(
        all_tasks,
        selected_categories=selected_categories,
        per_category_limit=per_category_limit,
        max_questions=max_questions,
    )
    filtered_counts = count_tasks_by_category(tasks)
    log(f"[{exp_name}] Evaluating {len(tasks)} QA records after filtering/sampling.")
    log(
        f"[{exp_name}] Category counts after filtering:  {json.dumps(filtered_counts, ensure_ascii=False)}"
    )
    if list_categories_only:
        return {}
    results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    results_lock = threading.Lock()

    def worker(task: Tuple[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        character_id, item = task
        return (character_id, build_output_record(character_id, item, judge_client=judge_client))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        futures = [executor.submit(worker, task) for task in tasks]
        iterator: Iterable[Any]
        if tqdm is not None:
            iterator = tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"Evaluating {exp_name}",
            )
        else:
            iterator = concurrent.futures.as_completed(futures)
        for future in iterator:
            character_id, record = future.result()
            with results_lock:
                results[character_id].append(record)
    for character_id in results:
        results[character_id].sort(
            key=lambda r: str(r.get("qa_source_id", "")) + "\t" + str(r.get("question", ""))
        )
    return dict(results)


def flatten_experiment_data(exp_data: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    all_items: List[Dict[str, Any]] = []
    for character_id in exp_data:
        all_items.extend(exp_data[character_id])
    return all_items


def mean_or_zero(values: List[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def summarize_items(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_names = ["bleu_score", "f1_score", "rouge_l_score", "llm_judge_score"]
    summary: Dict[str, Any] = {"overall": {}, "by_category": {}, "num_questions": len(items)}
    for metric in metric_names:
        vals = [float(x[metric]) for x in items if isinstance(x.get(metric), (int, float))]
        if vals:
            summary["overall"][metric] = {
                "mean": mean_or_zero(vals),
                "median": float(statistics.median(vals)),
                "std": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
                "min": min(vals),
                "max": max(vals),
                "count": len(vals),
            }
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item.get("category_name", "Unknown"))].append(item)
    ordered_categories = [c for c in CATEGORY_ORDER if c in grouped]
    ordered_categories += [c for c in sorted(grouped) if c not in ordered_categories]
    for category in ordered_categories:
        cat_items = grouped[category]
        cat_summary = {"count": len(cat_items)}
        for metric in metric_names:
            vals = [float(x[metric]) for x in cat_items if isinstance(x.get(metric), (int, float))]
            if vals:
                cat_summary[metric] = {
                    "mean": mean_or_zero(vals),
                    "median": float(statistics.median(vals)),
                    "std": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
                    "min": min(vals),
                    "max": max(vals),
                    "count": len(vals),
                }
        summary["by_category"][category] = cat_summary
    return summary


def print_scores_for_experiment(exp_name: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    log(f"\n=== {exp_name} ===")
    if not items:
        log("No items found.")
        return {"overall": {}, "by_category": {}, "num_questions": 0}
    summary = summarize_items(items)
    metrics_to_show = ["bleu_score", "f1_score", "rouge_l_score"]
    if "llm_judge_score" in summary.get("overall", {}):
        metrics_to_show.append("llm_judge_score")
    log("Mean Scores Per Category:")
    header = "category".ljust(24) + "count".rjust(8)
    for metric in metrics_to_show:
        header += metric.rjust(18)
    log(header)
    for category, cat_summary in summary["by_category"].items():
        row = category.ljust(24) + str(cat_summary.get("count", 0)).rjust(8)
        for metric in metrics_to_show:
            mean_val = cat_summary.get(metric, {}).get("mean", None)
            row += (f"{mean_val:.4f}" if mean_val is not None else "-").rjust(18)
        log(row)
    log("\nOverall Mean Scores:")
    for metric in metrics_to_show:
        if metric in summary["overall"]:
            log(
                f"  {metric:18s}: {summary['overall'][metric]['mean']:.4f} (n={summary['overall'][metric]['count']})"
            )
    log(f"\nTotal evaluated questions: {len(items)}")
    return summary


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PerLTQA results with TriMem-style F1/BLEU/ROUGE-L and optional vLLM LLM-as-judge."
    )
    parser.add_argument("--results_dir", type=str, default=".")
    parser.add_argument(
        "--result-prefix",
        type=str,
        default="results_perltqa_world_memory",
        help="Prefix for <prefix>.full.json / no_mental / no_situation files.",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=DEFAULT_EXPERIMENTS,
        help="Experiment suffixes to evaluate, e.g. full no_mental no_situation.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="metrics_perltqa_world_memory_trimem.json",
        help="Detailed per-question metrics output JSON.",
    )
    parser.add_argument(
        "--summary_output_file",
        type=str,
        default="scores_perltqa_world_memory_trimem.json",
        help="Aggregate summary output JSON.",
    )
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Only evaluate selected categories. Accepted names include: profile, social_relationship, events, dialogues. Example: --categories profile events",
    )
    parser.add_argument(
        "--per-category-limit",
        type=int,
        default=0,
        help="Evaluate only the first K examples per selected category in each experiment. 0 means no per-category limit. Example: --categories profile events --per-category-limit 100",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=0,
        help="Evaluate only the first N examples per experiment after category filtering. 0 means no global limit.",
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="Only print available category counts after filtering setup; do not compute metrics or call judge.",
    )
    parser.add_argument(
        "--nltk-data-path",
        type=str,
        default=None,
        help="Optional NLTK data path, useful for remote servers without default punkt path.",
    )
    parser.add_argument("--llm-judge", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--judge-base-url",
        type=str,
        default=os.environ.get("JUDGE_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument(
        "--judge-api-key",
        type=str,
        default=os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY",
    )
    parser.add_argument("--judge-model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--judge-temperature", type=float, default=0.3)
    parser.add_argument("--judge-timeout", type=float, default=120.0)
    parser.add_argument("--judge-max-retries", type=int, default=3)
    parser.add_argument(
        "--judge-json-format",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass response_format={type: json_object}. Enable only if your vLLM model/server supports JSON mode well.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.nltk_data_path:
        nltk.data.path.append(args.nltk_data_path)
    results_dir = Path(args.results_dir)
    selected_categories = normalize_category_filter(args.categories)
    if selected_categories:
        log(f"Category filter enabled: {sorted(selected_categories)}")
    if args.per_category_limit:
        log(
            f"Per-category limit enabled: first {args.per_category_limit} examples/category/experiment"
        )
    if args.max_questions:
        log(
            f"Global per-experiment limit enabled: first {args.max_questions} examples after category filtering"
        )
    if args.list_categories:
        log("List-categories mode enabled: no metrics or judge calls will be made.")
    judge_client: Optional[ThreadLocalJudgeClient] = None
    if args.llm_judge and (not args.list_categories):
        judge_client = ThreadLocalJudgeClient(
            base_url=args.judge_base_url,
            api_key=args.judge_api_key,
            model=args.judge_model,
            temperature=args.judge_temperature,
            timeout=args.judge_timeout,
            max_retries=args.judge_max_retries,
            use_json_format=args.judge_json_format,
        )
        log(
            f"LLM-as-judge enabled: model={args.judge_model}, base_url={args.judge_base_url}, workers={args.max_workers}"
        )
    else:
        log("LLM-as-judge disabled. Use --llm-judge to enable it.")
    all_results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    all_summaries: Dict[str, Any] = {}
    for exp_name in args.experiments:
        input_path = results_dir / f"{args.result_prefix}.{exp_name}.json"
        exp_results = evaluate_single_experiment(
            exp_name=exp_name,
            input_path=input_path,
            max_workers=args.max_workers,
            judge_client=judge_client,
            selected_categories=selected_categories,
            per_category_limit=args.per_category_limit,
            max_questions=args.max_questions,
            list_categories_only=args.list_categories,
        )
        if args.list_categories:
            continue
        all_results[exp_name] = exp_results
        items = flatten_experiment_data(exp_results)
        all_summaries[exp_name] = print_scores_for_experiment(exp_name, items)
    if args.list_categories:
        log("\nCategory listing complete.")
        return
    detailed_path = results_dir / args.output_file
    summary_path = results_dir / args.summary_output_file
    write_json(detailed_path, all_results)
    write_json(summary_path, all_summaries)
    log(f"\nDetailed metrics saved to {detailed_path}")
    log(f"Aggregate summary saved to {summary_path}")


if __name__ == "__main__":
    main()
