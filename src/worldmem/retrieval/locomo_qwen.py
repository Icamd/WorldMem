from __future__ import annotations

import json
import logging
import re
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from worldmem.clients import EmbeddingClient, LLMClient
from worldmem.config import MemoryConfig
from worldmem.storage import read_jsonl as read_jsonl

logger = logging.getLogger(__name__)
from worldmem.prompts import locomo_qwen as prompt_profile


def _normalize_category_value(value: Any) -> str:
    """
    Normalize category values so that:
    1, \"1\", \"Category 1\", \"category_1\", \"cat-1\" -> \"1\"
    \"temporal\", \"Temporal\" -> \"temporal\"
    """
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub("^(category|cat)\\s+", "", text).strip()
    return text


def parse_category_filter(categories_arg: str) -> Optional[Set[str]]:
    """
    Parse --categories argument.

    Examples:
      --categories 1,3
      --categories \"category 1, category 3\"
      --categories temporal,inference
    """
    if not categories_arg:
        return None
    parts = [_normalize_category_value(x) for x in categories_arg.split(",") if x.strip()]
    cats = {x for x in parts if x}
    return cats or None


def qa_matches_category_filter(qa: Dict[str, Any], category_filter: Optional[Set[str]]) -> bool:
    if not category_filter:
        return True
    cat = qa.get("category")
    if isinstance(cat, list):
        normalized = {_normalize_category_value(x) for x in cat}
        return bool(normalized & category_filter)
    return _normalize_category_value(cat) in category_filter


def normalize_text(s: Any) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return " ".join(s.strip().split())


def dedup_preserve_order(items: List[Any]) -> List[Any]:
    seen = set()
    out = []
    for x in items:
        k = (
            json.dumps(x, sort_keys=True, ensure_ascii=False)
            if isinstance(x, (dict, list))
            else str(x)
        )
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def format_semantic_lines(memories: List[Dict[str, Any]], with_timestamp: bool = False) -> str:
    lines = []
    for idx, item in enumerate(memories):
        ts = item.get("timestamp") or item.get("created_at") or ""
        content = item.get("content", "")
        if with_timestamp and ts:
            lines.append(f"{idx}. [{ts}] {content}")
        else:
            lines.append(f"{idx}. {content}")
    return "\n".join(lines) if lines else "(none)"


def format_message_lines(messages: List[Dict[str, Any]], with_timestamp: bool = False) -> str:
    lines = []
    for idx, msg in enumerate(messages):
        ts = msg.get("timestamp") or ""
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if with_timestamp and ts:
            lines.append(f"{idx}. [{ts}] {role}: {content}")
        else:
            lines.append(f"{idx}. {role}: {content}")
    return "\n".join(lines) if lines else "(none)"


def format_semantic_message_pairs(pairs: List[Dict[str, Any]], with_timestamp: bool = False) -> str:
    lines = []
    for idx, pair in enumerate(pairs):
        sem = pair.get("semantic", {}) or {}
        msg = pair.get("message", {}) or {}
        sem_ts = sem.get("timestamp") or sem.get("created_at") or ""
        sem_content = sem.get("content", "")
        msg_ts = msg.get("timestamp") or ""
        msg_role = msg.get("role", "user")
        msg_content = msg.get("content", "")
        lines.append(f"Pair {idx}:")
        if with_timestamp and sem_ts:
            lines.append(f"  Semantic: [{sem_ts}] {sem_content}")
        else:
            lines.append(f"  Semantic: {sem_content}")
        if msg:
            if with_timestamp and msg_ts:
                lines.append(f"  Matched message: [{msg_ts}] {msg_role}: {msg_content}")
            else:
                lines.append(f"  Matched message: {msg_role}: {msg_content}")
        else:
            lines.append("  Matched message: (none)")
    return "\n".join(lines) if lines else "(none)"


class Searcher:
    prompts = prompt_profile

    def __init__(
        self,
        config: MemoryConfig,
        *,
        llm_client: LLMClient,
        embedding_client: EmbeddingClient,
        mental_top_k_labels: int = 30,
        mental_top_k_keywords: int = 10,
        mental_extra_label_top_k: int = 4,
        mental_extra_keywords_per_label: int = 5,
        situation_character_extra_top_k: int = 3,
        situation_model_top_k_per_bin: int = 10,
        situation_model_top_k_total: int = 40,
        situation_semantic_budget: int = 50,
        force_situation_semantic_budget: bool = True,
        always_supplement_global_semantics: bool = True,
        global_semantic_supplement_top_k: int = 60,
        min_situation_semantics_preserved: int = 20,
        direct_semantic_top_k: int = 3,
        guardrail_message_top_k: int = 3,
        include_original_messages_top_k: int = 10,
        stage_thinking: Optional[Dict[str, bool]] = None,
    ) -> None:
        self.config = config
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.mental_top_k_labels = mental_top_k_labels
        self.mental_top_k_keywords = mental_top_k_keywords
        self.mental_extra_label_top_k = mental_extra_label_top_k
        self.mental_extra_keywords_per_label = mental_extra_keywords_per_label
        self.situation_character_extra_top_k = situation_character_extra_top_k
        self.situation_model_top_k_per_bin = situation_model_top_k_per_bin
        self.situation_model_top_k_total = situation_model_top_k_total
        self.situation_semantic_budget = situation_semantic_budget
        self.force_situation_semantic_budget = bool(force_situation_semantic_budget)
        self.always_supplement_global_semantics = bool(always_supplement_global_semantics)
        self.global_semantic_supplement_top_k = max(0, int(global_semantic_supplement_top_k or 0))
        self.min_situation_semantics_preserved = max(0, int(min_situation_semantics_preserved or 0))
        self.direct_semantic_top_k = direct_semantic_top_k
        self.guardrail_message_top_k = guardrail_message_top_k
        self.include_original_messages_top_k = include_original_messages_top_k
        self.stage_thinking = stage_thinking or {}
        self.thinking_generation_config = {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
        }
        self.non_thinking_generation_config = {
            "temperature": 0.0,
            "top_p": None,
            "top_k": None,
            "min_p": None,
        }
        self._user_cache: Dict[str, Dict[str, Any]] = {}
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._token_lock = threading.Lock()
        self._token_stats = {
            "full": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "num_queries": 0},
            "no_mental": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "num_queries": 0,
            },
            "no_situation": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "num_queries": 0,
            },
        }
        self._inference_re = re.compile("|".join(self.prompts.INFERENCE_PATTERNS), re.IGNORECASE)

    def _count_tokens(self, text: str) -> int:
        if not self.llm_client:
            return 0
        tokenizer = getattr(self.llm_client, "tokenizer", None)
        if tokenizer is None or not text:
            return 0
        try:
            enc = tokenizer(
                text, add_special_tokens=True, return_attention_mask=False, return_tensors=None
            )
            input_ids = enc.get("input_ids")
            if isinstance(input_ids, list):
                if len(input_ids) == 0:
                    return 0
                if isinstance(input_ids[0], list):
                    return len(input_ids[0])
                return len(input_ids)
            return 0
        except Exception:
            try:
                return self.llm_client.count_tokens(text)
            except Exception:
                return 0

    def _record_tokens(
        self,
        exp_name: str,
        input_text: str,
        output_text: str,
        llm_usage: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        if isinstance(llm_usage, dict):
            input_tokens = int(
                llm_usage.get("prompt_tokens_est", self._count_tokens(input_text)) or 0
            )
            output_tokens = int(
                llm_usage.get("completion_tokens_est", self._count_tokens(output_text)) or 0
            )
            total = int(llm_usage.get("total_tokens_est", input_tokens + output_tokens) or 0)
        else:
            input_tokens = self._count_tokens(input_text)
            output_tokens = self._count_tokens(output_text)
            total = input_tokens + output_tokens
        with self._token_lock:
            self._token_stats[exp_name]["input_tokens"] += input_tokens
            self._token_stats[exp_name]["output_tokens"] += output_tokens
            self._token_stats[exp_name]["total_tokens"] += total
        result = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total,
        }
        if isinstance(llm_usage, dict):
            result["thinking_tokens"] = int(llm_usage.get("thinking_tokens_est", 0) or 0)
            result["visible_output_tokens"] = int(
                llm_usage.get("visible_completion_tokens_est", output_tokens) or 0
            )
        return result

    def _mark_query(self, exp_name: str) -> None:
        with self._token_lock:
            self._token_stats[exp_name]["num_queries"] += 1

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return 0.0
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _embed_text(self, text: str) -> np.ndarray:
        key = normalize_text(text)
        if key in self._embedding_cache:
            return self._embedding_cache[key]
        emb = np.asarray(self.embedding_client.embed_text(key), dtype=np.float32)
        self._embedding_cache[key] = emb
        return emb

    def _topk_by_similarity(
        self, query_text: str, candidates: List[str], top_k: int
    ) -> List[Tuple[str, float]]:
        if not candidates:
            return []
        q_emb = self._embed_text(query_text)
        scored = []
        for c in candidates:
            try:
                sim = self._cosine_sim(q_emb, self._embed_text(c))
            except Exception:
                sim = 0.0
            scored.append((c, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _is_inference_question(self, query: str) -> bool:
        return bool(self._inference_re.search(query))

    def _load_user_memories(self, user_id: str) -> Dict[str, Any]:
        if user_id in self._user_cache:
            return self._user_cache[user_id]
        storage_root = Path(self.config.storage_path)
        mental_path = storage_root / "mental" / f"{user_id}_mental.jsonl"
        situation_path = storage_root / "situation" / f"{user_id}_situation.jsonl"
        semantic_path = storage_root / "semantic" / f"{user_id}_semantic.jsonl"
        episode_path = storage_root / "episodes" / f"{user_id}_episodes.jsonl"
        message_block_path = storage_root / "message_blocks" / f"{user_id}_message_blocks.jsonl"
        mental_items = read_jsonl(mental_path)
        situation_items = read_jsonl(situation_path)
        semantic_items = read_jsonl(semantic_path)
        episode_items = read_jsonl(episode_path)
        message_blocks = read_jsonl(message_block_path)
        semantic_map = {}
        for s in semantic_items:
            sid = s.get("memory_id")
            if sid:
                semantic_map[sid] = s
        episode_map = {}
        for e in episode_items:
            eid = e.get("episode_id")
            if eid:
                episode_map[eid] = e
        message_map = {}
        for ep in episode_items:
            for msg in ep.get("original_messages", []) or []:
                mid = msg.get("message_id")
                if mid:
                    message_map[mid] = msg
        message_block_map = {}
        for block in message_blocks:
            bid = block.get("block_id")
            if bid:
                message_block_map[bid] = block
            for msg in block.get("messages", []) or []:
                mid = msg.get("message_id")
                if mid and mid not in message_map:
                    message_map[mid] = msg
        label_to_keywords: Dict[str, Set[str]] = defaultdict(set)
        all_labels: Set[str] = set()
        all_keywords: Set[str] = set()
        for m in mental_items:
            clusters = m.get("clusters", []) or []
            for cluster in clusters:
                label = normalize_text(
                    cluster.get("label") or cluster.get("cluster_summary") or m.get("content", "")
                )
                if not label:
                    continue
                all_labels.add(label)
                for link in cluster.get("semantic_links", []) or []:
                    for kw in link.get("keywords", []) or []:
                        kw = normalize_text(kw)
                        if not kw:
                            continue
                        all_keywords.add(kw)
                        label_to_keywords[label].add(kw)
        semantic_label_map: Dict[str, Set[str]] = defaultdict(set)
        for s in semantic_items:
            sid = s.get("memory_id")
            if not sid:
                continue
            for ann in s.get("label_annotations", []) or []:
                label = normalize_text(ann.get("label", ""))
                if label:
                    semantic_label_map[sid].add(label)
        situation_keyword_map: Dict[str, Dict[str, Any]] = {}
        situation_to_semantic_ids: Dict[str, List[str]] = {}
        for sit in situation_items:
            sit_id = sit.get("situation_id")
            if not sit_id:
                continue
            keyword = normalize_text(
                sit.get("summary")
                or sit.get("routing_summary")
                or sit.get("final_summary")
                or (sit.get("metadata") or {}).get("keyword")
                or ""
            )
            if keyword:
                situation_keyword_map[keyword] = sit
            seen_sem = []
            seen_ids = set()
            for agg in sit.get("aggregates", []) or []:
                for sid in agg.get("semantic_ids", []) or []:
                    if sid not in seen_ids:
                        seen_ids.add(sid)
                        seen_sem.append(sid)
            situation_to_semantic_ids[sit_id] = seen_sem
        data = {
            "mental": mental_items,
            "situations": situation_items,
            "semantics": semantic_items,
            "episodes": episode_items,
            "message_blocks": message_blocks,
            "semantic_map": semantic_map,
            "episode_map": episode_map,
            "message_map": message_map,
            "message_block_map": message_block_map,
            "all_labels": sorted(list(all_labels)),
            "all_keywords": sorted(list(all_keywords)),
            "label_to_keywords": {k: sorted(list(v)) for k, v in label_to_keywords.items()},
            "semantic_label_map": semantic_label_map,
            "situation_keyword_map": situation_keyword_map,
            "situation_to_semantic_ids": situation_to_semantic_ids,
        }
        self._user_cache[user_id] = data
        return data

    def _llm_call_config_for_stage(self, stage_name: str) -> Dict[str, Any]:
        enable_thinking = bool(self.stage_thinking.get(stage_name, False))
        if enable_thinking:
            cfg = dict(self.thinking_generation_config)
        else:
            cfg = dict(self.non_thinking_generation_config)
        cfg["enable_thinking"] = enable_thinking
        return cfg

    def _json_stage_max_tokens(self, stage_name: str) -> int:
        """
        JSON-control stages should stay short. This is especially important for
        Llama-3.1-8B-Instruct, which can drift into code-generation when allowed
        a large completion budget.
        """
        stage_limits = {
            "mental_query_analysis": 4096,
            "mental_keyword_selection": 4096,
            "no_mental_selection": 4096,
            "situation_decision": 4096,
        }
        return int(stage_limits.get(stage_name, 4096))

    def _strict_json_system_prompt(self, stage_name: str) -> str:
        return """You are a strict JSON API used inside an automated memory-retrieval pipeline.
Return exactly one valid JSON object and nothing else.
Do not write Python code, pseudocode, markdown fences, examples, comments, explanations, or an Output section.
Your response must start with '{' and end with '}'.
Use double quotes for all JSON strings. Use null, true, and false where appropriate."""

    def _json_retry_prompt(self, prompt_text: str, default_response: Dict[str, Any]) -> str:
        expected_keys = list(default_response.keys())
        return """{0}

The previous response was invalid for this automated parser.
Retry now. Return ONLY the final JSON object.
Do not include Python code, markdown fences, natural-language explanation, examples, or labels like Output:.
The JSON object must contain these top-level keys: {1}.""".format(
            prompt_text, json.dumps(expected_keys, ensure_ascii=False)
        )

    def _extract_fenced_json_blocks(self, text: str) -> List[str]:
        blocks: List[str] = []
        for match in re.finditer("```([^\\n`]*)\\n(.*?)```", text or "", flags=re.DOTALL):
            lang = (match.group(1) or "").strip().lower()
            body = (match.group(2) or "").strip()
            if lang in {"json", "jsonc", ""} and body:
                blocks.append(body)
        return blocks

    def _iter_balanced_json_object_candidates(self, text: str) -> List[str]:
        """
        Return balanced {...} substrings while respecting quoted strings.
        We later validate candidates against expected top-level keys, so this can
        safely scan outputs that contain code before/after the actual JSON.
        """
        candidates: List[str] = []
        if not text:
            return candidates
        start = None
        depth = 0
        in_str = False
        escape = False
        for i, ch in enumerate(text):
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        candidates.append(text[start : i + 1])
                        start = None
        return candidates

    def _coerce_json_response_dict(
        self, obj: Any, default_response: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if not isinstance(obj, dict):
            return (None, f"expected dict, got {type(obj).__name__}")
        expected_keys = set(default_response.keys())
        missing = [k for k in expected_keys if k not in obj]
        if missing:
            return (None, f"missing expected keys: {missing}")
        out = dict(obj)
        for k, v in default_response.items():
            if k not in out:
                out[k] = v
        return (out, None)

    def _parse_json_object_from_raw_text(
        self, raw_text: str, default_response: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Robust parser for Llama-style outputs that may contain code plus a final
        JSON block. It prefers fenced JSON blocks, then whole-text JSON, then
        balanced {...} candidates from the end of the output.
        """
        errors: List[Dict[str, Any]] = []
        raw_text = raw_text or ""

        def try_candidate(
            candidate: str, method: str
        ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
            try:
                parsed = json.loads(candidate.strip())
            except Exception as e:
                errors.append({"method": method, "error": repr(e), "preview": candidate[:500]})
                return None
            coerced, err = self._coerce_json_response_dict(parsed, default_response)
            if coerced is None:
                errors.append({"method": method, "error": err, "preview": candidate[:500]})
                return None
            return (
                coerced,
                {"success": True, "method": method, "expected_keys": list(default_response.keys())},
            )

        for block in reversed(self._extract_fenced_json_blocks(raw_text)):
            result = try_candidate(block, "fenced_json_block")
            if result is not None:
                return result
        result = try_candidate(raw_text, "full_text")
        if result is not None:
            return result
        for candidate in reversed(self._iter_balanced_json_object_candidates(raw_text)):
            result = try_candidate(candidate, "balanced_object_scan")
            if result is not None:
                return result
        return (
            dict(default_response),
            {
                "success": False,
                "method": "parse_failed_default_response",
                "expected_keys": list(default_response.keys()),
                "candidate_errors": errors[-10:],
            },
        )

    def _run_json_prompt(
        self, exp_name: str, prompt_text: str, default_response: Dict[str, Any], stage_name: str
    ) -> Tuple[Dict[str, Any], Dict[str, int], Dict[str, Any]]:
        """
        Run JSON-control stages through the text-generation path, then parse here.

        This avoids the old client-side JSON wrapper swallowing invalid Llama
        outputs and returning only default_response. The trace now records the
        true raw model text, including code-like failure outputs, so save-full-
        results can diagnose the issue.
        """
        call_cfg = self._llm_call_config_for_stage(stage_name)
        max_tokens = self._json_stage_max_tokens(stage_name)
        system_prompt = self._strict_json_system_prompt(stage_name)
        attempts: List[Dict[str, Any]] = []
        aggregate_token_usage: Dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "thinking_tokens": 0,
            "visible_output_tokens": 0,
        }
        out: Dict[str, Any] = dict(default_response)
        raw_text = ""
        parse_info: Dict[str, Any] = {
            "success": False,
            "method": "not_run",
            "expected_keys": list(default_response.keys()),
        }
        llm_usage: Dict[str, Any] = {}
        max_attempts = 2
        for attempt_idx in range(max_attempts):
            effective_prompt = (
                prompt_text
                if attempt_idx == 0
                else self._json_retry_prompt(prompt_text, default_response)
            )
            if hasattr(self.llm_client, "generate_text_response_with_usage"):
                llm_resp = self.llm_client.generate_text_response_with_usage(
                    prompt=effective_prompt,
                    system_prompt=system_prompt,
                    temperature=call_cfg["temperature"],
                    max_tokens=max_tokens,
                    enable_thinking=call_cfg["enable_thinking"],
                    top_p=call_cfg["top_p"],
                    top_k=call_cfg["top_k"],
                    min_p=call_cfg["min_p"],
                )
                raw_text = (getattr(llm_resp, "content", None) or "").strip()
                llm_usage = getattr(llm_resp, "usage", {}) or {}
            else:
                raw_text = self.llm_client.generate_text_response(
                    prompt=effective_prompt,
                    system_prompt=system_prompt,
                    temperature=call_cfg["temperature"],
                    max_tokens=max_tokens,
                    enable_thinking=call_cfg["enable_thinking"],
                )
                raw_text = (raw_text or "").strip()
                llm_usage = {}
            attempt_token_usage = self._record_tokens(
                exp_name, effective_prompt, raw_text, llm_usage=llm_usage
            )
            for k, v in attempt_token_usage.items():
                aggregate_token_usage[k] = int(aggregate_token_usage.get(k, 0) or 0) + int(v or 0)
            parsed_out, parse_info = self._parse_json_object_from_raw_text(
                raw_text, default_response
            )
            attempts.append(
                {
                    "attempt": attempt_idx + 1,
                    "raw_response": raw_text,
                    "response": parsed_out,
                    "parse_info": parse_info,
                    "token_usage": attempt_token_usage,
                    "llm_usage": llm_usage,
                }
            )
            out = parsed_out
            if parse_info.get("success"):
                break
            logger.warning(
                "[%s] JSON parsing failed on attempt %s/%s. Raw output preview: %s",
                stage_name,
                attempt_idx + 1,
                max_attempts,
                raw_text[:1000],
            )
        if not isinstance(out, dict):
            out = dict(default_response)
        for k, v in default_response.items():
            if k not in out:
                logger.warning(
                    "[%s] JSON response missing key %r. Fill with default value.", stage_name, k
                )
                out[k] = v
        trace = {
            "stage": stage_name,
            "prompt": prompt_text,
            "response": out,
            "raw_response": raw_text,
            "parse_success": bool(parse_info.get("success")),
            "parse_info": parse_info,
            "attempts": attempts,
            "token_usage": aggregate_token_usage,
            "llm_usage": llm_usage,
            "enable_thinking": call_cfg["enable_thinking"],
            "generation_config": {
                "temperature": call_cfg["temperature"],
                "top_p": call_cfg["top_p"],
                "top_k": call_cfg["top_k"],
                "min_p": call_cfg["min_p"],
                "max_tokens": max_tokens,
                "json_system_prompt": system_prompt,
            },
        }
        return (out, aggregate_token_usage, trace)

    def _run_text_prompt(
        self, exp_name: str, prompt_text: str, max_tokens: int, stage_name: str
    ) -> Tuple[str, Dict[str, int], Dict[str, Any]]:
        call_cfg = self._llm_call_config_for_stage(stage_name)
        if hasattr(self.llm_client, "generate_text_response_with_usage"):
            llm_resp = self.llm_client.generate_text_response_with_usage(
                prompt=prompt_text,
                system_prompt=None,
                temperature=call_cfg["temperature"],
                max_tokens=max_tokens,
                enable_thinking=call_cfg["enable_thinking"],
                top_p=call_cfg["top_p"],
                top_k=call_cfg["top_k"],
                min_p=call_cfg["min_p"],
            )
            out = (llm_resp.content or "").strip()
            llm_usage = getattr(llm_resp, "usage", {}) or {}
        else:
            out = self.llm_client.generate_text_response(
                prompt=prompt_text,
                system_prompt=None,
                temperature=call_cfg["temperature"],
                max_tokens=max_tokens,
                enable_thinking=call_cfg["enable_thinking"],
            )
            out = (out or "").strip()
            llm_usage = {}
        token_usage = self._record_tokens(exp_name, prompt_text, out, llm_usage=llm_usage)
        trace = {
            "stage": stage_name,
            "prompt": prompt_text,
            "response": out,
            "token_usage": token_usage,
            "llm_usage": llm_usage,
            "enable_thinking": call_cfg["enable_thinking"],
            "generation_config": {
                "temperature": call_cfg["temperature"],
                "top_p": call_cfg["top_p"],
                "top_k": call_cfg["top_k"],
                "min_p": call_cfg["min_p"],
            },
        }
        return (out, token_usage, trace)

    def _select_topk_labels_for_query(self, query: str, labels: List[str], top_k: int) -> List[str]:
        return [x for x, _ in self._topk_by_similarity(query, labels, top_k)]

    def _analyze_query_with_mental(
        self, exp_name: str, query: str, candidate_labels: List[str]
    ) -> Tuple[Dict[str, Any], Dict[str, int], Dict[str, Any]]:
        prompt = self.prompts.MENTAL_QUERY_ANALYSIS_PROMPT.render(
            question=query, labels=json.dumps(candidate_labels, ensure_ascii=False)
        )
        return self._run_json_prompt(
            exp_name,
            prompt,
            default_response={"query_label_keywords": {}, "expected_answer_labels": []},
            stage_name="mental_query_analysis",
        )

    def _build_candidate_keywords(
        self,
        exp_name: str,
        user_data: Dict[str, Any],
        query: str,
        query_label_keywords: Dict[str, List[Any]],
        expected_answer_labels: List[str],
    ) -> Tuple[Dict[str, List[str]], List[str], Dict[str, Any]]:
        label_to_keywords = user_data["label_to_keywords"]
        all_labels = list(label_to_keywords.keys())
        candidate_keywords: Dict[str, List[str]] = defaultdict(list)
        debug_info: Dict[str, Any] = {"answer_label_ranking": {}, "extra_label_candidates": {}}
        extracted_query_keywords_flat = []
        for label, kws in (query_label_keywords or {}).items():
            label = normalize_text(label)
            all_label_keywords = dedup_preserve_order(label_to_keywords.get(label, []))
            if not all_label_keywords:
                continue
            if kws is None:
                kws = []
            if not isinstance(kws, list):
                kws = [kws]
            normalized_kws = []
            for kw in kws:
                if kw is None:
                    normalized_kws.append(None)
                else:
                    kw = normalize_text(kw)
                    if kw:
                        normalized_kws.append(kw)
            valid_kws = [kw for kw in normalized_kws if isinstance(kw, str)]
            for kw in valid_kws:
                extracted_query_keywords_flat.append(kw)
            if label == "Character":
                k_each = max(1, len(valid_kws) * 3)
            else:
                k_each = self.mental_top_k_keywords
            if label == "Time" and len(valid_kws) == 0:
                continue
            selected = []
            for kw in valid_kws:
                top = self._topk_by_similarity(kw, all_label_keywords, k_each)
                selected.extend([x for x, _ in top])
            candidate_keywords[label].extend(selected)
            candidate_keywords[label] = dedup_preserve_order(candidate_keywords[label])
        for label in expected_answer_labels or []:
            label = normalize_text(label)
            all_label_keywords = dedup_preserve_order(label_to_keywords.get(label, []))
            if not all_label_keywords:
                continue
            ranked = self._topk_by_similarity(query, all_label_keywords, self.mental_top_k_keywords)
            selected = [x for x, _ in ranked]
            candidate_keywords[label].extend(selected)
            candidate_keywords[label] = dedup_preserve_order(candidate_keywords[label])
            debug_info["answer_label_ranking"][label] = {
                "method": "embedding_similarity",
                "details": [{"keyword": x, "score": float(s)} for x, s in ranked],
            }
        main_labels = set(candidate_keywords.keys())
        excluded_labels = {"Character", "Time"} | main_labels
        remaining_labels = [lbl for lbl in all_labels if lbl not in excluded_labels]
        ranked_labels = self._topk_by_similarity(
            query, remaining_labels, self.mental_extra_label_top_k
        )
        for lbl, lbl_score in ranked_labels:
            kws = dedup_preserve_order(label_to_keywords.get(lbl, []))
            if not kws:
                continue
            top_kw = self._topk_by_similarity(query, kws, self.mental_extra_keywords_per_label)
            selected_kw = [x for x, _ in top_kw]
            candidate_keywords[lbl].extend(selected_kw)
            candidate_keywords[lbl] = dedup_preserve_order(candidate_keywords[lbl])
            debug_info["extra_label_candidates"][lbl] = {
                "label_score": float(lbl_score),
                "keywords": [{"keyword": x, "score": float(s)} for x, s in top_kw],
            }
        return (
            dict(candidate_keywords),
            dedup_preserve_order(extracted_query_keywords_flat),
            debug_info,
        )

    def _select_answer_keywords(
        self, exp_name: str, query: str, candidate_keywords: Dict[str, List[str]]
    ) -> Tuple[Dict[str, List[str]], Dict[str, int], Dict[str, Any]]:
        prompt = self.prompts.MENTAL_KEYWORD_SELECTION_PROMPT.render(
            question=query,
            candidate_keywords=json.dumps(candidate_keywords, ensure_ascii=False, indent=2),
        )
        return self._run_json_prompt(
            exp_name,
            prompt,
            default_response={"selected_keywords": {}},
            stage_name="mental_keyword_selection",
        )

    def _resolve_keywords_to_semantics_by_content(
        self, selected_keywords: Dict[str, List[str]], user_data: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        semantic_items = user_data["semantics"]
        resolved: Dict[str, List[Dict[str, Any]]] = {}
        for label, kws in (selected_keywords or {}).items():
            resolved[label] = []
            seen_sem = set()
            for kw in kws or []:
                kw_norm = normalize_text(kw).lower()
                if not kw_norm:
                    continue
                for sem in semantic_items:
                    sid = sem.get("memory_id")
                    if not sid:
                        continue
                    matched = False
                    content = normalize_text(sem.get("content", "")).lower()
                    if kw_norm in content:
                        matched = True
                    if not matched:
                        for ann in sem.get("label_annotations", []) or []:
                            ann_kws = [
                                normalize_text(x).lower() for x in ann.get("keywords", []) or []
                            ]
                            if kw_norm in ann_kws:
                                matched = True
                                break
                    if matched and sid not in seen_sem:
                        seen_sem.add(sid)
                        resolved[label].append(sem)
        return resolved

    def _run_mental_full(
        self, query: str, user_data: Dict[str, Any], exp_name: str = "full"
    ) -> Dict[str, Any]:
        labels = user_data["all_labels"] or list(self.prompts.DEFAULT_SEMANTIC_LABELS)
        candidate_labels = self._select_topk_labels_for_query(
            query, labels, self.mental_top_k_labels
        )
        analysis_resp, tok1, trace1 = self._analyze_query_with_mental(
            exp_name, query, candidate_labels
        )
        query_label_keywords = analysis_resp.get("query_label_keywords", {}) or {}
        expected_answer_labels = analysis_resp.get("expected_answer_labels", []) or []
        if "when" in query.lower():
            if "Time" not in query_label_keywords:
                query_label_keywords["Time"] = [None]
            if "Time" not in expected_answer_labels:
                expected_answer_labels.append("Time")
        candidate_keywords, extracted_query_keywords, build_debug = self._build_candidate_keywords(
            exp_name=exp_name,
            user_data=user_data,
            query=query,
            query_label_keywords=query_label_keywords,
            expected_answer_labels=expected_answer_labels,
        )
        selected_keywords_resp, tok2, trace2 = self._select_answer_keywords(
            exp_name, query, candidate_keywords
        )
        selected_keywords = selected_keywords_resp.get("selected_keywords", {}) or {}
        resolved_keyword_semantics = self._resolve_keywords_to_semantics_by_content(
            selected_keywords=selected_keywords, user_data=user_data
        )
        involves_time = False
        if "Time" in query_label_keywords or "Time" in expected_answer_labels:
            involves_time = True
        return {
            "candidate_labels": candidate_labels,
            "query_label_keywords": query_label_keywords,
            "expected_answer_labels": expected_answer_labels,
            "candidate_keywords": candidate_keywords,
            "selected_keywords": selected_keywords,
            "resolved_keyword_semantics": resolved_keyword_semantics,
            "extracted_query_keywords": extracted_query_keywords,
            "involves_time": involves_time,
            "debug": build_debug,
            "trace": {"mental_stage_1": trace1, "mental_stage_2": trace2},
            "token_usage": {
                "stage1": tok1,
                "stage2": tok2,
                "total_tokens": tok1["total_tokens"] + tok2["total_tokens"],
            },
        }

    def _run_mental_no_mental(
        self, query: str, user_data: Dict[str, Any], exp_name: str = "no_mental"
    ) -> Dict[str, Any]:
        semantics = user_data["semantics"]
        scored = []
        for idx, s in enumerate(semantics):
            content = s.get("content", "")
            sim = self._topk_by_similarity(query, [content], 1)[0][1] if content else 0.0
            scored.append((idx, s, sim))
        scored.sort(key=lambda x: x[2], reverse=True)
        top_semantics = [x[1] for x in scored[: self.mental_top_k_labels]]
        sem_lines = [f"{idx}. {s.get('content', '')}" for idx, s in enumerate(top_semantics)]
        prompt = self.prompts.NO_MENTAL_SELECTION_PROMPT.render(
            question=query,
            semantics="\n".join(sem_lines),
        )
        resp, tok, trace = self._run_json_prompt(
            exp_name,
            prompt,
            default_response={"selected_indices": []},
            stage_name="no_mental_selection",
        )
        selected_indices = resp.get("selected_indices", []) or []
        selected_semantics = []
        for i in selected_indices:
            try:
                i = int(i)
                if 0 <= i < len(top_semantics):
                    selected_semantics.append(top_semantics[i])
            except Exception:
                continue
        if not selected_semantics and top_semantics:
            selected_semantics = top_semantics[: min(3, len(top_semantics))]
        return {
            "candidate_semantics": top_semantics,
            "selected_semantics": selected_semantics,
            "trace": {"no_mental_stage_1": trace},
            "token_usage": tok,
        }

    def _keyword_as_phrase_in_text(self, keyword: str, text: str) -> bool:
        kw = normalize_text(keyword).lower()
        tx = normalize_text(text).lower()
        if not kw or not tx:
            return False
        pattern = "(?<!\\w)" + re.escape(kw) + "(?!\\w)"
        return re.search(pattern, tx) is not None

    def _supplement_semantics_from_query_keywords(
        self,
        query: str,
        query_label_keywords: Dict[str, List[Any]],
        existing_semantic_ids: Set[str],
        user_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        semantic_items = user_data["semantics"]
        extra = []
        raw_keywords = []
        for label, kws in (query_label_keywords or {}).items():
            if label in {"Character", "Time"}:
                continue
            for kw in kws or []:
                kw = normalize_text(kw)
                if kw:
                    raw_keywords.append(kw)
        for kw in dedup_preserve_order(raw_keywords):
            matches = []
            for sem in semantic_items:
                sid = sem.get("memory_id")
                if not sid or sid in existing_semantic_ids:
                    continue
                content = sem.get("content", "")
                if self._keyword_as_phrase_in_text(kw, content):
                    sim = self._topk_by_similarity(query, [content], 1)[0][1] if content else 0.0
                    matches.append((sem, sim))
            matches.sort(key=lambda x: x[1], reverse=True)
            if matches:
                chosen = matches[0][0]
                sid = chosen.get("memory_id")
                if sid and sid not in existing_semantic_ids:
                    existing_semantic_ids.add(sid)
                    extra.append(chosen)
        return extra

    def _supplement_messages_from_query_keywords(
        self,
        query: str,
        query_label_keywords: Dict[str, List[Any]],
        existing_message_ids: Set[str],
        user_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        message_map = user_data["message_map"]
        extra = []
        raw_keywords = []
        for label, kws in (query_label_keywords or {}).items():
            if label in {"Character", "Time"}:
                continue
            for kw in kws or []:
                kw = normalize_text(kw)
                if kw:
                    raw_keywords.append(kw)
        all_messages = list(message_map.values())
        for kw in dedup_preserve_order(raw_keywords):
            matches = []
            for msg in all_messages:
                mid = msg.get("message_id")
                if not mid or mid in existing_message_ids:
                    continue
                content = msg.get("content", "")
                if self._keyword_as_phrase_in_text(kw, content):
                    sim = self._topk_by_similarity(query, [content], 1)[0][1] if content else 0.0
                    matches.append((msg, sim))
            matches.sort(key=lambda x: x[1], reverse=True)
            if matches:
                chosen = matches[0][0]
                mid = chosen.get("message_id")
                if mid and mid not in existing_message_ids:
                    existing_message_ids.add(mid)
                    extra.append(chosen)
        return extra

    def _find_situations_for_keywords(
        self, keywords: List[str], user_data: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        situation_keyword_map = user_data["situation_keyword_map"]
        all_sit_keywords = list(situation_keyword_map.keys())
        selected = []
        seen_sit = set()
        replacement = {}
        for kw in dedup_preserve_order([normalize_text(x) for x in keywords if normalize_text(x)]):
            used_kw = kw
            if kw not in situation_keyword_map and all_sit_keywords:
                ranked = self._topk_by_similarity(kw, all_sit_keywords, 1)
                if ranked:
                    used_kw = ranked[0][0]
            replacement[kw] = used_kw
            sit = situation_keyword_map.get(used_kw)
            if sit is None:
                continue
            sid = sit.get("situation_id")
            if sid and sid not in seen_sit:
                seen_sit.add(sid)
                selected.append(sit)
        return (selected, replacement)

    def _collect_unique_semantics_from_situations(
        self, situations: List[Dict[str, Any]], user_data: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        semantic_map = user_data["semantic_map"]
        kept_situations = []
        collected_semantics = []
        seen_sem = set()
        for sit in situations:
            new_sem_for_sit = []
            for agg in sit.get("aggregates", []) or []:
                for sid in agg.get("semantic_ids", []) or []:
                    if sid in seen_sem:
                        continue
                    sem = semantic_map.get(sid)
                    if sem is None:
                        continue
                    seen_sem.add(sid)
                    new_sem_for_sit.append(sem)
            if new_sem_for_sit:
                kept_situations.append(sit)
                collected_semantics.extend(new_sem_for_sit)
        return (kept_situations, collected_semantics)

    def _semantic_dedup_key(self, sem: Dict[str, Any]) -> str:
        """Stable key for deduplicating semantic memories."""
        sid = sem.get("memory_id")
        if sid:
            return str(sid)
        try:
            return json.dumps(sem, sort_keys=True, ensure_ascii=False)
        except Exception:
            return repr(sem)

    def _semantic_debug_source(self, sem: Dict[str, Any]) -> str:
        """Best-effort source string for trace/debug output."""
        meta = sem.get("metadata") or {}
        for key in ["source", "source_type", "memory_source"]:
            value = normalize_text(meta.get(key, ""))
            if value:
                return value
        tags = sem.get("tags") or []
        if tags:
            return normalize_text(tags[0]) or "unknown"
        return "unknown"

    def _score_semantic_for_query(self, query: str, sem: Dict[str, Any]) -> float:
        content = sem.get("content", "")
        try:
            return float(self._topk_by_similarity(query, [content], 1)[0][1]) if content else 0.0
        except Exception:
            return 0.0

    def _select_global_similarity_semantics(
        self, *, query: str, user_data: Dict[str, Any], excluded_semantic_ids: Set[str], top_k: int
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Select top-k semantics from the whole user's semantic bank by query similarity.

        This supplement is independent of situation_semantic_budget: it can add
        globally similar semantics even when keyword-matched Situations already
        exceed the budget. The final hard cap is applied later by
        _limit_situation_stage1_semantics().
        """
        top_k = max(0, int(top_k or 0))
        all_semantics = user_data.get("semantics", []) or []
        info: Dict[str, Any] = {
            "enabled": bool(self.always_supplement_global_semantics),
            "top_k": top_k,
            "before_exclusion_count": len(all_semantics),
            "after_exclusion_count": 0,
            "selected_count": 0,
            "ranking_method": "query_content_embedding_similarity",
        }
        if not self.always_supplement_global_semantics or top_k <= 0:
            return ([], info)
        excluded = {str(x) for x in excluded_semantic_ids or set() if x}
        candidates: List[Dict[str, Any]] = []
        for sem in all_semantics:
            sid = sem.get("memory_id")
            if sid and str(sid) in excluded:
                continue
            candidates.append(sem)
        info["after_exclusion_count"] = len(candidates)
        scored: List[Tuple[Dict[str, Any], float]] = []
        for sem in candidates:
            scored.append((sem, self._score_semantic_for_query(query, sem)))
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [sem for sem, _ in scored[:top_k]]
        info["selected_count"] = len(selected)
        info["selected_debug"] = [
            {
                "rank": rank + 1,
                "memory_id": sem.get("memory_id", ""),
                "source": self._semantic_debug_source(sem),
                "score": float(score),
                "content_preview": normalize_text(sem.get("content", ""))[:240],
            }
            for rank, (sem, score) in enumerate(scored[:top_k])
        ]
        return (dedup_preserve_order(selected), info)

    def _limit_situation_stage1_semantics(
        self,
        query: str,
        situation_semantics: List[Dict[str, Any]],
        situation_model: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """Optionally cap Situation stage-1 semantic candidates.

        If force_situation_semantic_budget is enabled, the merged candidate pool
        is capped to situation_semantic_budget. To avoid global-similarity
        supplements crowding out keyword-routed Situation evidence, the limiter
        first preserves top-N Situation-sourced semantics by query similarity,
        then fills the remaining slots from the merged similarity ranking.
        """
        combined = dedup_preserve_order((situation_semantics or []) + (situation_model or []))
        budget = int(self.situation_semantic_budget or 0)
        before = len(combined)
        min_keep = max(0, int(self.min_situation_semantics_preserved or 0))
        info: Dict[str, Any] = {
            "enabled": bool(self.force_situation_semantic_budget),
            "budget": budget,
            "before_count": before,
            "after_count": before,
            "applied": False,
            "ranking_method": "query_content_embedding_similarity_with_situation_min_keep",
            "min_situation_semantics_preserved": min_keep,
        }
        if not self.force_situation_semantic_budget or budget <= 0 or before <= budget:
            return (situation_semantics or [], situation_model or [], info)
        situation_scored: List[Tuple[Dict[str, Any], float]] = [
            (sem, self._score_semantic_for_query(query, sem))
            for sem in dedup_preserve_order(situation_semantics or [])
        ]
        situation_scored.sort(key=lambda x: x[1], reverse=True)
        preserve_count = min(min_keep, budget, len(situation_scored))
        preserved = [sem for sem, _ in situation_scored[:preserve_count]]
        preserved_keys = {self._semantic_dedup_key(sem) for sem in preserved}
        all_scored: List[Tuple[Dict[str, Any], float]] = [
            (sem, self._score_semantic_for_query(query, sem)) for sem in combined
        ]
        all_scored.sort(key=lambda x: x[1], reverse=True)
        kept: List[Dict[str, Any]] = list(preserved)
        kept_keys = set(preserved_keys)
        for sem, _score in all_scored:
            if len(kept) >= budget:
                break
            key = self._semantic_dedup_key(sem)
            if key in kept_keys:
                continue
            kept.append(sem)
            kept_keys.add(key)
        situation_keys = {self._semantic_dedup_key(sem) for sem in situation_semantics or []}
        limited_situation_semantics = [
            sem for sem in situation_semantics or [] if self._semantic_dedup_key(sem) in kept_keys
        ]
        limited_situation_model = [
            sem
            for sem in situation_model or []
            if self._semantic_dedup_key(sem) in kept_keys
            and self._semantic_dedup_key(sem) not in situation_keys
        ]
        kept_ranked_debug = []
        kept_key_set = {self._semantic_dedup_key(sem) for sem in kept}
        for rank, (sem, score) in enumerate(all_scored):
            key = self._semantic_dedup_key(sem)
            if key not in kept_key_set:
                continue
            kept_ranked_debug.append(
                {
                    "rank_in_merged_similarity_pool": rank + 1,
                    "memory_id": sem.get("memory_id", ""),
                    "source": self._semantic_debug_source(sem),
                    "from_keyword_situation": key in situation_keys,
                    "preserved_by_min_keep": key in preserved_keys,
                    "score": float(score),
                    "content_preview": normalize_text(sem.get("content", ""))[:240],
                }
            )
            if len(kept_ranked_debug) >= budget:
                break
        preserved_debug = []
        for rank, (sem, score) in enumerate(situation_scored[:preserve_count]):
            sem_key = self._semantic_dedup_key(sem)
            global_rank = None
            for idx, (global_sem, _global_score) in enumerate(all_scored):
                if self._semantic_dedup_key(global_sem) == sem_key:
                    global_rank = idx + 1
                    break
            preserved_debug.append(
                {
                    "situation_rank": rank + 1,
                    "rank_in_merged_similarity_pool": global_rank,
                    "memory_id": sem.get("memory_id", ""),
                    "source": self._semantic_debug_source(sem),
                    "score": float(score),
                    "content_preview": normalize_text(sem.get("content", ""))[:240],
                }
            )
        info.update(
            {
                "applied": True,
                "after_count": len(
                    dedup_preserve_order(limited_situation_semantics + limited_situation_model)
                ),
                "kept_situation_semantics_count": len(limited_situation_semantics),
                "kept_situation_model_count": len(limited_situation_model),
                "preserve_count_requested": min_keep,
                "preserve_count_actual": preserve_count,
                "preserved_situation_debug": preserved_debug,
                "kept_ranked_debug": kept_ranked_debug,
            }
        )
        return (limited_situation_semantics, limited_situation_model, info)

    def _build_situation_model_with_budget(
        self,
        user_data: Dict[str, Any],
        query: str,
        expected_answer_labels: List[str],
        excluded_semantic_ids: Set[str],
        budget: int,
    ) -> List[Dict[str, Any]]:
        if budget <= 0:
            return []
        semantic_items = user_data["semantics"]
        semantic_label_map = user_data["semantic_label_map"]
        expected_set = set(expected_answer_labels)
        bins: Dict[int, List[Tuple[Dict[str, Any], float]]] = defaultdict(list)
        for sem in semantic_items:
            sid = sem.get("memory_id")
            if not sid or sid in excluded_semantic_ids:
                continue
            labels = semantic_label_map.get(sid, set())
            overlap = len(labels.intersection(expected_set))
            if overlap <= 0:
                continue
            content = sem.get("content", "")
            sim = self._topk_by_similarity(query, [content], 1)[0][1] if content else 0.0
            bins[overlap].append((sem, sim))
        selected = []
        remaining = budget
        for overlap in sorted(bins.keys(), reverse=True):
            items = sorted(bins[overlap], key=lambda x: x[1], reverse=True)
            take = min(self.situation_model_top_k_per_bin, remaining, len(items))
            selected.extend([x[0] for x in items[:take]])
            remaining -= take
            if remaining <= 0:
                break
        if remaining > 0:
            used_ids = {x.get("memory_id") for x in selected if x.get("memory_id")}
            scored = []
            for sem in semantic_items:
                sid = sem.get("memory_id")
                if not sid or sid in excluded_semantic_ids or sid in used_ids:
                    continue
                content = sem.get("content", "")
                sim = self._topk_by_similarity(query, [content], 1)[0][1] if content else 0.0
                scored.append((sem, sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            selected.extend([x[0] for x in scored[:remaining]])
        return dedup_preserve_order(selected[:budget])

    def _build_full_situation_context_from_keywords(
        self, query: str, user_data: Dict[str, Any], mental_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        selected_keywords = mental_result.get("selected_keywords", {}) or {}
        situation_keywords = []
        for label, kws in selected_keywords.items():
            if label in {"Character", "Time"}:
                continue
            for kw in kws or []:
                kw = normalize_text(kw)
                if kw:
                    situation_keywords.append(kw)
        situation_keywords = dedup_preserve_order(situation_keywords)
        candidate_situations, kw_replacement = self._find_situations_for_keywords(
            situation_keywords, user_data
        )
        kept_situations, existing_semantics = self._collect_unique_semantics_from_situations(
            candidate_situations, user_data
        )
        semantic_budget = int(self.situation_semantic_budget or 0)
        existing_ids = {x.get("memory_id") for x in existing_semantics if x.get("memory_id")}
        situation_model: List[Dict[str, Any]] = []
        global_supplement, global_supplement_info = self._select_global_similarity_semantics(
            query=query,
            user_data=user_data,
            excluded_semantic_ids=existing_ids,
            top_k=self.global_semantic_supplement_top_k,
        )
        for sem in global_supplement:
            sid = sem.get("memory_id")
            if sid and sid not in existing_ids:
                existing_ids.add(sid)
                situation_model.append(sem)
        remaining_budget = semantic_budget - len(existing_semantics) - len(situation_model)
        situation_model_filter_info: Dict[str, Any] = {
            "requested_budget": max(0, remaining_budget),
            "selected_count": 0,
            "skipped_because_budget_full": remaining_budget <= 0,
        }
        if remaining_budget > 0:
            label_model = self._build_situation_model_with_budget(
                user_data=user_data,
                query=query,
                expected_answer_labels=mental_result.get("expected_answer_labels", []) or [],
                excluded_semantic_ids=existing_ids,
                budget=remaining_budget,
            )
            situation_model_filter_info["selected_count"] = len(label_model)
            for sem in label_model:
                sid = sem.get("memory_id")
                if sid and sid not in existing_ids:
                    existing_ids.add(sid)
                    situation_model.append(sem)
        keyword_existing_ids = {
            x.get("memory_id") for x in existing_semantics + situation_model if x.get("memory_id")
        }
        supplemented = self._supplement_semantics_from_query_keywords(
            query=query,
            query_label_keywords=mental_result.get("query_label_keywords", {}) or {},
            existing_semantic_ids=keyword_existing_ids,
            user_data=user_data,
        )
        situation_model = dedup_preserve_order(situation_model + supplemented)
        prelimit_count = len(dedup_preserve_order(existing_semantics + situation_model))
        return {
            "input_keywords": situation_keywords,
            "keyword_replacement_map": kw_replacement,
            "situations": kept_situations,
            "situation_semantics": existing_semantics,
            "situation_model": situation_model,
            "global_similarity_supplement": global_supplement_info,
            "situation_model_filter": situation_model_filter_info,
            "keyword_supplement": {"selected_count": len(supplemented)},
            "stage1_semantic_prelimit_count": prelimit_count,
        }

    def _decision_with_situation(
        self,
        exp_name: str,
        query: str,
        situation_evidence: List[Dict[str, Any]],
        situation_model_evidence: List[Dict[str, Any]],
        involves_time: bool,
    ) -> Tuple[Dict[str, Any], Dict[str, int], Dict[str, Any]]:
        prompt = self.prompts.SITUATION_DECISION_PROMPT.render(
            question=query,
            situation_evidence=format_semantic_lines(
                situation_evidence, with_timestamp=involves_time
            ),
            situation_model_evidence=format_semantic_lines(
                situation_model_evidence, with_timestamp=involves_time
            ),
        )
        return self._run_json_prompt(
            exp_name,
            prompt,
            default_response={
                "answer_determined": False,
                "proposed_answer": "",
                "supporting_semantic_contents": [],
                "need_expand_semantic_contents": [],
            },
            stage_name="situation_decision",
        )

    def _get_message_for_semantic(
        self, semantic_obj: Dict[str, Any], user_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Prefer directly linked matched message from current semantic version.
        Fallback to episode-derived message if absent.
        """
        message_map = user_data["message_map"]
        matched_message_id = semantic_obj.get("matched_message_id")
        if matched_message_id and matched_message_id in message_map:
            return message_map[matched_message_id]
        matched_message_content = semantic_obj.get("matched_message_content")
        if matched_message_content:
            for msg in message_map.values():
                if normalize_text(msg.get("content", "")) == normalize_text(
                    matched_message_content
                ):
                    return msg
        episode_map = user_data["episode_map"]
        source_eps = semantic_obj.get("source_episodes", []) or []
        for eid in source_eps:
            ep = episode_map.get(eid)
            if ep is None:
                continue
            msgs = ep.get("original_messages", []) or []
            if not msgs:
                continue
            sem_text = semantic_obj.get("content", "")
            best_msg = None
            best_sim = -1.0
            for msg in msgs:
                msg_text = msg.get("content", "")
                if not msg_text:
                    continue
                sim = self._topk_by_similarity(sem_text, [msg_text], 1)[0][1]
                if sim > best_sim:
                    best_sim = sim
                    best_msg = msg
            if best_msg is not None:
                return best_msg
        return None

    def _dedup_semantic_message_pairs_by_message(
        self, query: str, supporting_semantics: List[Dict[str, Any]], user_data: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        grouped: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any], float]]] = defaultdict(list)
        for sem in supporting_semantics:
            msg = self._get_message_for_semantic(sem, user_data)
            if msg is None:
                key = f"NO_MSG::{sem.get('memory_id')}"
                grouped[key].append((sem, {}, 0.0))
                continue
            mid = msg.get("message_id") or json.dumps(msg, sort_keys=True, ensure_ascii=False)
            content = sem.get("content", "")
            sim = self._topk_by_similarity(query, [content], 1)[0][1] if content else 0.0
            grouped[mid].append((sem, msg, sim))
        kept_semantics = []
        kept_messages = []
        for _, items in grouped.items():
            items.sort(key=lambda x: x[2], reverse=True)
            sem, msg, _ = items[0]
            kept_semantics.append(sem)
            kept_messages.append(msg if msg else {})
        return (kept_semantics, kept_messages)

    def _match_messages_for_semantics_without_dedup(
        self, supporting_semantics: List[Dict[str, Any]], user_data: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Keep every semantic selected for the final answer and attach its best
        matched original message. This is used by the safeguard path where the
        situation decision agent fails to select any semantic; in that case we
        must pass all stage-1 semantics to the final answer agent, not a
        message-deduplicated subset.
        """
        kept_semantics = dedup_preserve_order(supporting_semantics or [])
        matched_messages: List[Dict[str, Any]] = []
        for sem in kept_semantics:
            msg = self._get_message_for_semantic(sem, user_data)
            matched_messages.append(msg if msg else {})
        return (kept_semantics, matched_messages)

    def _append_extra_messages_for_query_keywords(
        self,
        query: str,
        query_label_keywords: Dict[str, List[Any]],
        supporting_semantics: List[Dict[str, Any]],
        matched_messages: List[Dict[str, Any]],
        user_data: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        existing_message_ids = set()
        for msg in matched_messages:
            mid = msg.get("message_id")
            if mid:
                existing_message_ids.add(mid)
        extra_msgs = self._supplement_messages_from_query_keywords(
            query=query,
            query_label_keywords=query_label_keywords,
            existing_message_ids=existing_message_ids,
            user_data=user_data,
        )
        for msg in extra_msgs:
            supporting_semantics.append(
                {
                    "memory_id": f"EXTRA_MSG::{msg.get('message_id', '')}",
                    "content": f"Supplementary evidence from message: {msg.get('content', '')}",
                    "timestamp": msg.get("timestamp"),
                    "created_at": msg.get("timestamp"),
                    "source_episodes": [msg.get("metadata", {}).get("episode_id")]
                    if isinstance(msg.get("metadata"), dict)
                    else [],
                }
            )
            matched_messages.append(msg)
        return (supporting_semantics, matched_messages)

    def _build_pair_prompt_text(
        self,
        supporting_semantics: List[Dict[str, Any]],
        matched_messages: List[Dict[str, Any]],
        involves_time: bool,
    ) -> str:
        pairs = []
        for idx, sem in enumerate(supporting_semantics):
            msg = matched_messages[idx] if idx < len(matched_messages) else {}
            pairs.append({"semantic": sem, "message": msg})
        return format_semantic_message_pairs(pairs, with_timestamp=involves_time)

    def _final_time_answer_with_situation(
        self,
        exp_name: str,
        query: str,
        supporting_semantics: List[Dict[str, Any]],
        matched_messages: List[Dict[str, Any]],
        involves_time: bool,
    ) -> Tuple[str, Dict[str, int], Dict[str, Any]]:
        pair_text = self._build_pair_prompt_text(
            supporting_semantics, matched_messages, involves_time
        )
        prompt = self.prompts.TIME_FINAL_ANSWER_PROMPT.render(
            question=query, semantic_message_pairs=pair_text
        )
        return self._run_text_prompt(
            exp_name, prompt, max_tokens=4096, stage_name="situation_time_final_answer"
        )

    def _final_inference_answer_with_situation(
        self,
        exp_name: str,
        query: str,
        supporting_semantics: List[Dict[str, Any]],
        matched_messages: List[Dict[str, Any]],
        involves_time: bool,
    ) -> Tuple[str, Dict[str, int], Dict[str, Any]]:
        pair_text = self._build_pair_prompt_text(
            supporting_semantics, matched_messages, involves_time
        )
        prompt = self.prompts.TRIMEM_INFERENCE_PROMPT.render(
            question=query, semantic_message_pairs=pair_text
        )
        raw_text, token_usage, trace = self._run_text_prompt(
            exp_name, prompt, max_tokens=4096, stage_name="situation_inference_final_answer"
        )
        answer = raw_text
        try:
            parsed = json.loads(raw_text)
            answer = parsed.get("answer", raw_text)
            trace["parsed_json"] = parsed
        except Exception:
            pass
        return (answer, token_usage, trace)

    def _final_factual_answer_with_situation(
        self,
        exp_name: str,
        query: str,
        supporting_semantics: List[Dict[str, Any]],
        matched_messages: List[Dict[str, Any]],
        involves_time: bool,
    ) -> Tuple[str, Dict[str, int], Dict[str, Any]]:
        pair_text = self._build_pair_prompt_text(
            supporting_semantics, matched_messages, involves_time
        )
        prompt = self.prompts.TRIMEM_FACTUAL_PROMPT.render(
            question=query, semantic_message_pairs=pair_text
        )
        raw_text, token_usage, trace = self._run_text_prompt(
            exp_name, prompt, max_tokens=4096, stage_name="situation_factual_final_answer"
        )
        answer = raw_text
        try:
            parsed = json.loads(raw_text)
            answer = parsed.get("answer", raw_text)
            trace["parsed_json"] = parsed
        except Exception:
            pass
        return (answer, token_usage, trace)

    def _map_semantic_contents_to_real_semantics(
        self,
        content_list: List[Any],
        user_data: Dict[str, Any],
        candidate_semantics: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        if not content_list:
            return []
        semantics = user_data["semantics"]
        resolved = []
        for item in content_list:
            if isinstance(item, int):
                if candidate_semantics is not None and 0 <= item < len(candidate_semantics):
                    resolved.append(candidate_semantics[item])
                    continue
                else:
                    continue
            if isinstance(item, str):
                raw = item.strip()
                if raw.isdigit():
                    idx = int(raw)
                    if candidate_semantics is not None and 0 <= idx < len(candidate_semantics):
                        resolved.append(candidate_semantics[idx])
                        continue
            content = normalize_text(item)
            if not content:
                continue
            exact = None
            for sem in semantics:
                if normalize_text(sem.get("content", "")) == content:
                    exact = sem
                    break
            if exact is not None:
                resolved.append(exact)
                continue
            scored = []
            for sem in semantics:
                s_text = sem.get("content", "")
                if not s_text:
                    continue
                try:
                    sim = self._topk_by_similarity(content, [s_text], 1)[0][1]
                except Exception:
                    sim = 0.0
                scored.append((sem, sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            if scored:
                resolved.append(scored[0][0])
        return dedup_preserve_order(resolved)

    def _run_situation_full(
        self,
        query: str,
        user_data: Dict[str, Any],
        mental_result: Dict[str, Any],
        exp_name: str = "full",
    ) -> Dict[str, Any]:
        context = self._build_full_situation_context_from_keywords(
            query=query, user_data=user_data, mental_result=mental_result
        )
        situation_semantics = context["situation_semantics"]
        situation_model = context["situation_model"]
        situation_semantics, situation_model, stage1_limit_info = (
            self._limit_situation_stage1_semantics(
                query=query,
                situation_semantics=situation_semantics,
                situation_model=situation_model,
            )
        )
        decision_resp, tok1, trace1 = self._decision_with_situation(
            exp_name=exp_name,
            query=query,
            situation_evidence=situation_semantics,
            situation_model_evidence=situation_model,
            involves_time=mental_result.get("involves_time", False),
        )
        trace1["global_similarity_supplement"] = context.get("global_similarity_supplement", {})
        trace1["situation_model_filter"] = context.get("situation_model_filter", {})
        trace1["keyword_supplement"] = context.get("keyword_supplement", {})
        trace1["stage1_semantic_prelimit_count"] = context.get("stage1_semantic_prelimit_count", 0)
        trace1["stage1_semantic_budget_limit"] = stage1_limit_info
        answer_determined = bool(decision_resp.get("answer_determined", False))
        supporting_semantic_contents = decision_resp.get("supporting_semantic_contents", []) or []
        need_expand_semantic_contents = decision_resp.get("need_expand_semantic_contents", []) or []
        final_answer = ""
        tok2 = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        trace2 = None
        supporting_semantics: List[Dict[str, Any]] = []
        matched_messages: List[Dict[str, Any]] = []
        candidate_support_pool = dedup_preserve_order(situation_semantics + situation_model)
        merged_semantic_contents = dedup_preserve_order(
            supporting_semantic_contents + need_expand_semantic_contents
        )
        supporting_semantics = self._map_semantic_contents_to_real_semantics(
            merged_semantic_contents, user_data, candidate_semantics=candidate_support_pool
        )
        fallback_to_stage1_all_semantics = False
        final_evidence_fallback_info = {
            "enabled": False,
            "reason": "agent_selected_semantics",
            "stage1_semantic_count": len(candidate_support_pool),
            "final_semantic_count_before_fallback": len(supporting_semantics),
        }
        if not supporting_semantics:
            fallback_to_stage1_all_semantics = True
            supporting_semantics, matched_messages = (
                self._match_messages_for_semantics_without_dedup(candidate_support_pool, user_data)
            )
            final_evidence_fallback_info.update(
                {
                    "enabled": True,
                    "reason": "no_semantic_selected_by_situation_agent",
                    "final_semantic_count_after_fallback": len(supporting_semantics),
                    "matched_message_count_after_fallback": len([m for m in matched_messages if m]),
                }
            )
        else:
            supporting_semantics, matched_messages = self._dedup_semantic_message_pairs_by_message(
                query=query, supporting_semantics=supporting_semantics, user_data=user_data
            )
            supporting_semantics, matched_messages = self._append_extra_messages_for_query_keywords(
                query=query,
                query_label_keywords=mental_result.get("query_label_keywords", {}) or {},
                supporting_semantics=supporting_semantics,
                matched_messages=matched_messages,
                user_data=user_data,
            )
            final_evidence_fallback_info.update(
                {
                    "final_semantic_count_after_fallback": len(supporting_semantics),
                    "matched_message_count_after_fallback": len([m for m in matched_messages if m]),
                }
            )
        trace1["final_evidence_fallback"] = final_evidence_fallback_info
        is_time_answer = query.strip().startswith("When") and "Time" in (
            mental_result.get("expected_answer_labels", []) or []
        )
        is_inference = self._is_inference_question(query)
        if is_time_answer:
            final_answer, tok2, trace2 = self._final_time_answer_with_situation(
                exp_name=exp_name,
                query=query,
                supporting_semantics=supporting_semantics,
                matched_messages=matched_messages,
                involves_time=True,
            )
        elif is_inference:
            final_answer, tok2, trace2 = self._final_inference_answer_with_situation(
                exp_name=exp_name,
                query=query,
                supporting_semantics=supporting_semantics,
                matched_messages=matched_messages,
                involves_time=mental_result.get("involves_time", False),
            )
        else:
            final_answer, tok2, trace2 = self._final_factual_answer_with_situation(
                exp_name=exp_name,
                query=query,
                supporting_semantics=supporting_semantics,
                matched_messages=matched_messages,
                involves_time=mental_result.get("involves_time", False),
            )
        if isinstance(trace2, dict):
            trace2["final_evidence_fallback"] = final_evidence_fallback_info
        return {
            "keyword_input": context["input_keywords"],
            "keyword_replacement_map": context["keyword_replacement_map"],
            "situations": context["situations"],
            "situation_semantics": situation_semantics,
            "situation_model": situation_model,
            "global_similarity_supplement": context.get("global_similarity_supplement", {}),
            "situation_model_filter": context.get("situation_model_filter", {}),
            "keyword_supplement": context.get("keyword_supplement", {}),
            "stage1_semantic_prelimit_count": context.get("stage1_semantic_prelimit_count", 0),
            "stage1_semantic_budget_limit": stage1_limit_info,
            "agent_response": {
                "answer_determined": answer_determined,
                "answer": final_answer,
                "proposed_answer": decision_resp.get("proposed_answer", ""),
                "supporting_semantic_contents": supporting_semantic_contents,
                "need_expand_semantic_contents": need_expand_semantic_contents,
                "fallback_to_stage1_all_semantics": fallback_to_stage1_all_semantics,
                "final_evidence_fallback": final_evidence_fallback_info,
            },
            "supporting_semantics": supporting_semantics,
            "matched_messages": matched_messages,
            "trace": {
                "situation_stage_1_decision": trace1,
                "situation_stage_2_final_answer": trace2,
            },
            "token_usage": {
                "stage1": tok1,
                "stage2": tok2,
                "total_tokens": tok1["total_tokens"] + tok2["total_tokens"],
            },
        }

    def _run_situation_no_mental(
        self,
        query: str,
        user_data: Dict[str, Any],
        no_mental_result: Dict[str, Any],
        exp_name: str = "no_mental",
    ) -> Dict[str, Any]:
        selected_semantics = no_mental_result.get("selected_semantics", []) or []
        if not selected_semantics:
            return {
                "clustered_semantics": [],
                "agent_response": {
                    "answer_determined": False,
                    "answer": "",
                    "proposed_answer": "",
                    "supporting_semantic_contents": [],
                    "need_expand_semantic_contents": [],
                },
                "supporting_semantics": [],
                "matched_messages": [],
                "trace": {},
                "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            }
        target_count = min(self.situation_model_top_k_total, max(len(selected_semantics), 10))
        selected_ids = {x.get("memory_id") for x in selected_semantics if x.get("memory_id")}
        scored = []
        for sem in user_data["semantics"]:
            sid = sem.get("memory_id")
            if sid in selected_ids:
                scored.append((sem, 1.0))
                continue
            content = sem.get("content", "")
            best = 0.0
            for center in selected_semantics:
                c_text = center.get("content", "")
                if not c_text or not content:
                    continue
                sim = self._topk_by_similarity(c_text, [content], 1)[0][1]
                best = max(best, sim)
            scored.append((sem, best))
        scored.sort(key=lambda x: x[1], reverse=True)
        clustered_semantics = [x[0] for x in scored[:target_count]]
        decision_resp, tok1, trace1 = self._decision_with_situation(
            exp_name=exp_name,
            query=query,
            situation_evidence=clustered_semantics,
            situation_model_evidence=[],
            involves_time=False,
        )
        answer_determined = bool(decision_resp.get("answer_determined", False))
        supporting_semantic_contents = decision_resp.get("supporting_semantic_contents", []) or []
        need_expand_semantic_contents = decision_resp.get("need_expand_semantic_contents", []) or []
        final_answer = ""
        tok2 = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        trace2 = None
        supporting_semantics: List[Dict[str, Any]] = []
        matched_messages: List[Dict[str, Any]] = []
        candidate_support_pool = clustered_semantics
        merged_semantic_contents = dedup_preserve_order(
            supporting_semantic_contents + need_expand_semantic_contents
        )
        supporting_semantics = self._map_semantic_contents_to_real_semantics(
            merged_semantic_contents, user_data, candidate_semantics=candidate_support_pool
        )
        fallback_to_stage1_all_semantics = False
        final_evidence_fallback_info = {
            "enabled": False,
            "reason": "agent_selected_semantics",
            "stage1_semantic_count": len(candidate_support_pool),
            "final_semantic_count_before_fallback": len(supporting_semantics),
        }
        if not supporting_semantics:
            fallback_to_stage1_all_semantics = True
            supporting_semantics, matched_messages = (
                self._match_messages_for_semantics_without_dedup(candidate_support_pool, user_data)
            )
            final_evidence_fallback_info.update(
                {
                    "enabled": True,
                    "reason": "no_semantic_selected_by_situation_agent",
                    "final_semantic_count_after_fallback": len(supporting_semantics),
                    "matched_message_count_after_fallback": len([m for m in matched_messages if m]),
                }
            )
        else:
            supporting_semantics, matched_messages = self._dedup_semantic_message_pairs_by_message(
                query=query, supporting_semantics=supporting_semantics, user_data=user_data
            )
            final_evidence_fallback_info.update(
                {
                    "final_semantic_count_after_fallback": len(supporting_semantics),
                    "matched_message_count_after_fallback": len([m for m in matched_messages if m]),
                }
            )
        trace1["final_evidence_fallback"] = final_evidence_fallback_info
        is_time_answer = query.strip().startswith("When")
        is_inference = self._is_inference_question(query)
        if is_time_answer:
            final_answer, tok2, trace2 = self._final_time_answer_with_situation(
                exp_name=exp_name,
                query=query,
                supporting_semantics=supporting_semantics,
                matched_messages=matched_messages,
                involves_time=True,
            )
        elif is_inference:
            final_answer, tok2, trace2 = self._final_inference_answer_with_situation(
                exp_name=exp_name,
                query=query,
                supporting_semantics=supporting_semantics,
                matched_messages=matched_messages,
                involves_time=False,
            )
        else:
            final_answer, tok2, trace2 = self._final_factual_answer_with_situation(
                exp_name=exp_name,
                query=query,
                supporting_semantics=supporting_semantics,
                matched_messages=matched_messages,
                involves_time=False,
            )
        if isinstance(trace2, dict):
            trace2["final_evidence_fallback"] = final_evidence_fallback_info
        return {
            "clustered_semantics": clustered_semantics,
            "agent_response": {
                "answer_determined": answer_determined,
                "answer": final_answer,
                "proposed_answer": decision_resp.get("proposed_answer", ""),
                "supporting_semantic_contents": supporting_semantic_contents,
                "need_expand_semantic_contents": need_expand_semantic_contents,
                "fallback_to_stage1_all_semantics": fallback_to_stage1_all_semantics,
                "final_evidence_fallback": final_evidence_fallback_info,
            },
            "supporting_semantics": supporting_semantics,
            "matched_messages": matched_messages,
            "trace": {
                "situation_stage_1_decision": trace1,
                "situation_stage_2_final_answer": trace2,
            },
            "token_usage": {
                "stage1": tok1,
                "stage2": tok2,
                "total_tokens": tok1["total_tokens"] + tok2["total_tokens"],
            },
        }

    def _run_situation_no_situation(
        self,
        query: str,
        user_data: Dict[str, Any],
        mental_result: Dict[str, Any],
        exp_name: str = "no_situation",
    ) -> Dict[str, Any]:
        resolved = mental_result.get("resolved_keyword_semantics", {}) or {}
        query_keywords = mental_result.get("extracted_query_keywords", []) or []
        kw_num = max(1, len(query_keywords))
        semantic_budget = max(1, self.situation_model_top_k_total // kw_num)
        selected_semantics = []
        for _, sems in resolved.items():
            sems = dedup_preserve_order(sems)
            if len(sems) <= semantic_budget:
                selected_semantics.extend(sems)
            else:
                scored = []
                for s in sems:
                    content = s.get("content", "")
                    sim = self._topk_by_similarity(query, [content], 1)[0][1] if content else 0.0
                    scored.append((s, sim))
                scored.sort(key=lambda x: x[1], reverse=True)
                selected_semantics.extend([x[0] for x in scored[:semantic_budget]])
        selected_semantics = dedup_preserve_order(selected_semantics)
        decision_resp, tok1, trace1 = self._decision_with_situation(
            exp_name=exp_name,
            query=query,
            situation_evidence=selected_semantics,
            situation_model_evidence=[],
            involves_time=mental_result.get("involves_time", False),
        )
        answer_determined = bool(decision_resp.get("answer_determined", False))
        supporting_semantic_contents = decision_resp.get("supporting_semantic_contents", []) or []
        need_expand_semantic_contents = decision_resp.get("need_expand_semantic_contents", []) or []
        final_answer = ""
        tok2 = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        trace2 = None
        supporting_semantics: List[Dict[str, Any]] = []
        matched_messages: List[Dict[str, Any]] = []
        candidate_support_pool = selected_semantics
        merged_semantic_contents = dedup_preserve_order(
            supporting_semantic_contents + need_expand_semantic_contents
        )
        supporting_semantics = self._map_semantic_contents_to_real_semantics(
            merged_semantic_contents, user_data, candidate_semantics=candidate_support_pool
        )
        fallback_to_stage1_all_semantics = False
        final_evidence_fallback_info = {
            "enabled": False,
            "reason": "agent_selected_semantics",
            "stage1_semantic_count": len(candidate_support_pool),
            "final_semantic_count_before_fallback": len(supporting_semantics),
        }
        if not supporting_semantics:
            fallback_to_stage1_all_semantics = True
            supporting_semantics, matched_messages = (
                self._match_messages_for_semantics_without_dedup(candidate_support_pool, user_data)
            )
            final_evidence_fallback_info.update(
                {
                    "enabled": True,
                    "reason": "no_semantic_selected_by_situation_agent",
                    "final_semantic_count_after_fallback": len(supporting_semantics),
                    "matched_message_count_after_fallback": len([m for m in matched_messages if m]),
                }
            )
        else:
            supporting_semantics, matched_messages = self._dedup_semantic_message_pairs_by_message(
                query=query, supporting_semantics=supporting_semantics, user_data=user_data
            )
            final_evidence_fallback_info.update(
                {
                    "final_semantic_count_after_fallback": len(supporting_semantics),
                    "matched_message_count_after_fallback": len([m for m in matched_messages if m]),
                }
            )
        trace1["final_evidence_fallback"] = final_evidence_fallback_info
        is_time_answer = query.strip().startswith("When") and "Time" in (
            mental_result.get("expected_answer_labels", []) or []
        )
        is_inference = self._is_inference_question(query)
        if is_time_answer:
            final_answer, tok2, trace2 = self._final_time_answer_with_situation(
                exp_name=exp_name,
                query=query,
                supporting_semantics=supporting_semantics,
                matched_messages=matched_messages,
                involves_time=True,
            )
        elif is_inference:
            final_answer, tok2, trace2 = self._final_inference_answer_with_situation(
                exp_name=exp_name,
                query=query,
                supporting_semantics=supporting_semantics,
                matched_messages=matched_messages,
                involves_time=mental_result.get("involves_time", False),
            )
        else:
            final_answer, tok2, trace2 = self._final_factual_answer_with_situation(
                exp_name=exp_name,
                query=query,
                supporting_semantics=supporting_semantics,
                matched_messages=matched_messages,
                involves_time=mental_result.get("involves_time", False),
            )
        if isinstance(trace2, dict):
            trace2["final_evidence_fallback"] = final_evidence_fallback_info
        return {
            "selected_semantics": selected_semantics,
            "agent_response": {
                "answer_determined": answer_determined,
                "answer": final_answer,
                "proposed_answer": decision_resp.get("proposed_answer", ""),
                "supporting_semantic_contents": supporting_semantic_contents,
                "need_expand_semantic_contents": need_expand_semantic_contents,
                "fallback_to_stage1_all_semantics": fallback_to_stage1_all_semantics,
                "final_evidence_fallback": final_evidence_fallback_info,
            },
            "supporting_semantics": supporting_semantics,
            "matched_messages": matched_messages,
            "trace": {
                "situation_stage_1_decision": trace1,
                "situation_stage_2_final_answer": trace2,
            },
            "token_usage": {
                "stage1": tok1,
                "stage2": tok2,
                "total_tokens": tok1["total_tokens"] + tok2["total_tokens"],
            },
        }

    def _run_full_pipeline_for_query(self, query: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
        self._mark_query("full")
        mental = self._run_mental_full(query, user_data, exp_name="full")
        situation = self._run_situation_full(query, user_data, mental, exp_name="full")
        return {
            "mental": mental,
            "situation": situation,
            "episode": None,
            "response": situation["agent_response"].get("answer", ""),
            "trace": {**mental.get("trace", {}), **situation.get("trace", {})},
        }

    def _run_no_mental_pipeline_for_query(
        self, query: str, user_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        self._mark_query("no_mental")
        mental = self._run_mental_no_mental(query, user_data, exp_name="no_mental")
        situation = self._run_situation_no_mental(query, user_data, mental, exp_name="no_mental")
        return {
            "mental": mental,
            "situation": situation,
            "episode": None,
            "response": situation["agent_response"].get("answer", ""),
            "trace": {**mental.get("trace", {}), **situation.get("trace", {})},
        }

    def _run_no_situation_pipeline_for_query(
        self, query: str, user_data: Dict[str, Any], full_mental: Dict[str, Any]
    ) -> Dict[str, Any]:
        self._mark_query("no_situation")
        situation = self._run_situation_no_situation(
            query, user_data, full_mental, exp_name="no_situation"
        )
        return {
            "mental": full_mental,
            "situation": situation,
            "episode": None,
            "response": situation["agent_response"].get("answer", ""),
            "trace": {**full_mental.get("trace", {}), **situation.get("trace", {})},
        }
