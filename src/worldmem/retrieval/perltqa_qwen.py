from __future__ import annotations

import json
import logging
import re
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from worldmem.clients import EmbeddingClient, LLMClient
from worldmem.config import MemoryConfig

logger = logging.getLogger(__name__)
from worldmem.prompts import perltqa_qwen as prompt_profile
from worldmem.prompts.perltqa_qwen import PERLTQA_CATEGORY_ALIASES
from worldmem.retrieval.locomo_qwen import Searcher as SharedSearcher
from worldmem.retrieval.locomo_qwen import (
    dedup_preserve_order,
    format_message_lines,
    format_semantic_message_pairs,
    normalize_text,
)


def sanitize_owner_id(text: str, fallback: str) -> str:
    text = (text or "").strip() or fallback
    text = re.sub("\\s+", "_", text)
    text = re.sub("[^A-Za-z0-9_\\-\\.]+", "_", text)
    text = re.sub("_+", "_", text).strip("_")
    return text or fallback


def make_perltqa_owner_id(character_key: str, idx: int, character_name: str, fmt: str) -> str:
    key = sanitize_owner_id(str(character_key or idx), f"character_{idx}")
    name = sanitize_owner_id(character_name or character_key or str(idx), f"character_{idx}")
    if fmt == "name":
        return name
    if fmt == "index":
        return f"character_{idx}"
    if fmt == "key":
        return key
    if fmt == "name_index":
        return f"{name}_{idx}"
    return f"{name}_{key}"


def normalize_perltqa_category(value: Any) -> str:
    text = normalize_text(value).strip()
    low = text.lower().replace("_", " ").replace("-", " ")
    low = re.sub("\\s+", " ", low).strip()
    if low in PERLTQA_CATEGORY_ALIASES:
        return PERLTQA_CATEGORY_ALIASES[low]
    return text or "Unknown"


def parse_perltqa_category_filter(categories_arg: str) -> Optional[Set[str]]:
    if not categories_arg:
        return None
    parts = []
    for raw in categories_arg.split(","):
        raw = raw.strip()
        if not raw:
            continue
        parts.append(normalize_perltqa_category(raw))
    cats = {x for x in parts if x}
    return cats or None


def _iter_perltqa_character_items(raw_dataset: Any) -> List[Tuple[str, Dict[str, Any]]]:
    """Normalize PerLTQA QA JSON into [(character_key, character_payload)]."""
    if isinstance(raw_dataset, list):
        items: List[Tuple[str, Dict[str, Any]]] = []
        for entry in raw_dataset:
            if isinstance(entry, dict):
                for key, payload in entry.items():
                    if isinstance(payload, dict):
                        items.append((str(key), payload))
        return items
    if isinstance(raw_dataset, dict):
        return [(str(k), v) for k, v in raw_dataset.items() if isinstance(v, dict)]
    return []


def _iter_qas_for_perltqa_category(
    category_name: str, category_payload: Any
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Return [(source_id, qa_record)] from one PerLTQA category.

    profile: list[qa]
    social_relationship/events/dialogues: list[{source_id: [qa, ...]}]
    """
    out: List[Tuple[str, Dict[str, Any]]] = []
    if isinstance(category_payload, list):
        for item in category_payload:
            if isinstance(item, dict) and "Question" in item:
                out.append(("", item))
                continue
            if isinstance(item, dict):
                for source_id, qas in item.items():
                    if isinstance(qas, dict) and "Question" in qas:
                        out.append((str(source_id), qas))
                    elif isinstance(qas, list):
                        for qa in qas:
                            if isinstance(qa, dict):
                                out.append((str(source_id), qa))
            elif isinstance(item, list):
                for qa in item:
                    if isinstance(qa, dict):
                        out.append(("", qa))
    elif isinstance(category_payload, dict):
        for source_id, qas in category_payload.items():
            if isinstance(qas, dict) and "Question" in qas:
                out.append((str(source_id), qas))
            elif isinstance(qas, list):
                for qa in qas:
                    if isinstance(qa, dict):
                        out.append((str(source_id), qa))
    return out


def flatten_perltqa_qa_dataset(
    raw_dataset: Any,
    *,
    owner_id_format: str = "name_key",
    category_filter: Optional[Set[str]] = None,
    character_limit: int = 0,
    character_offset: int = 0,
    include_character_regex: Optional[str] = None,
    limit_qa_per_character: int = 0,
) -> List[Dict[str, Any]]:
    characters = _iter_perltqa_character_items(raw_dataset)
    pattern = re.compile(include_character_regex) if include_character_regex else None
    flattened: List[Dict[str, Any]] = []
    for raw_idx, (character_key, payload) in enumerate(characters):
        if raw_idx < max(0, character_offset):
            continue
        character_name = character_key
        owner_id = make_perltqa_owner_id(character_key, raw_idx, character_name, owner_id_format)
        if pattern and (
            not (
                pattern.search(character_key)
                or pattern.search(character_name)
                or pattern.search(owner_id)
            )
        ):
            continue
        qas: List[Dict[str, Any]] = []
        for raw_cat in ["profile", "social_relationship", "events", "dialogues"]:
            category_name = normalize_perltqa_category(raw_cat)
            if category_filter and category_name not in category_filter:
                continue
            for source_id, qa in _iter_qas_for_perltqa_category(
                category_name, payload.get(raw_cat, [])
            ):
                question = qa.get("Question", qa.get("question", ""))
                answer = qa.get("Answer", qa.get("answer", ""))
                if not normalize_text(question):
                    continue
                qas.append(
                    {
                        "question": question,
                        "answer": answer,
                        "category": category_name,
                        "category_name": category_name,
                        "reference_memory": qa.get(
                            "Reference Memory", qa.get("reference_memory", "")
                        ),
                        "memory_anchors": qa.get("Memory Anchors", qa.get("memory_anchors", [])),
                        "qa_source_id": source_id,
                        "character": character_name,
                        "character_key": character_key,
                        "owner_id": owner_id,
                    }
                )
        if limit_qa_per_character and limit_qa_per_character > 0:
            qas = qas[:limit_qa_per_character]
        flattened.append(
            {
                "character": character_name,
                "character_key": character_key,
                "owner_id": owner_id,
                "qa": qas,
                "raw_index": raw_idx,
            }
        )
        if character_limit and character_limit > 0 and (len(flattened) >= character_limit):
            break
    return flattened


class BaseSearcher(SharedSearcher):
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

    def _run_json_prompt(
        self, exp_name: str, prompt_text: str, default_response: Dict[str, Any], stage_name: str
    ) -> Tuple[Dict[str, Any], Dict[str, int], Dict[str, Any]]:
        call_cfg = self._llm_call_config_for_stage(stage_name)
        if hasattr(self.llm_client, "generate_json_response_with_usage"):
            out, llm_resp = self.llm_client.generate_json_response_with_usage(
                prompt=prompt_text,
                system_prompt=None,
                temperature=call_cfg["temperature"],
                max_tokens=4096,
                default_response=default_response,
                max_retries=3,
                enable_thinking=call_cfg["enable_thinking"],
                top_p=call_cfg["top_p"],
                top_k=call_cfg["top_k"],
                min_p=call_cfg["min_p"],
            )
            llm_usage = getattr(llm_resp, "usage", {}) if llm_resp is not None else {}
        else:
            out = self.llm_client.generate_json_response(
                prompt=prompt_text,
                system_prompt=None,
                temperature=call_cfg["temperature"],
                max_tokens=4096,
                default_response=default_response,
                enable_thinking=call_cfg["enable_thinking"],
            )
            llm_resp = None
            llm_usage = {}
        raw_out = out
        if not isinstance(out, dict):
            logger.warning(
                "[%s] JSON response type mismatch: expected dict, got %s. Fallback to default_response. Raw response: %s",
                stage_name,
                type(out).__name__,
                json.dumps(out, ensure_ascii=False) if isinstance(out, (dict, list)) else repr(out),
            )
            out = default_response
        for k, v in default_response.items():
            if k not in out:
                logger.warning(
                    "[%s] JSON response missing key %r. Fill with default value.", stage_name, k
                )
                out[k] = v
        out_text = json.dumps(out, ensure_ascii=False)
        token_usage = self._record_tokens(exp_name, prompt_text, out_text, llm_usage=llm_usage)
        trace = {
            "stage": stage_name,
            "prompt": prompt_text,
            "response": out,
            "raw_response": raw_out,
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
        semantic_budget = self.situation_semantic_budget
        existing_ids = {x.get("memory_id") for x in existing_semantics if x.get("memory_id")}
        if len(existing_semantics) >= semantic_budget:
            situation_model = []
        else:
            remaining_budget = semantic_budget - len(existing_semantics)
            situation_model = self._build_situation_model_with_budget(
                user_data=user_data,
                query=query,
                expected_answer_labels=mental_result.get("expected_answer_labels", []) or [],
                excluded_semantic_ids=existing_ids,
                budget=remaining_budget,
            )
        supplemented = self._supplement_semantics_from_query_keywords(
            query=query,
            query_label_keywords=mental_result.get("query_label_keywords", {}) or {},
            existing_semantic_ids=existing_ids,
            user_data=user_data,
        )
        situation_model = dedup_preserve_order(situation_model + supplemented)
        return {
            "input_keywords": situation_keywords,
            "keyword_replacement_map": kw_replacement,
            "situations": kept_situations,
            "situation_semantics": existing_semantics,
            "situation_model": situation_model,
        }

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
        if not supporting_semantics:
            supporting_semantics = candidate_support_pool
        supporting_semantics, matched_messages = self._dedup_semantic_message_pairs_by_message(
            query=query, supporting_semantics=supporting_semantics, user_data=user_data
        )
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
        return {
            "keyword_input": context["input_keywords"],
            "keyword_replacement_map": context["keyword_replacement_map"],
            "situations": context["situations"],
            "situation_semantics": situation_semantics,
            "situation_model": situation_model,
            "agent_response": {
                "answer_determined": answer_determined,
                "answer": final_answer,
                "proposed_answer": decision_resp.get("proposed_answer", ""),
                "supporting_semantic_contents": supporting_semantic_contents,
                "need_expand_semantic_contents": need_expand_semantic_contents,
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
        if not supporting_semantics:
            supporting_semantics = candidate_support_pool
        supporting_semantics, matched_messages = self._dedup_semantic_message_pairs_by_message(
            query=query, supporting_semantics=supporting_semantics, user_data=user_data
        )
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
        return {
            "clustered_semantics": clustered_semantics,
            "agent_response": {
                "answer_determined": answer_determined,
                "answer": final_answer,
                "proposed_answer": decision_resp.get("proposed_answer", ""),
                "supporting_semantic_contents": supporting_semantic_contents,
                "need_expand_semantic_contents": need_expand_semantic_contents,
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
        if not supporting_semantics:
            supporting_semantics = candidate_support_pool
        supporting_semantics, matched_messages = self._dedup_semantic_message_pairs_by_message(
            query=query, supporting_semantics=supporting_semantics, user_data=user_data
        )
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
        return {
            "selected_semantics": selected_semantics,
            "agent_response": {
                "answer_determined": answer_determined,
                "answer": final_answer,
                "proposed_answer": decision_resp.get("proposed_answer", ""),
                "supporting_semantic_contents": supporting_semantic_contents,
                "need_expand_semantic_contents": need_expand_semantic_contents,
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


class Searcher(BaseSearcher):
    """WorldMemorySearcher with PerLTQA-specific evidence retrieval/stitching."""

    SOURCE_PROFILE = "profile"
    SOURCE_SOCIAL = "social_relationship"
    SOURCE_EVENTS = "events"

    def __init__(
        self,
        *args,
        final_evidence_scope: str = "all",
        restrict_candidates_by_query_characters: bool = True,
        situation_model_require_label_overlap: bool = False,
        force_situation_semantic_budget: bool = True,
        always_supplement_global_semantics: bool = True,
        global_semantic_supplement_top_k: int = 60,
        min_situation_semantics_preserved: int = 20,
        final_source_info_top_k: int = 3,
        final_answer_parse_max_retries: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        scope = normalize_text(final_evidence_scope).lower()
        if scope not in {"all", "predicted_category"}:
            raise ValueError("final_evidence_scope must be either 'all' or 'predicted_category'")
        self.final_evidence_scope = scope
        self.restrict_candidates_by_query_characters = bool(restrict_candidates_by_query_characters)
        self.situation_model_require_label_overlap = bool(situation_model_require_label_overlap)
        self.force_situation_semantic_budget = bool(force_situation_semantic_budget)
        self.always_supplement_global_semantics = bool(always_supplement_global_semantics)
        self.global_semantic_supplement_top_k = max(0, int(global_semantic_supplement_top_k or 0))
        self.min_situation_semantics_preserved = max(0, int(min_situation_semantics_preserved or 0))
        self.final_source_info_top_k = int(final_source_info_top_k or 0)
        self.final_answer_parse_max_retries = max(1, int(final_answer_parse_max_retries or 1))

    def _semantic_source(self, sem: Dict[str, Any]) -> str:
        meta = sem.get("metadata") or {}
        source = normalize_text(meta.get("source") or "")
        if source in {self.SOURCE_PROFILE, self.SOURCE_SOCIAL, self.SOURCE_EVENTS}:
            return source
        tags = sem.get("tags") or []
        for tag in tags:
            tag = normalize_text(tag)
            if tag in {self.SOURCE_PROFILE, self.SOURCE_SOCIAL, self.SOURCE_EVENTS}:
                return tag
        return source or "unknown"

    def _load_user_memories(self, user_id: str) -> Dict[str, Any]:
        data = super()._load_user_memories(user_id)
        episode_to_block_map: Dict[str, Dict[str, Any]] = {}
        for block in data.get("message_blocks", []) or []:
            eid = block.get("episode_id") or (block.get("metadata") or {}).get("episode_id")
            if eid and eid not in episode_to_block_map:
                episode_to_block_map[eid] = block
        semantic_source_map: Dict[str, str] = {}
        episode_to_semantics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        social_semantics = []
        profile_semantics = []
        event_semantics = []
        for sem in data.get("semantics", []) or []:
            sid = sem.get("memory_id")
            source = self._semantic_source(sem)
            if sid:
                semantic_source_map[sid] = source
            for ep_id in sem.get("source_episodes", []) or []:
                if ep_id:
                    episode_to_semantics[ep_id].append(sem)
            meta_ep = (sem.get("metadata") or {}).get("episode_id")
            if meta_ep:
                episode_to_semantics[meta_ep].append(sem)
            if source == self.SOURCE_SOCIAL:
                social_semantics.append(sem)
            elif source == self.SOURCE_PROFILE:
                profile_semantics.append(sem)
            elif source == self.SOURCE_EVENTS:
                event_semantics.append(sem)
        episode_to_semantics = {
            ep_id: dedup_preserve_order(sems) for ep_id, sems in episode_to_semantics.items()
        }
        data["episode_to_block_map"] = episode_to_block_map
        data["episode_to_semantics"] = episode_to_semantics
        data["semantic_source_map"] = semantic_source_map
        data["social_semantics"] = social_semantics
        data["profile_semantics"] = profile_semantics
        data["event_semantics"] = event_semantics
        return data

    def _normalize_query_memory_category(self, value: Any) -> str:
        text = normalize_text(value).lower().replace("-", "_").replace(" ", "_")
        if text in {"profile", "profiles", "identity", "appearance", "career", "education"}:
            return "profile"
        if text in {
            "social_relationship",
            "social",
            "relationship",
            "relationships",
            "social_relationships",
        }:
            return "social_relationship"
        if text in {
            "event",
            "events",
            "dialogue",
            "dialogues",
            "dialog",
            "dialogs",
            "events_dialogues",
            "event_dialogue",
        }:
            return "events"
        return "unknown"

    def _semantic_matches_query_category(self, sem: Dict[str, Any], category: str) -> bool:
        category = self._normalize_query_memory_category(category)
        source = self._semantic_source(sem)
        if category == "profile":
            return source == self.SOURCE_PROFILE
        if category == "social_relationship":
            return source == self.SOURCE_SOCIAL
        if category == "events":
            return source == self.SOURCE_EVENTS
        return True

    @staticmethod
    def _normalize_character_match_text(value: Any) -> str:
        text = normalize_text(value).lower()
        text = re.sub("[^a-z0-9]+", " ", text)
        return re.sub("\\s+", " ", text).strip()

    def _extract_query_characters_from_mental(self, mental_result: Dict[str, Any]) -> List[str]:
        query_label_keywords = mental_result.get("query_label_keywords", {}) or {}
        chars: List[str] = []
        for label, values in query_label_keywords.items():
            if normalize_text(label).lower() != "character":
                continue
            if not isinstance(values, list):
                values = [values]
            for value in values:
                if value is None:
                    continue
                text = normalize_text(value)
                if text:
                    chars.append(text)
        return dedup_preserve_order(chars)

    def _character_candidate_strings_for_semantic(self, sem: Dict[str, Any]) -> List[str]:
        candidates: List[str] = []
        for ann in sem.get("label_annotations", []) or []:
            if normalize_text(ann.get("label", "")).lower() == "character":
                for kw in ann.get("keywords", []) or []:
                    kw_text = normalize_text(kw)
                    if not kw_text:
                        continue
                    candidates.append(kw_text)
                    if ":" in kw_text:
                        tail = normalize_text(kw_text.split(":", 1)[1])
                        if tail:
                            candidates.append(tail)
        content = normalize_text(sem.get("content", ""))
        if content:
            candidates.append(content)
        msg_content = normalize_text(sem.get("matched_message_content", ""))
        if msg_content:
            candidates.append(msg_content)
        return dedup_preserve_order(candidates)

    def _text_matches_character(self, text: Any, character: str) -> bool:
        lhs = self._normalize_character_match_text(text)
        rhs = self._normalize_character_match_text(character)
        if not lhs or not rhs:
            return False
        if re.search("(^|\\s)" + re.escape(rhs) + "($|\\s)", lhs):
            return True
        if rhs in lhs or lhs in rhs:
            return True
        rhs_tokens = rhs.split()
        if len(rhs_tokens) > 1 and rhs_tokens[-1] and (rhs_tokens[-1] in lhs):
            return True
        return False

    def _semantic_mentions_character(self, sem: Dict[str, Any], character: str) -> bool:
        return any(
            (
                self._text_matches_character(candidate, character)
                for candidate in self._character_candidate_strings_for_semantic(sem)
            )
        )

    def _episode_contains_all_query_characters(
        self, episode_id: str, query_characters: List[str], user_data: Dict[str, Any]
    ) -> bool:
        if not episode_id or not query_characters:
            return False
        ep = user_data.get("episode_map", {}).get(episode_id, {}) or {}
        block = user_data.get("episode_to_block_map", {}).get(episode_id, {}) or {}
        source_text_parts: List[str] = [
            ep.get("title", ""),
            ep.get("content", ""),
            json.dumps(ep.get("metadata", {}) or {}, ensure_ascii=False),
            block.get("content", ""),
        ]
        for msg in block.get("messages", []) or []:
            source_text_parts.append(msg.get("role", ""))
            source_text_parts.append(msg.get("content", ""))
        source_text = "\n".join([normalize_text(x) for x in source_text_parts if normalize_text(x)])
        episode_semantics = user_data.get("episode_to_semantics", {}).get(episode_id, []) or []
        for required_char in query_characters:
            found = False
            if source_text and self._text_matches_character(source_text, required_char):
                found = True
            if not found:
                for sem in episode_semantics:
                    if self._semantic_mentions_character(sem, required_char):
                        found = True
                        break
            if not found:
                return False
        return True

    def _semantic_allowed_by_query_characters(
        self, sem: Dict[str, Any], query_characters: List[str], user_data: Dict[str, Any]
    ) -> bool:
        if not query_characters:
            return True
        episode_ids = []
        for ep_id in sem.get("source_episodes", []) or []:
            if ep_id:
                episode_ids.append(ep_id)
        meta_ep = (sem.get("metadata") or {}).get("episode_id")
        if meta_ep:
            episode_ids.append(meta_ep)
        episode_ids = dedup_preserve_order(episode_ids)
        if episode_ids:
            return any(
                (
                    self._episode_contains_all_query_characters(ep_id, query_characters, user_data)
                    for ep_id in episode_ids
                )
            )
        return all((self._semantic_mentions_character(sem, ch) for ch in query_characters))

    def _filter_semantics_by_query_characters(
        self,
        semantics: List[Dict[str, Any]],
        query_characters: List[str],
        user_data: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        before = len(semantics or [])
        if not self.restrict_candidates_by_query_characters or not query_characters:
            return (
                semantics or [],
                {
                    "enabled": bool(self.restrict_candidates_by_query_characters),
                    "applied": False,
                    "query_characters": query_characters,
                    "before_count": before,
                    "after_count": before,
                },
            )
        filtered = [
            sem
            for sem in semantics or []
            if self._semantic_allowed_by_query_characters(sem, query_characters, user_data)
        ]
        return (
            filtered,
            {
                "enabled": True,
                "applied": True,
                "query_characters": query_characters,
                "before_count": before,
                "after_count": len(filtered),
            },
        )

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
        if not self.situation_model_require_label_overlap:
            scored: List[Tuple[Dict[str, Any], float]] = []
            for sem in semantic_items:
                sid = sem.get("memory_id")
                if not sid or sid in excluded_semantic_ids:
                    continue
                content = sem.get("content", "")
                sim = self._topk_by_similarity(query, [content], 1)[0][1] if content else 0.0
                scored.append((sem, sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            return dedup_preserve_order([x[0] for x in scored[:budget]])
        expected_set = {
            normalize_text(x) for x in expected_answer_labels or [] if normalize_text(x)
        }
        if not expected_set:
            return []
        bins: Dict[int, List[Tuple[Dict[str, Any], float]]] = defaultdict(list)
        for sem in semantic_items:
            sid = sem.get("memory_id")
            if not sid or sid in excluded_semantic_ids:
                continue
            labels = {
                normalize_text(x) for x in semantic_label_map.get(sid, set()) if normalize_text(x)
            }
            overlap = len(labels.intersection(expected_set))
            if overlap <= 0:
                continue
            content = sem.get("content", "")
            sim = self._topk_by_similarity(query, [content], 1)[0][1] if content else 0.0
            bins[overlap].append((sem, sim))
        selected: List[Dict[str, Any]] = []
        remaining = budget
        for overlap in sorted(bins.keys(), reverse=True):
            items = sorted(bins[overlap], key=lambda x: x[1], reverse=True)
            take = min(self.situation_model_top_k_per_bin, remaining, len(items))
            selected.extend([x[0] for x in items[:take]])
            remaining -= take
            if remaining <= 0:
                break
        return dedup_preserve_order(selected[:budget])

    def _semantic_dedup_key(self, sem: Dict[str, Any]) -> str:
        sid = sem.get("memory_id")
        if sid:
            return str(sid)
        try:
            return json.dumps(sem, sort_keys=True, ensure_ascii=False)
        except Exception:
            return repr(sem)

    def _select_global_similarity_semantics(
        self,
        *,
        query: str,
        user_data: Dict[str, Any],
        excluded_semantic_ids: Set[str],
        top_k: int,
        query_characters: Optional[List[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Select top-k semantics from the whole user's semantic bank by query similarity.

        This supplement is intentionally independent of the current
        situation_semantic_budget.  It can add globally similar semantics even when
        the keyword-matched Situations already exceed the budget; the final hard cap
        is handled later by _limit_situation_stage1_semantics().
        """
        top_k = max(0, int(top_k or 0))
        all_semantics = user_data.get("semantics", []) or []
        info: Dict[str, Any] = {
            "enabled": bool(self.always_supplement_global_semantics),
            "top_k": top_k,
            "before_exclusion_count": len(all_semantics),
            "after_exclusion_count": 0,
            "before_character_filter_count": 0,
            "after_character_filter_count": 0,
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
        info["before_character_filter_count"] = len(candidates)
        candidates, char_filter_info = self._filter_semantics_by_query_characters(
            candidates, query_characters or [], user_data
        )
        info["after_character_filter_count"] = len(candidates)
        info["character_filter"] = char_filter_info
        scored: List[Tuple[Dict[str, Any], float]] = []
        for sem in candidates:
            content = sem.get("content", "")
            try:
                sim = self._topk_by_similarity(query, [content], 1)[0][1] if content else 0.0
            except Exception:
                sim = 0.0
            scored.append((sem, float(sim)))
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [sem for sem, _ in scored[:top_k]]
        info["selected_count"] = len(selected)
        info["selected_debug"] = [
            {
                "memory_id": sem.get("memory_id", ""),
                "source": self._semantic_source(sem),
                "score": float(score),
                "content_preview": normalize_text(sem.get("content", ""))[:240],
            }
            for sem, score in scored[:top_k]
        ]
        return (dedup_preserve_order(selected), info)

    def _score_semantic_for_query(self, query: str, sem: Dict[str, Any]) -> float:
        content = sem.get("content", "")
        try:
            return float(self._topk_by_similarity(query, [content], 1)[0][1]) if content else 0.0
        except Exception:
            return 0.0

    def _limit_situation_stage1_semantics(
        self,
        query: str,
        situation_semantics: List[Dict[str, Any]],
        situation_model: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """Optionally enforce situation_semantic_budget before Situation stage-1.

        When enabled, the final candidate pool is capped to
        situation_semantic_budget.  To avoid losing keyword-routed Situation
        evidence after adding global-similarity supplements, the limiter first
        preserves the top-N Situation-sourced semantics by query similarity, then
        fills the remaining budget with the highest-similarity semantics from the
        merged pool.
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
                    "source": self._semantic_source(sem),
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
            global_rank = None
            sem_key = self._semantic_dedup_key(sem)
            for idx, (global_sem, _global_score) in enumerate(all_scored):
                if self._semantic_dedup_key(global_sem) == sem_key:
                    global_rank = idx + 1
                    break
            preserved_debug.append(
                {
                    "situation_rank": rank + 1,
                    "rank_in_merged_similarity_pool": global_rank,
                    "memory_id": sem.get("memory_id", ""),
                    "source": self._semantic_source(sem),
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

    def _format_classified_semantic_evidence(
        self, candidate_semantics: List[Dict[str, Any]], *, with_timestamp: bool = True
    ) -> str:
        groups = [
            ("[Profile Semantic Evidence]", self.SOURCE_PROFILE),
            ("[Social Relationship Semantic Evidence]", self.SOURCE_SOCIAL),
            ("[Events / Dialogues Semantic Evidence]", self.SOURCE_EVENTS),
            ("[Other Semantic Evidence]", "other"),
        ]
        by_group: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {key: [] for _, key in groups}
        for idx, sem in enumerate(candidate_semantics or []):
            source = self._semantic_source(sem)
            key = (
                source
                if source in {self.SOURCE_PROFILE, self.SOURCE_SOCIAL, self.SOURCE_EVENTS}
                else "other"
            )
            by_group[key].append((idx, sem))
        sections: List[str] = []
        for title, key in groups:
            items = by_group.get(key, [])
            if not items:
                continue
            lines = [title]
            for idx, sem in items:
                ts = sem.get("timestamp") or sem.get("created_at") or ""
                content = sem.get("content", "")
                if key == self.SOURCE_EVENTS and with_timestamp and ts:
                    lines.append(f"Index {idx}: [{ts}] {content}")
                else:
                    lines.append(f"Index {idx}: {content}")
            sections.append("\n".join(lines))
        return "\n\n".join(sections) if sections else "(none)"

    def _decision_with_situation(
        self,
        exp_name: str,
        query: str,
        situation_evidence: List[Dict[str, Any]],
        situation_model_evidence: List[Dict[str, Any]],
        involves_time: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, int], Dict[str, Any]]:
        candidate_pool = dedup_preserve_order(
            (situation_evidence or []) + (situation_model_evidence or [])
        )
        classified_evidence = self._format_classified_semantic_evidence(
            candidate_pool, with_timestamp=True
        )
        prompt = self.prompts.PERLTQA_SITUATION_DECISION_WITH_CATEGORY_PROMPT.render(
            question=query, classified_semantic_evidence=classified_evidence
        )
        default_response = {
            "query_memory_category": "events",
            "answer_determined": False,
            "proposed_answer": "",
            "supporting_semantic_indices": [],
            "supporting_semantic_contents": [],
            "need_expand_semantic_indices": [],
            "need_expand_semantic_contents": [],
        }
        resp, tok, trace = self._run_json_prompt(
            exp_name, prompt, default_response=default_response, stage_name="situation_decision"
        )
        resp["query_memory_category"] = self._normalize_query_memory_category(
            resp.get("query_memory_category", "events")
        )
        trace["candidate_pool_size"] = len(candidate_pool)
        trace["classified_semantic_evidence"] = classified_evidence
        return (resp, tok, trace)

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
        query_characters = self._extract_query_characters_from_mental(mental_result)
        existing_semantics, character_filter_info = self._filter_semantics_by_query_characters(
            existing_semantics, query_characters, user_data
        )
        kept_ids = {sem.get("memory_id") for sem in existing_semantics if sem.get("memory_id")}
        if character_filter_info.get("applied"):
            filtered_situations: List[Dict[str, Any]] = []
            for sit in kept_situations:
                sit_sem_ids = []
                for agg in sit.get("aggregates", []) or []:
                    sit_sem_ids.extend(agg.get("semantic_ids", []) or [])
                if any((sid in kept_ids for sid in sit_sem_ids)):
                    filtered_situations.append(sit)
            kept_situations = filtered_situations
        semantic_budget = self.situation_semantic_budget
        existing_ids = {x.get("memory_id") for x in existing_semantics if x.get("memory_id")}
        situation_model: List[Dict[str, Any]] = []
        relationship_triggered = any(
            (self._semantic_source(s) == self.SOURCE_SOCIAL for s in existing_semantics)
        )
        if relationship_triggered:
            social_candidates = user_data.get("social_semantics", []) or []
            social_candidates, social_filter_info = self._filter_semantics_by_query_characters(
                social_candidates, query_characters, user_data
            )
            for sem in social_candidates:
                sid = sem.get("memory_id")
                if sid and sid not in existing_ids:
                    existing_ids.add(sid)
                    situation_model.append(sem)
        else:
            social_filter_info = {
                "enabled": bool(self.restrict_candidates_by_query_characters),
                "applied": False,
                "query_characters": query_characters,
                "before_count": 0,
                "after_count": 0,
            }
        global_excluded_ids = {
            x.get("memory_id")
            for x in (existing_semantics or []) + (situation_model or [])
            if x.get("memory_id")
        }
        global_supplement, global_supplement_info = self._select_global_similarity_semantics(
            query=query,
            user_data=user_data,
            excluded_semantic_ids=global_excluded_ids,
            top_k=self.global_semantic_supplement_top_k,
            query_characters=query_characters,
        )
        for sem in global_supplement:
            sid = sem.get("memory_id")
            if sid and sid not in existing_ids:
                existing_ids.add(sid)
                situation_model.append(sem)
        remaining_budget = semantic_budget - len(existing_semantics) - len(situation_model)
        label_model_filter_info = {
            "require_label_overlap": bool(self.situation_model_require_label_overlap),
            "before_character_filter_count": 0,
            "after_character_filter_count": 0,
        }
        if remaining_budget > 0:
            label_model = self._build_situation_model_with_budget(
                user_data=user_data,
                query=query,
                expected_answer_labels=mental_result.get("expected_answer_labels", []) or [],
                excluded_semantic_ids=existing_ids,
                budget=remaining_budget,
            )
            label_model_filter_info["before_character_filter_count"] = len(label_model)
            label_model, lm_char_filter = self._filter_semantics_by_query_characters(
                label_model, query_characters, user_data
            )
            label_model_filter_info["after_character_filter_count"] = len(label_model)
            label_model_filter_info["character_filter"] = lm_char_filter
            for sem in label_model:
                sid = sem.get("memory_id")
                if sid and sid not in existing_ids:
                    existing_ids.add(sid)
                    situation_model.append(sem)
        remaining_budget = semantic_budget - len(existing_semantics) - len(situation_model)
        keyword_supplement_filter_info = {
            "before_character_filter_count": 0,
            "after_character_filter_count": 0,
        }
        if remaining_budget > 0:
            supplemented = self._supplement_semantics_from_query_keywords(
                query=query,
                query_label_keywords=mental_result.get("query_label_keywords", {}) or {},
                existing_semantic_ids=existing_ids,
                user_data=user_data,
            )
            keyword_supplement_filter_info["before_character_filter_count"] = len(supplemented)
            supplemented, kw_char_filter = self._filter_semantics_by_query_characters(
                supplemented, query_characters, user_data
            )
            keyword_supplement_filter_info["after_character_filter_count"] = len(supplemented)
            keyword_supplement_filter_info["character_filter"] = kw_char_filter
            for sem in supplemented:
                if remaining_budget <= 0:
                    break
                sid = sem.get("memory_id")
                if sid and sid not in existing_ids:
                    existing_ids.add(sid)
                    situation_model.append(sem)
                    remaining_budget -= 1
        return {
            "input_keywords": situation_keywords,
            "keyword_replacement_map": kw_replacement,
            "situations": kept_situations,
            "situation_semantics": existing_semantics,
            "situation_model": dedup_preserve_order(situation_model),
            "relationship_triggered": relationship_triggered,
            "query_characters": query_characters,
            "character_episode_filter": character_filter_info,
            "social_relationship_filter": social_filter_info,
            "global_similarity_supplement": global_supplement_info,
            "situation_model_filter": label_model_filter_info,
            "keyword_supplement_filter": keyword_supplement_filter_info,
        }

    def _is_time_answer_query(
        self, query: str, mental_result: Optional[Dict[str, Any]] = None
    ) -> bool:
        q = normalize_text(query).lower()
        time_patterns = [
            "^when\\b",
            "^on what date\\b",
            "^what date\\b",
            "^which date\\b",
            "^in what year\\b",
            "^which year\\b",
            "^what year\\b",
            "^at what time\\b",
            "^how long\\b",
        ]
        if any((re.search(p, q) for p in time_patterns)):
            return True
        return False

    def _message_for_semantic_without_embedding(
        self, semantic_obj: Dict[str, Any], user_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        message_map = user_data.get("message_map", {})
        mid = semantic_obj.get("matched_message_id")
        if mid and mid in message_map:
            return message_map[mid]
        content = normalize_text(semantic_obj.get("matched_message_content"))
        if content:
            for msg in message_map.values():
                if normalize_text(msg.get("content")) == content:
                    return msg
        ids = semantic_obj.get("source_message_ids") or []
        for mid in ids:
            if mid in message_map:
                return message_map[mid]
        return self._get_message_for_semantic(semantic_obj, user_data)

    def _format_event_block_messages(
        self, block: Dict[str, Any], with_timestamp: bool = True
    ) -> str:
        if not block:
            return "(none)"
        messages = block.get("messages", []) or []
        if messages:
            return format_message_lines(messages, with_timestamp=with_timestamp)
        content = block.get("content", "")
        return content or "(none)"

    def _build_perltqa_evidence_text(
        self,
        supporting_semantics: List[Dict[str, Any]],
        matched_messages: Optional[List[Dict[str, Any]]],
        user_data: Dict[str, Any],
        involves_time: bool,
        query: str = "",
        return_info: bool = False,
    ):
        """Build final-answer evidence as direct semantic-message pairs.

        PerLTQA v6 no longer expands evidence into profile/event source blocks.
        The Situation stage chooses semantic memories; the final-answer stage
        receives exactly those semantics paired with their matched source message.
        This keeps the fourth-round prompt compact and preserves the direct link
        between each semantic and its source context.
        """
        sections: List[str] = ["[Semantic-Message Pair Evidence]"]
        pair_debug: List[Dict[str, Any]] = []
        semantics = supporting_semantics or []
        messages = matched_messages or []
        for idx, sem in enumerate(semantics):
            msg = messages[idx] if idx < len(messages) else None
            if not msg:
                msg = self._message_for_semantic_without_embedding(sem, user_data) or {}
            meta = sem.get("metadata", {}) or {}
            msg_meta = msg.get("metadata", {}) or {}
            source = self._semantic_source(sem)
            source_subtype = normalize_text(
                meta.get("source_subtype") or msg_meta.get("source_subtype") or ""
            )
            character_name = normalize_text(
                meta.get("character_name")
                or msg_meta.get("character_name")
                or msg.get("role")
                or ""
            )
            semantic_ts = sem.get("timestamp") or sem.get("created_at") or ""
            message_ts = msg.get("timestamp") or ""
            event_id = meta.get("event_id") or msg_meta.get("event_id") or ""
            dialogue_id = meta.get("dialogue_id") or msg_meta.get("dialogue_id") or ""
            speaker = msg.get("role") or msg.get("speaker") or msg.get("original_speaker") or ""
            lines: List[str] = [f"Pair {idx}:"]
            lines.append(
                f"  Memory source: {source}" + (f" / {source_subtype}" if source_subtype else "")
            )
            if character_name:
                lines.append(f"  Character: {character_name}")
            if source == self.SOURCE_EVENTS:
                if event_id:
                    lines.append(f"  Event ID: {event_id}")
                if dialogue_id:
                    lines.append(f"  Dialogue ID: {dialogue_id}")
                if semantic_ts:
                    lines.append(f"  Semantic timestamp: {semantic_ts}")
                if message_ts:
                    lines.append(f"  Source-message timestamp: {message_ts}")
                if speaker:
                    lines.append(f"  Source speaker: {speaker}")
            elif message_ts:
                lines.append(f"  Source-message timestamp: {message_ts}")
            lines.append(f"  Semantic: {sem.get('content', '')}")
            if msg:
                msg_content = msg.get("content", "")
                role = msg.get("role") or speaker or ""
                ts_prefix = f"[{message_ts}] " if message_ts else ""
                if role:
                    lines.append(f"  Source message: {ts_prefix}{role}: {msg_content}")
                else:
                    lines.append(f"  Source message: {ts_prefix}{msg_content}")
            elif sem.get("matched_message_content"):
                lines.append(f"  Source message: {sem.get('matched_message_content')}")
            else:
                lines.append("  Source message: (none)")
            sections.append("\n".join(lines))
            pair_debug.append(
                {
                    "index": idx,
                    "memory_id": sem.get("memory_id", ""),
                    "source": source,
                    "source_subtype": source_subtype,
                    "event_id": event_id,
                    "dialogue_id": dialogue_id,
                    "character_name": character_name,
                    "semantic_timestamp": semantic_ts,
                    "message_id": msg.get("message_id", "") if msg else "",
                    "message_timestamp": message_ts,
                    "message_role": speaker,
                }
            )
        evidence_text = "\n\n".join(sections) if semantics else "(none)"
        info: Dict[str, Any] = {
            "mode": "semantic_message_pairs",
            "pair_count": len(semantics),
            "source_block_limit_applied": False,
            "pair_debug": pair_debug,
        }
        if return_info:
            return (evidence_text, info)
        return evidence_text

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

    def _final_time_answer_with_perltqa_evidence(
        self,
        exp_name: str,
        query: str,
        supporting_semantics: List[Dict[str, Any]],
        matched_messages: List[Dict[str, Any]],
        user_data: Dict[str, Any],
    ) -> Tuple[str, Dict[str, int], Dict[str, Any]]:
        evidence_text = self._build_perltqa_evidence_text(
            supporting_semantics, matched_messages, user_data, involves_time=True
        )
        prompt = self.prompts.PERLTQA_TIME_FINAL_ANSWER_PROMPT.render(
            question=query, evidence_text=evidence_text
        )
        return self._run_text_prompt(
            exp_name, prompt, max_tokens=2048, stage_name="situation_time_final_answer"
        )

    def _final_factual_answer_with_perltqa_evidence(
        self,
        exp_name: str,
        query: str,
        supporting_semantics: List[Dict[str, Any]],
        matched_messages: List[Dict[str, Any]],
        user_data: Dict[str, Any],
    ) -> Tuple[str, Dict[str, int], Dict[str, Any]]:
        evidence_text = self._build_perltqa_evidence_text(
            supporting_semantics, matched_messages, user_data, involves_time=False
        )
        prompt = self.prompts.PERLTQA_FACTUAL_FINAL_ANSWER_PROMPT.render(
            question=query, evidence_text=evidence_text
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

    def _dedup_semantic_message_pairs_by_message(
        self, query: str, supporting_semantics: List[Dict[str, Any]], user_data: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        kept_semantics = dedup_preserve_order(supporting_semantics)
        kept_messages: List[Dict[str, Any]] = []
        for sem in kept_semantics:
            kept_messages.append(self._message_for_semantic_without_embedding(sem, user_data) or {})
        return (kept_semantics, kept_messages)

    def _select_supporting_semantics_after_decision(
        self,
        decision_resp: Dict[str, Any],
        user_data: Dict[str, Any],
        candidate_support_pool: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []

        def add_by_indices(values: List[Any]) -> None:
            for raw in values or []:
                try:
                    idx = int(raw)
                except Exception:
                    continue
                if 0 <= idx < len(candidate_support_pool):
                    selected.append(candidate_support_pool[idx])

        add_by_indices(decision_resp.get("supporting_semantic_indices", []) or [])
        add_by_indices(decision_resp.get("need_expand_semantic_indices", []) or [])
        supporting_semantic_contents = decision_resp.get("supporting_semantic_contents", []) or []
        need_expand_semantic_contents = decision_resp.get("need_expand_semantic_contents", []) or []
        merged_semantic_contents = dedup_preserve_order(
            supporting_semantic_contents + need_expand_semantic_contents
        )
        selected.extend(
            self._map_semantic_contents_to_real_semantics(
                merged_semantic_contents, user_data, candidate_semantics=candidate_support_pool
            )
        )
        selected = dedup_preserve_order(selected)
        return selected or candidate_support_pool

    def _apply_final_evidence_scope(
        self,
        supporting_semantics: List[Dict[str, Any]],
        matched_messages: List[Dict[str, Any]],
        decision_resp: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        query_memory_category = self._normalize_query_memory_category(
            decision_resp.get("query_memory_category", "events")
        )
        info = {
            "final_evidence_scope": self.final_evidence_scope,
            "query_memory_category": query_memory_category,
            "before_filter_count": len(supporting_semantics or []),
            "after_filter_count": len(supporting_semantics or []),
        }
        if self.final_evidence_scope != "predicted_category" or query_memory_category == "unknown":
            return (supporting_semantics, matched_messages, info)
        filtered_semantics: List[Dict[str, Any]] = []
        filtered_messages: List[Dict[str, Any]] = []
        for sem, msg in zip(supporting_semantics or [], matched_messages or []):
            if self._semantic_matches_query_category(sem, query_memory_category):
                filtered_semantics.append(sem)
                filtered_messages.append(msg)
        info["after_filter_count"] = len(filtered_semantics)
        return (filtered_semantics, filtered_messages, info)

    def _parse_perltqa_answer_json_or_repair(
        self, raw_text: str
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Parse final-answer JSON, or salvage the answer field from malformed JSON."""
        raw_text = (raw_text or "").strip()
        info: Dict[str, Any] = {"success": False, "method": "none"}
        if not raw_text:
            info["error"] = "empty_response"
            return (None, info)
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict) and normalize_text(parsed.get("answer", "")):
                info.update({"success": True, "method": "json", "parsed_json": parsed})
                return (str(parsed.get("answer", "")).strip(), info)
            info.update({"method": "json_no_answer", "parsed_json": parsed})
        except Exception as exc:
            info["json_error"] = repr(exc)
        first = raw_text.find("{")
        last = raw_text.rfind("}")
        if first >= 0 and last > first:
            candidate = raw_text[first : last + 1]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and normalize_text(parsed.get("answer", "")):
                    info.update(
                        {"success": True, "method": "json_substring", "parsed_json": parsed}
                    )
                    return (str(parsed.get("answer", "")).strip(), info)
            except Exception as exc:
                info["json_substring_error"] = repr(exc)
        if raw_text.lstrip().startswith("{") and (not raw_text.rstrip().endswith("}")):
            repaired = raw_text.rstrip() + "}"
            try:
                parsed = json.loads(repaired)
                if isinstance(parsed, dict) and normalize_text(parsed.get("answer", "")):
                    info.update(
                        {"success": True, "method": "append_missing_brace", "parsed_json": parsed}
                    )
                    return (str(parsed.get("answer", "")).strip(), info)
            except Exception as exc:
                info["append_brace_error"] = repr(exc)
        m = re.search('"answer"\\s*:\\s*"((?:\\\\.|[^"\\\\])*)"', raw_text, flags=re.DOTALL)
        if m:
            answer_raw = m.group(1)
            try:
                answer = json.loads('"' + answer_raw + '"')
            except Exception:
                answer = answer_raw.replace('\\"', '"').replace(
                    "\\n",
                    "\n",
                )
            answer = normalize_text(answer)
            if answer:
                info.update({"success": True, "method": "regex_answer_quoted"})
                return (answer, info)
        m = re.search('"answer"\\s*:\\s*(.+)$', raw_text, flags=re.DOTALL)
        if m:
            answer = m.group(1).strip().rstrip("}\n \t")
            if answer.startswith('"'):
                answer = answer[1:]
            if answer.endswith('"'):
                answer = answer[:-1]
            answer = answer.strip()
            answer = re.sub(
                ',\\s*"[A-Za-z_][A-Za-z0-9_]*"\\s*:\\s*.*$', "", answer, flags=re.DOTALL
            ).strip()
            answer = normalize_text(answer)
            if answer:
                info.update({"success": True, "method": "regex_answer_tail"})
                return (answer, info)
        return (None, info)

    @staticmethod
    def _merge_token_usage_dicts(usages: List[Dict[str, int]]) -> Dict[str, int]:
        merged = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        for usage in usages or []:
            for key in list(merged.keys()):
                try:
                    merged[key] += int(usage.get(key, 0) or 0)
                except Exception:
                    pass
            for opt_key in ["thinking_tokens", "visible_output_tokens"]:
                if opt_key in usage:
                    merged.setdefault(opt_key, 0)
                    try:
                        merged[opt_key] += int(usage.get(opt_key, 0) or 0)
                    except Exception:
                        pass
        return merged

    def _final_answer_with_perltqa_evidence(
        self,
        exp_name: str,
        query: str,
        supporting_semantics: List[Dict[str, Any]],
        matched_messages: List[Dict[str, Any]],
        user_data: Dict[str, Any],
        query_memory_category: str,
    ) -> Tuple[str, Dict[str, int], Dict[str, Any]]:
        evidence_text, pair_evidence_info = self._build_perltqa_evidence_text(
            supporting_semantics,
            matched_messages,
            user_data,
            involves_time=True,
            query=query,
            return_info=True,
        )
        prompt = self.prompts.PERLTQA_UNIFIED_FINAL_ANSWER_PROMPT.render(
            question=query,
            query_memory_category=self._normalize_query_memory_category(query_memory_category),
            evidence_text=evidence_text,
        )
        attempts: List[Dict[str, Any]] = []
        token_usages: List[Dict[str, int]] = []
        last_raw_text = ""
        for attempt_idx in range(self.final_answer_parse_max_retries):
            raw_text, token_usage, trace = self._run_text_prompt(
                exp_name, prompt, max_tokens=4096, stage_name="situation_final_answer"
            )
            token_usages.append(token_usage)
            last_raw_text = raw_text
            answer, parse_info = self._parse_perltqa_answer_json_or_repair(raw_text)
            attempts.append(
                {
                    "attempt": attempt_idx + 1,
                    "raw_response": raw_text,
                    "token_usage": token_usage,
                    "parse_info": parse_info,
                    "llm_trace": trace,
                }
            )
            if answer:
                merged_usage = self._merge_token_usage_dicts(token_usages)
                final_trace = {
                    "stage": "situation_final_answer",
                    "prompt": prompt,
                    "response": raw_text,
                    "answer": answer,
                    "parse_success": True,
                    "parse_method": parse_info.get("method", "unknown"),
                    "attempts": attempts,
                    "num_attempts": len(attempts),
                    "token_usage": merged_usage,
                    "pair_evidence_info": pair_evidence_info,
                    "final_source_info_limit": {
                        "enabled": False,
                        "reason": "PerLTQA v6 uses direct semantic-message pairs",
                    },
                }
                if "parsed_json" in parse_info:
                    final_trace["parsed_json"] = parse_info["parsed_json"]
                return (answer, merged_usage, final_trace)
        merged_usage = self._merge_token_usage_dicts(token_usages)
        final_trace = {
            "stage": "situation_final_answer",
            "prompt": prompt,
            "response": last_raw_text,
            "answer": last_raw_text,
            "parse_success": False,
            "parse_method": "raw_fallback_after_retries",
            "attempts": attempts,
            "num_attempts": len(attempts),
            "token_usage": merged_usage,
            "pair_evidence_info": pair_evidence_info,
            "final_source_info_limit": {
                "enabled": False,
                "reason": "PerLTQA v6 uses direct semantic-message pairs",
            },
        }
        return (last_raw_text, merged_usage, final_trace)

    def _run_situation_full(
        self,
        query: str,
        user_data: Dict[str, Any],
        mental_result: Dict[str, Any],
        exp_name: str = "full",
    ) -> Dict[str, Any]:
        context = self._build_full_situation_context_from_keywords(query, user_data, mental_result)
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
            involves_time=True,
        )
        candidate_support_pool = dedup_preserve_order(situation_semantics + situation_model)
        supporting_semantics = self._select_supporting_semantics_after_decision(
            decision_resp, user_data, candidate_support_pool
        )
        supporting_semantics, matched_messages = self._dedup_semantic_message_pairs_by_message(
            query, supporting_semantics, user_data
        )
        supporting_semantics, matched_messages, filter_info = self._apply_final_evidence_scope(
            supporting_semantics, matched_messages, decision_resp
        )
        final_answer, tok2, trace2 = self._final_answer_with_perltqa_evidence(
            exp_name,
            query,
            supporting_semantics,
            matched_messages,
            user_data,
            query_memory_category=decision_resp.get("query_memory_category", "events"),
        )
        trace1["stage1_semantic_budget_limit"] = stage1_limit_info
        trace2["final_evidence_filter"] = filter_info
        return {
            "keyword_input": context["input_keywords"],
            "keyword_replacement_map": context["keyword_replacement_map"],
            "situations": context["situations"],
            "situation_semantics": situation_semantics,
            "situation_model": situation_model,
            "relationship_triggered": context.get("relationship_triggered", False),
            "query_characters": context.get("query_characters", []),
            "character_episode_filter": context.get("character_episode_filter", {}),
            "social_relationship_filter": context.get("social_relationship_filter", {}),
            "situation_model_filter": context.get("situation_model_filter", {}),
            "keyword_supplement_filter": context.get("keyword_supplement_filter", {}),
            "stage1_semantic_budget_limit": stage1_limit_info,
            "query_memory_category": decision_resp.get("query_memory_category", "events"),
            "final_evidence_scope": self.final_evidence_scope,
            "final_evidence_filter": filter_info,
            "agent_response": {
                "query_memory_category": decision_resp.get("query_memory_category", "events"),
                "answer_determined": bool(decision_resp.get("answer_determined", False)),
                "answer": final_answer,
                "proposed_answer": decision_resp.get("proposed_answer", ""),
                "supporting_semantic_indices": decision_resp.get("supporting_semantic_indices", [])
                or [],
                "supporting_semantic_contents": decision_resp.get(
                    "supporting_semantic_contents", []
                )
                or [],
                "need_expand_semantic_indices": decision_resp.get(
                    "need_expand_semantic_indices", []
                )
                or [],
                "need_expand_semantic_contents": decision_resp.get(
                    "need_expand_semantic_contents", []
                )
                or [],
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
                    "query_memory_category": "unknown",
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
        clustered_semantics, _, stage1_limit_info = self._limit_situation_stage1_semantics(
            query=query, situation_semantics=clustered_semantics, situation_model=[]
        )
        decision_resp, tok1, trace1 = self._decision_with_situation(
            exp_name=exp_name,
            query=query,
            situation_evidence=clustered_semantics,
            situation_model_evidence=[],
            involves_time=True,
        )
        supporting_semantics = self._select_supporting_semantics_after_decision(
            decision_resp, user_data, clustered_semantics
        )
        supporting_semantics, matched_messages = self._dedup_semantic_message_pairs_by_message(
            query, supporting_semantics, user_data
        )
        supporting_semantics, matched_messages, filter_info = self._apply_final_evidence_scope(
            supporting_semantics, matched_messages, decision_resp
        )
        final_answer, tok2, trace2 = self._final_answer_with_perltqa_evidence(
            exp_name,
            query,
            supporting_semantics,
            matched_messages,
            user_data,
            query_memory_category=decision_resp.get("query_memory_category", "events"),
        )
        trace1["stage1_semantic_budget_limit"] = stage1_limit_info
        trace2["final_evidence_filter"] = filter_info
        return {
            "clustered_semantics": clustered_semantics,
            "query_memory_category": decision_resp.get("query_memory_category", "events"),
            "final_evidence_scope": self.final_evidence_scope,
            "final_evidence_filter": filter_info,
            "stage1_semantic_budget_limit": stage1_limit_info,
            "agent_response": {
                "query_memory_category": decision_resp.get("query_memory_category", "events"),
                "answer_determined": bool(decision_resp.get("answer_determined", False)),
                "answer": final_answer,
                "proposed_answer": decision_resp.get("proposed_answer", ""),
                "supporting_semantic_indices": decision_resp.get("supporting_semantic_indices", [])
                or [],
                "supporting_semantic_contents": decision_resp.get(
                    "supporting_semantic_contents", []
                )
                or [],
                "need_expand_semantic_indices": decision_resp.get(
                    "need_expand_semantic_indices", []
                )
                or [],
                "need_expand_semantic_contents": decision_resp.get(
                    "need_expand_semantic_contents", []
                )
                or [],
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
                for sem in sems:
                    content = sem.get("content", "")
                    sim = self._topk_by_similarity(query, [content], 1)[0][1] if content else 0.0
                    scored.append((sem, sim))
                scored.sort(key=lambda x: x[1], reverse=True)
                selected_semantics.extend([x[0] for x in scored[:semantic_budget]])
        selected_semantics = dedup_preserve_order(selected_semantics)
        selected_semantics, _, stage1_limit_info = self._limit_situation_stage1_semantics(
            query=query, situation_semantics=selected_semantics, situation_model=[]
        )
        decision_resp, tok1, trace1 = self._decision_with_situation(
            exp_name=exp_name,
            query=query,
            situation_evidence=selected_semantics,
            situation_model_evidence=[],
            involves_time=True,
        )
        supporting_semantics = self._select_supporting_semantics_after_decision(
            decision_resp, user_data, selected_semantics
        )
        supporting_semantics, matched_messages = self._dedup_semantic_message_pairs_by_message(
            query, supporting_semantics, user_data
        )
        supporting_semantics, matched_messages, filter_info = self._apply_final_evidence_scope(
            supporting_semantics, matched_messages, decision_resp
        )
        final_answer, tok2, trace2 = self._final_answer_with_perltqa_evidence(
            exp_name,
            query,
            supporting_semantics,
            matched_messages,
            user_data,
            query_memory_category=decision_resp.get("query_memory_category", "events"),
        )
        trace1["stage1_semantic_budget_limit"] = stage1_limit_info
        trace2["final_evidence_filter"] = filter_info
        return {
            "selected_semantics": selected_semantics,
            "query_memory_category": decision_resp.get("query_memory_category", "events"),
            "final_evidence_scope": self.final_evidence_scope,
            "final_evidence_filter": filter_info,
            "stage1_semantic_budget_limit": stage1_limit_info,
            "agent_response": {
                "query_memory_category": decision_resp.get("query_memory_category", "events"),
                "answer_determined": bool(decision_resp.get("answer_determined", False)),
                "answer": final_answer,
                "proposed_answer": decision_resp.get("proposed_answer", ""),
                "supporting_semantic_indices": decision_resp.get("supporting_semantic_indices", [])
                or [],
                "supporting_semantic_contents": decision_resp.get(
                    "supporting_semantic_contents", []
                )
                or [],
                "need_expand_semantic_indices": decision_resp.get(
                    "need_expand_semantic_indices", []
                )
                or [],
                "need_expand_semantic_contents": decision_resp.get(
                    "need_expand_semantic_contents", []
                )
                or [],
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
