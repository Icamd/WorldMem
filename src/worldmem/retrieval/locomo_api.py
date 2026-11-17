from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from worldmem.clients import EmbeddingClient, LLMClient
from worldmem.config import MemoryConfig

logger = logging.getLogger(__name__)
from worldmem.prompts import locomo_api as prompt_profile
from worldmem.retrieval.locomo_qwen import Searcher as SharedSearcher
from worldmem.retrieval.locomo_qwen import (
    dedup_preserve_order,
    format_semantic_lines,
    normalize_text,
)


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
        msg_meta = msg.get("metadata") if isinstance(msg.get("metadata"), dict) else {}
        expanded_source_context = bool(msg_meta.get("expanded_source_context"))
        lines.append(f"Pair {idx}:")
        if with_timestamp and sem_ts:
            lines.append(f"  Semantic: [{sem_ts}] {sem_content}")
        else:
            lines.append(f"  Semantic: {sem_content}")
        if msg:
            if expanded_source_context:
                if with_timestamp and msg_ts:
                    lines.append(f"  Matched message: [{msg_ts}] {msg_content}")
                else:
                    lines.append(f"  Matched message: {msg_content}")
            elif with_timestamp and msg_ts:
                lines.append(f"  Matched message: [{msg_ts}] {msg_role}: {msg_content}")
            else:
                lines.append(f"  Matched message: {msg_role}: {msg_content}")
        else:
            lines.append("  Matched message: (none)")
    return "\n".join(lines) if lines else "(none)"


class Searcher(SharedSearcher):
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
        situation_min_selected_semantics: int = 5,
        character_keyword_situation_max_semantics: int = 3,
        stage_thinking: Optional[Dict[str, bool]] = None,
        stage_reasoning_effort: Optional[Dict[str, Optional[str]]] = None,
        openai_api_stage_max_tokens: int = 16384,
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
        self.situation_min_selected_semantics = max(0, int(situation_min_selected_semantics or 0))
        self.character_keyword_situation_max_semantics = max(
            0, int(character_keyword_situation_max_semantics or 0)
        )
        self.stage_thinking = stage_thinking or {}
        self.stage_reasoning_effort = {
            str(k): None if v is None else str(v).strip()
            for k, v in (stage_reasoning_effort or {}).items()
            if v is not None and str(v).strip()
        }
        self.openai_api_stage_max_tokens = max(0, int(openai_api_stage_max_tokens or 0))
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
            "full": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "num_queries": 0,
                "reasoning_tokens": 0,
                "thinking_tokens": 0,
                "visible_output_tokens": 0,
            },
            "no_mental": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "num_queries": 0,
                "reasoning_tokens": 0,
                "thinking_tokens": 0,
                "visible_output_tokens": 0,
            },
            "no_situation": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "num_queries": 0,
                "reasoning_tokens": 0,
                "thinking_tokens": 0,
                "visible_output_tokens": 0,
            },
        }
        self._inference_re = re.compile("|".join(self.prompts.INFERENCE_PATTERNS), re.IGNORECASE)

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
            reasoning_tokens = int(
                llm_usage.get("reasoning_tokens_est", llm_usage.get("thinking_tokens_est", 0)) or 0
            )
            visible_output_tokens = int(
                llm_usage.get(
                    "visible_completion_tokens_est", max(0, output_tokens - reasoning_tokens)
                )
                or 0
            )
        else:
            input_tokens = self._count_tokens(input_text)
            output_tokens = self._count_tokens(output_text)
            total = input_tokens + output_tokens
            reasoning_tokens = 0
            visible_output_tokens = output_tokens
        with self._token_lock:
            self._token_stats[exp_name].setdefault("reasoning_tokens", 0)
            self._token_stats[exp_name].setdefault("thinking_tokens", 0)
            self._token_stats[exp_name].setdefault("visible_output_tokens", 0)
            self._token_stats[exp_name]["input_tokens"] += input_tokens
            self._token_stats[exp_name]["output_tokens"] += output_tokens
            self._token_stats[exp_name]["total_tokens"] += total
            self._token_stats[exp_name]["reasoning_tokens"] += reasoning_tokens
            self._token_stats[exp_name]["thinking_tokens"] += reasoning_tokens
            self._token_stats[exp_name]["visible_output_tokens"] += visible_output_tokens
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total,
            "reasoning_tokens": reasoning_tokens,
            "thinking_tokens": reasoning_tokens,
            "visible_output_tokens": visible_output_tokens,
        }

    def _llm_backend_name(self) -> str:
        return str(getattr(self.llm_client, "backend", "") or "").lower().replace("-", "_")

    def _is_openai_api_backend(self) -> bool:
        return self._llm_backend_name() in {"openai_api", "openai"}

    def _llm_call_config_for_stage(self, stage_name: str) -> Dict[str, Any]:
        enable_thinking = bool(self.stage_thinking.get(stage_name, False))
        if enable_thinking:
            cfg = dict(self.thinking_generation_config)
        else:
            cfg = dict(self.non_thinking_generation_config)
        cfg["enable_thinking"] = enable_thinking
        cfg["api_reasoning_effort"] = None
        if self._is_openai_api_backend():
            effort = self.stage_reasoning_effort.get(stage_name)
            if effort:
                cfg["api_reasoning_effort"] = effort
        return cfg

    def _effective_stage_max_tokens(self, stage_name: str, requested_max_tokens: int) -> int:
        requested = int(requested_max_tokens or 0)
        if self._is_openai_api_backend() and self.openai_api_stage_max_tokens > 0:
            return max(requested, self.openai_api_stage_max_tokens)
        return requested

    def _openai_api_extra_kwargs_for_stage(self, call_cfg: Dict[str, Any]) -> Dict[str, Any]:
        if not self._is_openai_api_backend():
            return {}
        extra: Dict[str, Any] = {}
        effort = call_cfg.get("api_reasoning_effort")
        if effort:
            extra["api_reasoning_effort"] = effort
        return extra

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
        return self._effective_stage_max_tokens(stage_name, int(stage_limits.get(stage_name, 4096)))

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
                    **self._openai_api_extra_kwargs_for_stage(call_cfg),
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
                    **self._openai_api_extra_kwargs_for_stage(call_cfg),
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
            "api_reasoning_effort": call_cfg.get("api_reasoning_effort"),
            "generation_config": {
                "temperature": call_cfg["temperature"],
                "top_p": call_cfg["top_p"],
                "top_k": call_cfg["top_k"],
                "min_p": call_cfg["min_p"],
                "max_tokens": max_tokens,
                "json_system_prompt": system_prompt,
                "api_reasoning_effort": call_cfg.get("api_reasoning_effort"),
            },
        }
        return (out, aggregate_token_usage, trace)

    def _run_text_prompt(
        self, exp_name: str, prompt_text: str, max_tokens: int, stage_name: str
    ) -> Tuple[str, Dict[str, int], Dict[str, Any]]:
        call_cfg = self._llm_call_config_for_stage(stage_name)
        max_tokens = self._effective_stage_max_tokens(stage_name, max_tokens)
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
                **self._openai_api_extra_kwargs_for_stage(call_cfg),
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
                **self._openai_api_extra_kwargs_for_stage(call_cfg),
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
            "api_reasoning_effort": call_cfg.get("api_reasoning_effort"),
            "generation_config": {
                "temperature": call_cfg["temperature"],
                "top_p": call_cfg["top_p"],
                "top_k": call_cfg["top_k"],
                "min_p": call_cfg["min_p"],
                "max_tokens": max_tokens,
                "api_reasoning_effort": call_cfg.get("api_reasoning_effort"),
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
        semantic_budget = int(self.situation_semantic_budget or 0)
        existing_ids = {x.get("memory_id") for x in existing_semantics if x.get("memory_id")}
        character_supplement, character_supplement_info = (
            self._select_character_keyword_situation_semantics(
                query=query,
                user_data=user_data,
                selected_keywords=selected_keywords,
                existing_semantic_ids=existing_ids,
            )
        )
        if character_supplement:
            existing_semantics = dedup_preserve_order(existing_semantics + character_supplement)
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
            "character_keyword_supplement": character_supplement_info,
            "stage1_semantic_prelimit_count": prelimit_count,
        }

    def _time_keywords_have_explicit_value(self, mental_result: Dict[str, Any]) -> bool:
        """Third-round timestamp control: Time=[null] should not add timestamps."""
        nullish = {"", "null", "none", "n/a", "na", "unknown"}
        for key in ("query_label_keywords", "selected_keywords"):
            mapping = mental_result.get(key, {}) or {}
            if not isinstance(mapping, dict):
                continue
            values = []
            for label, kws in mapping.items():
                if normalize_text(label).lower() == "time":
                    values.extend(kws or [])
            for value in values:
                if value is None:
                    continue
                cleaned = normalize_text(value).strip().lower()
                if cleaned and cleaned not in nullish:
                    return True
        return False

    def _should_include_timestamp_in_situation_selection(
        self, mental_result: Dict[str, Any]
    ) -> bool:
        return self._time_keywords_have_explicit_value(mental_result)

    def _strip_agent_semantic_item_text(self, item: Any) -> str:
        raw = normalize_text(item)
        if not raw:
            return ""
        raw = raw.strip().strip('"').strip("'")
        raw = re.sub("^\\s*(?:[-*]\\s*)?\\d+\\s*[\\).:\\-]\\s*", "", raw)
        raw = re.sub("^\\s*\\[[^\\]]+\\]\\s*", "", raw)
        raw = re.sub("^\\s*Semantic\\s*:\\s*", "", raw, flags=re.IGNORECASE)
        return normalize_text(raw)

    def _best_semantic_by_similarity(
        self, query_text: str, candidate_semantics: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        query_text = normalize_text(query_text)
        if not query_text or not candidate_semantics:
            return None
        q_emb = self._embed_text(query_text)
        best_sem = None
        best_score = -1.0
        for sem in candidate_semantics:
            content = normalize_text(sem.get("content", ""))
            if not content:
                continue
            try:
                score = self._cosine_sim(q_emb, self._embed_text(content))
            except Exception:
                score = 0.0
            if score > best_score:
                best_score = score
                best_sem = sem
        return best_sem

    def _ensure_min_selected_semantics(
        self,
        query: str,
        selected_semantics: List[Dict[str, Any]],
        candidate_semantics: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        min_count = int(self.situation_min_selected_semantics or 0)
        selected = dedup_preserve_order(selected_semantics or [])
        info: Dict[str, Any] = {
            "enabled": min_count > 0,
            "min_count": min_count,
            "before_count": len(selected),
            "after_count": len(selected),
            "added_count": 0,
            "added_memory_ids": [],
        }
        if min_count <= 0 or len(selected) >= min_count:
            return (selected, info)
        selected_keys = {self._semantic_dedup_key(sem) for sem in selected}
        scored = []
        for sem in candidate_semantics or []:
            key = self._semantic_dedup_key(sem)
            if key in selected_keys:
                continue
            scored.append((sem, self._score_semantic_for_query(query, sem)))
        scored.sort(key=lambda x: x[1], reverse=True)
        added = []
        for sem, _score in scored:
            if len(selected) >= min_count:
                break
            key = self._semantic_dedup_key(sem)
            if key in selected_keys:
                continue
            selected.append(sem)
            selected_keys.add(key)
            added.append(sem)
        info.update(
            {
                "after_count": len(selected),
                "added_count": len(added),
                "added_memory_ids": [x.get("memory_id", "") for x in added],
            }
        )
        return (selected, info)

    def _select_character_keyword_situation_semantics(
        self,
        query: str,
        user_data: Dict[str, Any],
        selected_keywords: Dict[str, List[Any]],
        existing_semantic_ids: Set[str],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        threshold = int(self.character_keyword_situation_max_semantics or 0)
        info: Dict[str, Any] = {
            "enabled": threshold > 0,
            "threshold_exclusive": threshold,
            "keyword_records": [],
            "selected_count": 0,
        }
        if threshold <= 0:
            return ([], info)
        char_keywords = []
        for label, kws in (selected_keywords or {}).items():
            if normalize_text(label).lower() != "character":
                continue
            for kw in kws or []:
                kw = normalize_text(kw)
                if kw:
                    char_keywords.append(kw)
        char_keywords = dedup_preserve_order(char_keywords)
        selected: List[Dict[str, Any]] = []
        for kw in char_keywords:
            candidate_situations, replacement = self._find_situations_for_keywords([kw], user_data)
            kept_situations, sems = self._collect_unique_semantics_from_situations(
                candidate_situations, user_data
            )
            sems = dedup_preserve_order(sems)
            record = {
                "keyword": kw,
                "replacement_map": replacement,
                "matched_situation_count": len(kept_situations),
                "matched_semantic_count": len(sems),
                "accepted": False,
                "added_count": 0,
                "reason": "",
            }
            if 0 < len(sems) < threshold:
                added = []
                for sem in sems:
                    sid = sem.get("memory_id")
                    if sid and sid in existing_semantic_ids:
                        continue
                    if sid:
                        existing_semantic_ids.add(sid)
                    selected.append(sem)
                    added.append(sem)
                record.update(
                    {
                        "accepted": True,
                        "added_count": len(added),
                        "reason": "character_keyword_low_frequency_situation",
                        "added_memory_ids": [x.get("memory_id", "") for x in added],
                    }
                )
            elif len(sems) >= threshold:
                record["reason"] = "matched_semantic_count_not_below_threshold"
            else:
                record["reason"] = "no_matching_situation_semantics"
            info["keyword_records"].append(record)
        info["selected_count"] = len(selected)
        return (selected, info)

    def _question_mark_terminal(self, content: str) -> bool:
        content = normalize_text(content).rstrip()
        content = content.rstrip(" \"'”’)]}）")
        return content.endswith("?") or content.endswith("？")

    def _ordered_messages_for_context(self, user_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        ordered = []
        seen = set()

        def add_msg(msg: Dict[str, Any]) -> None:
            if not isinstance(msg, dict):
                return
            key = msg.get("message_id") or (
                normalize_text(msg.get("timestamp", "")),
                normalize_text(msg.get("role", "")),
                normalize_text(msg.get("content", "")),
            )
            if key in seen:
                return
            seen.add(key)
            ordered.append(msg)

        for ep in user_data.get("episodes", []) or []:
            for msg in ep.get("original_messages", []) or []:
                add_msg(msg)
        for block in user_data.get("message_blocks", []) or []:
            for msg in block.get("messages", []) or []:
                add_msg(msg)
        return ordered

    def _format_role_content_for_context(self, msg: Dict[str, Any]) -> str:
        role = normalize_text(msg.get("role", ""))
        content = normalize_text(msg.get("content", ""))
        return f"{role}: {content}" if role else content

    def _augment_matched_message_with_adjacent_question_context(
        self, msg: Dict[str, Any], user_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not msg:
            return msg
        ordered = self._ordered_messages_for_context(user_data)
        if not ordered:
            return msg
        target_idx = None
        mid = msg.get("message_id")
        if mid:
            for idx, candidate in enumerate(ordered):
                if candidate.get("message_id") == mid:
                    target_idx = idx
                    break
        if target_idx is None:
            target_content = normalize_text(msg.get("content", ""))
            target_ts = normalize_text(msg.get("timestamp", ""))
            for idx, candidate in enumerate(ordered):
                if normalize_text(candidate.get("content", "")) == target_content:
                    if not target_ts or normalize_text(candidate.get("timestamp", "")) == target_ts:
                        target_idx = idx
                        break
        if target_idx is None:
            return msg
        pieces: List[Dict[str, Any]] = []
        current = ordered[target_idx]
        if target_idx > 0:
            prev_msg = ordered[target_idx - 1]
            if self._question_mark_terminal(prev_msg.get("content", "")):
                pieces.append(prev_msg)
        pieces.append(current)
        if self._question_mark_terminal(current.get("content", "")) and target_idx + 1 < len(
            ordered
        ):
            pieces.append(ordered[target_idx + 1])
        if len(pieces) <= 1:
            return msg
        out = dict(msg)
        metadata = dict(out.get("metadata") or {}) if isinstance(out.get("metadata"), dict) else {}
        metadata.update(
            {
                "expanded_source_context": True,
                "expanded_source_message_ids": [x.get("message_id", "") for x in pieces],
                "expanded_source_rule": "prepend_previous_question_or_append_next_answer",
            }
        )
        out["metadata"] = metadata
        out["content"] = " ".join((self._format_role_content_for_context(x) for x in pieces if x))
        out["role"] = ""
        if pieces[0].get("timestamp"):
            out["timestamp"] = pieces[0].get("timestamp")
        return out

    def _decision_with_situation(
        self,
        exp_name: str,
        query: str,
        situation_evidence: List[Dict[str, Any]],
        situation_model_evidence: List[Dict[str, Any]],
        involves_time: bool,
    ) -> Tuple[Dict[str, Any], Dict[str, int], Dict[str, Any]]:
        candidate_semantics = dedup_preserve_order(
            (situation_evidence or []) + (situation_model_evidence or [])
        )
        prompt = self.prompts.SITUATION_DECISION_PROMPT.render(
            question=query,
            min_semantics=self.situation_min_selected_semantics,
            semantic_evidence=format_semantic_lines(
                candidate_semantics, with_timestamp=involves_time
            ),
        )
        response, token_usage, trace = self._run_json_prompt(
            exp_name,
            prompt,
            default_response={"supporting_semantic_contents": []},
            stage_name="situation_decision",
        )
        if isinstance(trace, dict):
            trace["stage1_unified_semantic_evidence_count"] = len(candidate_semantics)
            trace["stage1_unified_semantic_evidence"] = candidate_semantics
            trace["min_semantics_requested"] = self.situation_min_selected_semantics
            trace["with_timestamp"] = bool(involves_time)
        return (response, token_usage, trace)

    def _build_pair_prompt_text(
        self,
        supporting_semantics: List[Dict[str, Any]],
        matched_messages: List[Dict[str, Any]],
        involves_time: bool,
        user_data: Dict[str, Any],
    ) -> str:
        pairs = []
        for idx, sem in enumerate(supporting_semantics):
            msg = matched_messages[idx] if idx < len(matched_messages) else {}
            msg = self._augment_matched_message_with_adjacent_question_context(msg, user_data)
            pairs.append({"semantic": sem, "message": msg})
        return format_semantic_message_pairs(pairs, with_timestamp=involves_time)

    def _final_time_answer_with_situation(
        self,
        exp_name: str,
        query: str,
        supporting_semantics: List[Dict[str, Any]],
        matched_messages: List[Dict[str, Any]],
        involves_time: bool,
        user_data: Dict[str, Any],
    ) -> Tuple[str, Dict[str, int], Dict[str, Any]]:
        pair_text = self._build_pair_prompt_text(
            supporting_semantics, matched_messages, involves_time, user_data
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
        user_data: Dict[str, Any],
    ) -> Tuple[str, Dict[str, int], Dict[str, Any]]:
        pair_text = self._build_pair_prompt_text(
            supporting_semantics, matched_messages, involves_time, user_data
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
        user_data: Dict[str, Any],
    ) -> Tuple[str, Dict[str, int], Dict[str, Any]]:
        pair_text = self._build_pair_prompt_text(
            supporting_semantics, matched_messages, involves_time, user_data
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
        candidate_pool = dedup_preserve_order(candidate_semantics or [])
        search_pool = candidate_pool if candidate_pool else semantics
        resolved = []

        def find_exact_or_contained(
            clean_text: str, pool: List[Dict[str, Any]]
        ) -> Optional[Dict[str, Any]]:
            target = normalize_text(clean_text)
            if not target:
                return None
            for sem in pool:
                sem_text = normalize_text(sem.get("content", ""))
                if sem_text == target:
                    return sem
            for sem in pool:
                sem_text = normalize_text(sem.get("content", ""))
                if sem_text and (sem_text in target or target in sem_text):
                    return sem
            return None

        for item in content_list:
            if isinstance(item, int):
                if candidate_pool and 0 <= item < len(candidate_pool):
                    resolved.append(candidate_pool[item])
                    continue
                else:
                    continue
            if isinstance(item, str):
                raw = item.strip()
                if raw.isdigit():
                    idx = int(raw)
                    if candidate_pool and 0 <= idx < len(candidate_pool):
                        resolved.append(candidate_pool[idx])
                        continue
            content = self._strip_agent_semantic_item_text(item)
            if not content:
                continue
            exact = find_exact_or_contained(content, search_pool)
            if exact is not None:
                resolved.append(exact)
                continue
            best = self._best_semantic_by_similarity(content, search_pool)
            if best is not None:
                resolved.append(best)
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
        stage1_with_timestamp = self._should_include_timestamp_in_situation_selection(mental_result)
        decision_resp, tok1, trace1 = self._decision_with_situation(
            exp_name=exp_name,
            query=query,
            situation_evidence=situation_semantics,
            situation_model_evidence=situation_model,
            involves_time=stage1_with_timestamp,
        )
        trace1["third_round_timestamp_policy"] = {
            "with_timestamp": stage1_with_timestamp,
            "reason": "explicit_non_null_time_keyword_required",
            "original_involves_time": mental_result.get("involves_time", False),
        }
        trace1["global_similarity_supplement"] = context.get("global_similarity_supplement", {})
        trace1["situation_model_filter"] = context.get("situation_model_filter", {})
        trace1["keyword_supplement"] = context.get("keyword_supplement", {})
        trace1["character_keyword_supplement"] = context.get("character_keyword_supplement", {})
        trace1["stage1_semantic_prelimit_count"] = context.get("stage1_semantic_prelimit_count", 0)
        trace1["stage1_semantic_budget_limit"] = stage1_limit_info
        answer_determined = False
        supporting_semantic_contents = decision_resp.get("supporting_semantic_contents", []) or []
        need_expand_semantic_contents: List[Any] = []
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
        supporting_semantics, min_selection_info = self._ensure_min_selected_semantics(
            query=query,
            selected_semantics=supporting_semantics,
            candidate_semantics=candidate_support_pool,
        )
        fallback_to_stage1_all_semantics = False
        final_evidence_fallback_info = {
            "enabled": False,
            "reason": "agent_selected_semantics",
            "stage1_semantic_count": len(candidate_support_pool),
            "final_semantic_count_before_fallback": len(supporting_semantics),
            "min_selection_enforcement": min_selection_info,
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
                user_data=user_data,
            )
        elif is_inference:
            final_answer, tok2, trace2 = self._final_inference_answer_with_situation(
                exp_name=exp_name,
                query=query,
                supporting_semantics=supporting_semantics,
                matched_messages=matched_messages,
                involves_time=mental_result.get("involves_time", False),
                user_data=user_data,
            )
        else:
            final_answer, tok2, trace2 = self._final_factual_answer_with_situation(
                exp_name=exp_name,
                query=query,
                supporting_semantics=supporting_semantics,
                matched_messages=matched_messages,
                involves_time=mental_result.get("involves_time", False),
                user_data=user_data,
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
            "character_keyword_supplement": context.get("character_keyword_supplement", {}),
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
        answer_determined = False
        supporting_semantic_contents = decision_resp.get("supporting_semantic_contents", []) or []
        need_expand_semantic_contents: List[Any] = []
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
        supporting_semantics, min_selection_info = self._ensure_min_selected_semantics(
            query=query,
            selected_semantics=supporting_semantics,
            candidate_semantics=candidate_support_pool,
        )
        fallback_to_stage1_all_semantics = False
        final_evidence_fallback_info = {
            "enabled": False,
            "reason": "agent_selected_semantics",
            "stage1_semantic_count": len(candidate_support_pool),
            "final_semantic_count_before_fallback": len(supporting_semantics),
            "min_selection_enforcement": min_selection_info,
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
                user_data=user_data,
            )
        elif is_inference:
            final_answer, tok2, trace2 = self._final_inference_answer_with_situation(
                exp_name=exp_name,
                query=query,
                supporting_semantics=supporting_semantics,
                matched_messages=matched_messages,
                involves_time=False,
                user_data=user_data,
            )
        else:
            final_answer, tok2, trace2 = self._final_factual_answer_with_situation(
                exp_name=exp_name,
                query=query,
                supporting_semantics=supporting_semantics,
                matched_messages=matched_messages,
                involves_time=False,
                user_data=user_data,
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
        stage1_with_timestamp = self._should_include_timestamp_in_situation_selection(mental_result)
        decision_resp, tok1, trace1 = self._decision_with_situation(
            exp_name=exp_name,
            query=query,
            situation_evidence=selected_semantics,
            situation_model_evidence=[],
            involves_time=stage1_with_timestamp,
        )
        trace1["third_round_timestamp_policy"] = {
            "with_timestamp": stage1_with_timestamp,
            "reason": "explicit_non_null_time_keyword_required",
            "original_involves_time": mental_result.get("involves_time", False),
        }
        answer_determined = False
        supporting_semantic_contents = decision_resp.get("supporting_semantic_contents", []) or []
        need_expand_semantic_contents: List[Any] = []
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
        supporting_semantics, min_selection_info = self._ensure_min_selected_semantics(
            query=query,
            selected_semantics=supporting_semantics,
            candidate_semantics=candidate_support_pool,
        )
        fallback_to_stage1_all_semantics = False
        final_evidence_fallback_info = {
            "enabled": False,
            "reason": "agent_selected_semantics",
            "stage1_semantic_count": len(candidate_support_pool),
            "final_semantic_count_before_fallback": len(supporting_semantics),
            "min_selection_enforcement": min_selection_info,
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
                user_data=user_data,
            )
        elif is_inference:
            final_answer, tok2, trace2 = self._final_inference_answer_with_situation(
                exp_name=exp_name,
                query=query,
                supporting_semantics=supporting_semantics,
                matched_messages=matched_messages,
                involves_time=mental_result.get("involves_time", False),
                user_data=user_data,
            )
        else:
            final_answer, tok2, trace2 = self._final_factual_answer_with_situation(
                exp_name=exp_name,
                query=query,
                supporting_semantics=supporting_semantics,
                matched_messages=matched_messages,
                involves_time=mental_result.get("involves_time", False),
                user_data=user_data,
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
