from __future__ import annotations

import contextlib
import json
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

from worldmem.construction.builder import (
    DEFAULT_SEMANTIC_LABELS,
    SITUATION_EXCLUDED_LABELS,
    MemoryBuilder,
)
from worldmem.models import (
    AtomicMemoryEntry,
    MessageBlock,
    SemanticMemory,
    SituationAggregate,
    SituationMemory,
)

GPT5_PERLTQA_COMPLETION_TOKENS = 32768


class PerLTQABuilder(MemoryBuilder):
    default_labels = list(DEFAULT_SEMANTIC_LABELS) + [
        "Appearance",
        "Achievement",
        "Awards & Role Models",
    ]

    @staticmethod
    def _coerce_naive_datetime(value):
        """Return a timezone-naive datetime for safe PerLTQA timestamp comparison.

        PerLTQA event/dialogue records contain a single logical timestamp. Some
        upstream parsers/model outputs may attach an offset such as +00:00 while
        other generated/profile records use naive datetime.now(). Python refuses
        to compare offset-aware and offset-naive datetimes, so this helper strips
        tzinfo while preserving the dataset's wall-clock date/time.
        """
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.replace(tzinfo=None) if value.tzinfo is not None else value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
                return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt
            except Exception:
                return None
        return None

    def _api_llm_scope(
        self,
        reasoning_effort: str = "minimal",
        min_completion_tokens: int = GPT5_PERLTQA_COMPLETION_TOKENS,
    ):
        """Stage-local GPT-5/OpenAI API policy.

        For non-OpenAI/HF clients this is a no-op.  For the unified
        `LLMClient(backend=\"openai_api\")`, it sets thread-local
        reasoning_effort and max_completion_tokens policy for the current
        construction stage.
        """
        client = getattr(self, "llm_client", None)
        if client is not None and hasattr(client, "api_call_scope"):
            return client.api_call_scope(
                reasoning_effort=reasoning_effort, min_completion_tokens=min_completion_tokens
            )
        return contextlib.nullcontext()

    def _extract_atomic_entries_from_single_block(
        self, owner_id: str, block: MessageBlock
    ) -> List[AtomicMemoryEntry]:
        prompt = """
Your task is to extract all valuable FACTUAL information from the following dialogues and convert them into structured memory entries.

[Current Message Block]
{0}

[Requirements]
1. **Extract Facts, Not Social Gestures**: SKIP greetings, thank-yous, compliments, and generic praise. Only extract entries that contain novel factual information (events, activities, plans, preferences, relationships, specific details like names/titles/numbers).
2. **Source Dialogue IDs**: Each dialogue line starts with an ID in brackets like [ID:42]. For each memory entry, list the dialogue IDs that the entry was derived from in the \"source_dialogue_ids\" field. This is CRITICAL for tracing back to original context.
3. **Force Disambiguation**: Absolutely PROHIBIT using pronouns (he, she, it, they, this, that). Always use the person's actual name. Every memory MUST explicitly state WHO did/said/experienced the thing.
4. **Resolve Temporal References**: Convert ALL relative time expressions to absolute dates based on the dialogue timestamp:
   - \"yesterday\" on May 8 -> May 7
   - \"last year\" in 2023 -> 2022
   - \"last week\" -> compute the actual date
   - \"next month\" -> compute the actual month
   The \"timestamp\" field should be the EVENT time, NOT the conversation time.
5. **Atomic Facts**: Extract individual facts as SEPARATE entries. If one dialogue mentions 3 activities, create 3 entries. Do not merge unrelated facts into one summary.
6. **Preserve Specific Details**: Always capture exact names (people, pets, books, songs), exact numbers (durations, counts, ages), and specific entities.
7. **Identify Described-But-Unnamed Things Using World Knowledge**: When the dialogue describes something without naming it, IDENTIFY it by name in the memory entry. This is CRITICAL — future queries will search by name, not by description.
   - A study method like \"25 minutes on, 5 minutes off\" → identify as \"Pomodoro technique\"
   - A composer whose music is in a named movie → identify the composer (e.g. Harry Potter soundtrack → John Williams)
   - A game described by its mechanics → identify the game (e.g. \"card game where you find the imposter\" → Mafia, \"game with colored cards you match\" → UNO)
   - A location described by its features → identify it (e.g. \"national park in northern Minnesota with lakes\" → Voyageurs National Park)
   - A shop/brand described by what it does → identify it (e.g. \"they made all the props for Harry Potter\" → MinaLima / House of MinaLima)
   - A health condition implied by symptoms → identify it (e.g. frequent overeating + weight gain → obesity risk)
   - A geographic location implied by context → identify the state/country (e.g. a shelter in a named city → identify the state)
   Include BOTH the original description AND the identified name in the lossless_restatement, and add the identified name to keywords and entities.
8. **Precise Extraction**:
   - keywords: Core keywords (names, places, entities, topic words)
   - timestamp: Absolute time of the EVENT in ISO 8601 format (resolved from relative expressions)
   - location: Specific location name (if mentioned)
   - persons: All person names mentioned
   - entities: Companies, products, organizations, book titles, song names, etc.
   - topic: The topic of this information

[Output Format]
Return a JSON array, each element is a memory entry:

```json
[
  {{
    \"lossless_restatement\": \"Complete unambiguous restatement (must include WHO, WHAT, WHEN, WHERE)\",
    \"keywords\": [\"keyword1\", \"keyword2\", ...],
    \"timestamp\": \"YYYY-MM-DDTHH:MM:SS or null\",
    \"location\": \"location name or null\",
    \"persons\": [\"name1\", \"name2\", ...],
    \"entities\": [\"entity1\", \"entity2\", ...],
    \"topic\": \"topic phrase\",
    \"source_message_indices\": [42, 43]
  }},
  ...
]
```

[Example]
Dialogues:
[ID:1] [2025-11-15T14:30:00] Alice: I just started working at Google! Bob, let's meet at Starbucks tomorrow at 2pm.
[ID:2] [2025-11-15T14:31:00] Bob: Congrats! I've been playing tennis a lot lately.
[ID:3] [2025-11-15T14:32:00] Alice: Nice! I've been studying with that method where you work 25 minutes then take a 5-minute break. It really helps!
[ID:4] [2025-11-15T14:33:00] Bob: I love playing that theme song from the wizard movie on piano. You know, the one with the boy who goes to magic school.

Output:
```json
[
  {{
    \"lossless_restatement\": \"Alice started working at Google as of 2025-11-15.\",
    \"keywords\": [\"Alice\", \"Google\", \"employment\"],
    \"timestamp\": \"2025-11-15T14:30:00\",
    \"location\": null,
    \"persons\": [\"Alice\"],
    \"entities\": [\"Google\"],
    \"topic\": \"Alice's new job\",
    \"source_message_indices\": [1]
  }},
  {{
    \"lossless_restatement\": \"Alice suggested meeting with Bob at Starbucks on 2025-11-16T14:00:00.\",
    \"keywords\": [\"Alice\", \"Bob\", \"Starbucks\", \"meeting\"],
    \"timestamp\": \"2025-11-16T14:00:00\",
    \"location\": \"Starbucks\",
    \"persons\": [\"Alice\", \"Bob\"],
    \"entities\": [],
    \"topic\": \"Meeting arrangement\",
    \"source_message_indices\": [1]
  }},
  {{
    \"lossless_restatement\": \"Bob has been playing tennis frequently as of 2025-11-15.\",
    \"keywords\": [\"Bob\", \"tennis\"],
    \"timestamp\": \"2025-11-15T14:31:00\",
    \"location\": null,
    \"persons\": [\"Bob\"],
    \"entities\": [],
    \"topic\": \"Bob's hobby\",
    \"source_message_indices\": [2]
  }},
  {{
    \"lossless_restatement\": \"Alice uses the Pomodoro technique (25 minutes work, 5-minute break) for studying.\",
    \"keywords\": [\"Alice\", \"Pomodoro technique\", \"studying\", \"time management\"],
    \"timestamp\": \"2025-11-15T14:32:00\",
    \"location\": null,
    \"persons\": [\"Alice\"],
    \"entities\": [\"Pomodoro technique\"],
    \"topic\": \"Alice's study method\",
    \"source_message_indices\": [3]
  }},
  {{
    \"lossless_restatement\": \"Bob enjoys playing the Harry Potter theme song (composed by John Williams) on piano.\",
    \"keywords\": [\"Bob\", \"piano\", \"Harry Potter\", \"John Williams\"],
    \"timestamp\": \"2025-11-15T14:33:00\",
    \"location\": null,
    \"persons\": [\"Bob\"],
    \"entities\": [\"Harry Potter\", \"John Williams\"],
    \"topic\": \"Bob's piano playing\",
    \"source_message_indices\": [4]
  }}
]
```

Note how \"tomorrow\" was resolved to 2025-11-16, the greeting was skipped, described-but-unnamed things were identified (Pomodoro technique, John Williams), and Alice's job and meeting were extracted as separate atomic facts.

Now process the above dialogues. Return ONLY the JSON array, no other explanations.
    """.format(block.content)
        with self._api_llm_scope(
            reasoning_effort="medium", min_completion_tokens=GPT5_PERLTQA_COMPLETION_TOKENS
        ):
            resp = self.llm_client.generate_json_response(
                prompt=prompt,
                temperature=0.0,
                max_tokens=GPT5_PERLTQA_COMPLETION_TOKENS,
                default_response=[],
                enable_thinking=True,
            )
        if isinstance(resp, dict):
            resp = [resp]
        if not isinstance(resp, list):
            return []
        entries = []
        for item in resp:
            try:
                idxs = []
                for x in item.get("source_message_indices", []):
                    try:
                        idxs.append(int(x))
                    except Exception:
                        continue
                msg_ids = []
                for idx in idxs:
                    if 0 <= idx < len(block.source_message_ids):
                        msg_ids.append(block.source_message_ids[idx])
                ts = item.get("timestamp")
                entry = AtomicMemoryEntry(
                    user_id=owner_id,
                    block_id=block.block_id,
                    lossless_restatement=item.get("lossless_restatement", "").strip(),
                    keywords=item.get("keywords", []),
                    timestamp=datetime.fromisoformat(ts) if ts else None,
                    location=item.get("location"),
                    persons=item.get("persons", []),
                    entities=item.get("entities", []),
                    topic=item.get("topic"),
                    source_message_ids=msg_ids,
                    source_message_indices=idxs,
                    metadata={"extraction_source": "message_block_atomic"},
                )
                if entry.lossless_restatement:
                    entries.append(entry)
            except Exception:
                continue
        return entries

    def _convert_atomic_entries_to_candidate_semantics(
        self,
        owner_id: str,
        atomic_entries: List[AtomicMemoryEntry],
        message_blocks: List[MessageBlock],
    ) -> List[SemanticMemory]:
        candidates = []
        block_map = {b.block_id: b for b in message_blocks}
        for entry in atomic_entries:
            matched_message_id = None
            matched_message_index = None
            matched_message_content = None
            if len(entry.source_message_ids) == 1:
                matched_message_id = entry.source_message_ids[0]
                if entry.source_message_indices:
                    matched_message_index = entry.source_message_indices[0]
                    block = block_map.get(entry.block_id)
                    if block and 0 <= matched_message_index < len(block.messages):
                        matched_message_content = block.messages[matched_message_index].get(
                            "content"
                        )
            else:
                matched_message_id, matched_message_index, matched_message_content = (
                    self._pick_best_message_for_semantic_text(
                        semantic_text=entry.lossless_restatement,
                        block=block_map.get(entry.block_id),
                    )
                )
            fallback_ts = entry.timestamp
            if fallback_ts is None:
                block = block_map.get(entry.block_id)
                if (
                    block
                    and matched_message_index is not None
                    and (0 <= matched_message_index < len(block.messages))
                ):
                    raw_ts = block.messages[matched_message_index].get("timestamp")
                    if raw_ts:
                        try:
                            fallback_ts = datetime.fromisoformat(raw_ts)
                        except Exception:
                            fallback_ts = None
            sem = SemanticMemory(
                user_id=owner_id,
                content=entry.lossless_restatement,
                knowledge_type="knowledge",
                created_at=fallback_ts or datetime.now(),
                timestamp=fallback_ts,
                confidence=1.0,
                metadata={
                    "generation_method": "message_block_atomic_candidate",
                    "topic": entry.topic,
                    "location": entry.location,
                    "persons": entry.persons,
                    "entities": entry.entities,
                },
                matched_message_id=matched_message_id,
                matched_message_index=matched_message_index,
                matched_message_content=matched_message_content,
                matched_block_id=entry.block_id,
                source_message_ids=entry.source_message_ids,
                source_message_indices=entry.source_message_indices,
            )
            candidates.append(sem)
        return candidates

    def _rebuild_situations_from_keywords(
        self,
        owner_id: str,
        created_episodes: List[Dict[str, Any]],
        semantic_memories: List[SemanticMemory],
    ) -> List[SituationMemory]:
        if not semantic_memories:
            self._situation_storage.overwrite_all(owner_id, [])
            return []
        keyword_to_labels = self._collect_global_keyword_label_map(semantic_memories)
        keyword_bucket = defaultdict(list)

        def _semantic_sort_key(sem: SemanticMemory):
            block_id = getattr(sem, "matched_block_id", "") or ""
            msg_idx = getattr(sem, "matched_message_index", None)
            if msg_idx is None:
                msg_idx = 10**9
            created = self._coerce_naive_datetime(getattr(sem, "created_at", None)) or datetime.max
            return (block_id, msg_idx, created)

        ordered_semantics = sorted(semantic_memories, key=_semantic_sort_key)
        for sem in ordered_semantics:
            for ann in getattr(sem, "label_annotations", []) or []:
                label = self._normalize_label(getattr(ann, "label", ""))
                if not label or label in SITUATION_EXCLUDED_LABELS:
                    continue
                for kw in getattr(ann, "keywords", []) or []:
                    if not isinstance(kw, str):
                        continue
                    kw = kw.strip()
                    if not kw:
                        continue
                    keyword_bucket[kw].append((label, sem))
        situations: List[SituationMemory] = []
        for keyword in sorted(keyword_bucket.keys()):
            pairs = keyword_bucket[keyword]
            labels_for_keyword = [
                lbl
                for lbl in keyword_to_labels.get(keyword, [])
                if self._normalize_label(lbl) not in SITUATION_EXCLUDED_LABELS
            ]
            keyword_label = labels_for_keyword[0] if labels_for_keyword else None
            aggregates = []
            timestamp_start = None
            timestamp_end = None
            for label, sem in pairs:
                ep_id = sem.source_episodes[0] if getattr(sem, "source_episodes", None) else None
                ts = self._coerce_naive_datetime(getattr(sem, "timestamp", None))
                if ts is None:
                    ts = self._coerce_naive_datetime(getattr(sem, "created_at", None))
                agg = SituationAggregate(
                    episode_id=ep_id,
                    content=sem.content,
                    semantic_ids=[sem.memory_id],
                    timestamp_start=ts,
                    timestamp_end=ts,
                )
                aggregates.append(agg)
                if ts is not None:
                    if timestamp_start is None or ts < timestamp_start:
                        timestamp_start = ts
                    if timestamp_end is None or ts > timestamp_end:
                        timestamp_end = ts
            situation = SituationMemory(
                user_id=owner_id,
                summary=keyword,
                routing_summary=keyword,
                final_summary=keyword,
                aggregates=aggregates,
                timestamp_start=timestamp_start,
                timestamp_end=timestamp_end,
                metadata={"label": keyword_label} if keyword_label else {},
            )
            situations.append(situation)
        self._situation_storage.overwrite_all(owner_id, situations)
        return situations

    def _normalize_label(self, label: str) -> str:
        if not label:
            return ""
        label = " ".join(str(label).strip().split())
        alias_map = {
            "characters": "Character",
            "character": "Character",
            "identity": "Identity",
            "identities": "Identity",
            "education": "Education",
            "career": "Career",
            "careers": "Career",
            "preference": "Preferences",
            "preferences": "Preferences",
            "health": "Health",
            "emotion": "Emotions & Moods",
            "emotions": "Emotions & Moods",
            "mood": "Emotions & Moods",
            "moods": "Emotions & Moods",
            "emotion & mood": "Emotions & Moods",
            "emotions & moods": "Emotions & Moods",
            "competence": "Competence & Experience",
            "experience": "Competence & Experience",
            "competence & experience": "Competence & Experience",
            "technical detail": "Technical Details",
            "technical details": "Technical Details",
            "relationship": "Relationship",
            "relationships": "Relationship",
            "goal": "Goals & Plans",
            "goals": "Goals & Plans",
            "plan": "Goals & Plans",
            "plans": "Goals & Plans",
            "goal & plan": "Goals & Plans",
            "goals & plans": "Goals & Plans",
            "belief": "Beliefs & Values",
            "beliefs": "Beliefs & Values",
            "value": "Beliefs & Values",
            "values": "Beliefs & Values",
            "belief & value": "Beliefs & Values",
            "beliefs & values": "Beliefs & Values",
            "habit": "Habits & Routines",
            "habits": "Habits & Routines",
            "routine": "Habits & Routines",
            "routines": "Habits & Routines",
            "habit & routine": "Habits & Routines",
            "habits & routines": "Habits & Routines",
            "art": "Art & Culture",
            "culture": "Art & Culture",
            "art & culture": "Art & Culture",
            "object": "Object",
            "objects": "Object",
            "location": "Location",
            "locations": "Location",
            "organization": "Organization",
            "organizations": "Organization",
            "action": "Action",
            "actions": "Action",
            "event": "Event",
            "events": "Event",
            "time": "Time",
            "appearance": "Appearance",
            "appearances": "Appearance",
            "physical characteristic": "Appearance",
            "physical characteristics": "Appearance",
            "achievement": "Achievement",
            "achievements": "Achievement",
            "award": "Awards & Role Models",
            "awards": "Awards & Role Models",
            "role model": "Awards & Role Models",
            "role models": "Awards & Role Models",
            "award & role model": "Awards & Role Models",
            "awards & role models": "Awards & Role Models",
            "awards and role models": "Awards & Role Models",
        }
        return alias_map.get(label.lower(), label)

    def _judge_whether_new_label_can_merge_into_existing(
        self,
        semantic_content: str,
        new_label: str,
        new_label_keywords: List[str],
        candidate_existing_label: str,
    ) -> Dict[str, Any]:
        preview_keywords = []
        for kw in new_label_keywords[:3]:
            if isinstance(kw, str) and kw.strip():
                preview_keywords.append(kw.strip())
        prompt = """
    You are reviewing a proposed new high-level semantic label.

    Semantic content:
    {0}

    Proposed new label:
    {1}

    Keywords / key phrases currently assigned to this new label:
    {2}

    Most similar existing label:
    {3}

    Task:
    Decide whether the proposed new label is actually unnecessary because its keywords / phrases can already be categorized under the most similar existing label.

    Rules:
    1. Prefer reusing the existing label whenever it can reasonably cover the keywords.
    2. Only keep the new label if it captures a genuinely different high-level information category.
    3. Do not keep the proposed new label merely because it is slightly more specific.
    4. High-level labels should stay broad, reusable, and category-like.
    5. If the keywords can be categorized under the existing label, return merge_to_existing=true.

    Return JSON only:
    {{
      \"merge_to_existing\": true,
      \"reason\": \"...\"
    }}
    """.format(
            semantic_content,
            new_label,
            json.dumps(preview_keywords, ensure_ascii=False),
            candidate_existing_label,
        )
        try:
            resp = self.llm_client.generate_json_response(
                prompt=prompt,
                temperature=0.0,
                max_tokens=GPT5_PERLTQA_COMPLETION_TOKENS,
                default_response={"merge_to_existing": True, "reason": ""},
            )
            return {
                "merge_to_existing": bool(resp.get("merge_to_existing", True)),
                "reason": str(resp.get("reason", "")).strip(),
            }
        except Exception:
            return {"merge_to_existing": True, "reason": "fallback_merge_on_exception"}

    def _extract_label_annotations_for_semantic(
        self, semantic: SemanticMemory, label_inventory: List[str]
    ) -> Dict[str, Any]:
        prompt = """
    You are annotating one semantic memory using a controlled inventory of high-level information-category labels.

    Semantic content:
    {0}

    Existing high-level labels:
    {1}

    Goal:
    Identify ALL meaningful information categories represented in the semantic content.

    Important rules:
    1. You must cover as many meaningful keywords / key phrases as possible from the semantic content.
    2. For each extracted keyword or key phrase, assign it to the most appropriate high-level label.
    3. You must FIRST reuse labels from the existing high-level labels list whenever possible.
    4. ONLY create a new label if the existing labels **truly cannot represent** that information category.
    5. Any new label must follow the style of the existing labels:
       - high-level
       - reusable
       - category-like
       - not overly specific
       - not focused on low-level details
    6. Do NOT create a new label that is similar to or overlapping with an existing label.
    7. Do NOT create labels for tiny details, one-off objects, or event-specific fragments if an existing higher-level label already covers them.
    8. Each label should map to one or more supporting keywords / key phrases from the semantic content.

    Example:
    Semantic content:
    Caroline received a necklace with a cross and a heart as a gift from her grandmother in Sweden.

    Good output:
    {{
      \"existing_labels\": {{
        \"Character\": [\"Caroline\", \"grandmother\"],
        \"Objects\": [\"necklace\"],
        \"Technical Details\": [\"with a cross and a heart\"],
        \"Actions\": [\"received\"],
        \"Locations\": [\"Sweden\"]
      }},
      \"new_labels\": {{}}
    }}

    Another example:
    Semantic content:
    Caroline regularly journals to reflect on her emotions and personal growth.

    Possible output:
    {{
      \"existing_labels\": {{
        \"Character\": [\"Caroline\"],
        \"Habits & Routines\": [\"regularly journals\"],
        \"Beliefs & Values\": [\"personal growth\"]
      }},
      \"new_labels\": {{
        \"Emotions & Moods\": [\"reflect on her emotions\"]
      }}
    }}

    In the second example, \"Emotions & Moods\" is only allowed when none of the existing labels fully captures that information category.
    When proposing a new label, make sure it is still high-level and reusable like the **Existing Labels**.

    Return JSON only in exactly this format:
    {{
      \"existing_labels\": {{
        \"Label A\": [\"keyword 1\", \"keyword 2\"],
        \"Label B\": [\"keyword 3\"]
      }},
      \"new_labels\": {{
        \"New Label A\": [\"keyword 4\"],
        \"New Label B\": [\"keyword 5\", \"keyword 6\"]
      }}
    }}
    """.format(semantic.content, json.dumps(label_inventory, ensure_ascii=False))
        try:
            with self._api_llm_scope(
                reasoning_effort="minimal", min_completion_tokens=GPT5_PERLTQA_COMPLETION_TOKENS
            ):
                resp = self.llm_client.generate_json_response(
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=GPT5_PERLTQA_COMPLETION_TOKENS,
                    default_response={"existing_labels": {}, "new_labels": {}},
                )
            return resp
        except Exception:
            return {"existing_labels": {}, "new_labels": {}}

    def _judge_sparse_new_label_final_merge(
        self, target_label: str, collected_keywords: List[str], candidate_existing_labels: List[str]
    ) -> Dict[str, Any]:
        preview_keywords = []
        for kw in collected_keywords[:20]:
            if isinstance(kw, str) and kw.strip():
                preview_keywords.append(kw.strip())
        prompt = """
    You are reviewing a newly created high-level semantic label with other existing labels.

    New label:
    {0}

    Keywords / key phrases associated with this new label:
    {1}

    Most related existing labels:
    {2}

    Task:
    Decide whether this new label should remain as an independent label, or be merged into one of the existing labels above.

    Rules:
    1. If one of the existing labels can reasonably cover the meaning of this new label and its keywords, merge it.
    2. Only keep this new label if it represents a genuinely independent, reusable, high-level information category.
    3. Do not keep it merely because it is more specific, or more detailed.
    4. If merging, choose exactly one best existing label.

    Return JSON only:
    {{
      \"merge\": true,
      \"target_label\": \"one of the existing labels above\",
      \"reason\": \"...\"
    }}
    """.format(
            target_label,
            json.dumps(preview_keywords, ensure_ascii=False),
            json.dumps(candidate_existing_labels, ensure_ascii=False),
        )
        try:
            resp = self.llm_client.generate_json_response(
                prompt=prompt,
                temperature=0.0,
                max_tokens=GPT5_PERLTQA_COMPLETION_TOKENS,
                default_response={"merge": False, "target_label": "", "reason": ""},
            )
            return {
                "merge": bool(resp.get("merge", False)),
                "target_label": self._normalize_label(resp.get("target_label", "")),
                "reason": str(resp.get("reason", "")).strip(),
            }
        except Exception:
            return {"merge": False, "target_label": "", "reason": "fallback_keep_on_exception"}
