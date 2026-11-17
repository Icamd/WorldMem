from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from worldmem.generation.episode_generator import EpisodeGenerator
from worldmem.generation.prompts import PromptTemplates
from worldmem.generation.semantic_generator import SemanticGenerator
from worldmem.models import (
    AtomicMemoryEntry,
    Episode,
    MentalCluster,
    MentalMemory,
    MentalSemanticLink,
    Message,
    MessageBlock,
    SemanticLabelAnnotation,
    SemanticMemory,
    SituationAggregate,
    SituationMemory,
)
from worldmem.storage import JsonlRepository

logger = logging.getLogger(__name__)
DEFAULT_SEMANTIC_LABELS = [
    "Character",
    "Identity",
    "Education",
    "Career",
    "Preferences",
    "Health",
    "Emotions & Moods",
    "Competence & Experience",
    "Technical Details",
    "Relationship",
    "Goals & Plans",
    "Beliefs & Values",
    "Habits & Routines",
    "Art & Culture",
    "Object",
    "Location",
    "Organization",
    "Action",
    "Event",
    "Time",
]
SITUATION_EXCLUDED_LABELS = {"Character", "Person", "Time"}


class MemoryBuilder:
    """Construct the four WorldMem levels from complete interaction histories."""

    default_labels = DEFAULT_SEMANTIC_LABELS

    def __init__(self, config, llm_client, embedding_client):
        self.config = config
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.prompts = PromptTemplates()
        self._episode_repository = JsonlRepository(
            config.storage_path, "episodes", Episode, "episode_id"
        )
        self._semantic_repository = JsonlRepository(
            config.storage_path, "semantic", SemanticMemory, "memory_id"
        )
        self._situation_storage = JsonlRepository(
            config.storage_path, "situation", SituationMemory, "situation_id"
        )
        self._mental_storage = JsonlRepository(
            config.storage_path, "mental", MentalMemory, "mental_id"
        )
        self._message_block_dir = Path(config.storage_path) / "message_blocks"
        self._message_block_dir.mkdir(parents=True, exist_ok=True)
        self._episode_generator = EpisodeGenerator(llm_client, config)
        self._semantic_generator = SemanticGenerator(llm_client, embedding_client, config)
        self._user_processing_locks = {}
        self._user_locks_manager = threading.RLock()
        self.stats = dict(
            messages_processed=0,
            episodes_created=0,
            semantic_memories_created=0,
            total_processing_time=0.0,
        )
        self.stats_lock = threading.Lock()

    def _api_llm_scope(
        self, *, reasoning_effort: str = "minimal", min_completion_tokens: int = 4096
    ):
        """Return a thread-local OpenAI API policy scope when supported.

        HF/vLLM-compatible clients that do not implement api_call_scope simply
        get a no-op context.  For GPT-5-nano through openai_api, MemorySystem
        uses this to assign different reasoning budgets by construction stage:
        semantic extraction uses medium, while episode summaries and label
        annotation use minimal.
        """
        if hasattr(self.llm_client, "api_call_scope"):
            floor = (
                self.config.construction_semantic_min_tokens
                if reasoning_effort == "medium"
                else self.config.construction_regular_min_tokens
            )
            return self.llm_client.api_call_scope(
                reasoning_effort=reasoning_effort,
                min_completion_tokens=max(floor, min_completion_tokens),
            )
        return contextlib.nullcontext()

    def _get_user_processing_lock(self, owner_id: str) -> threading.RLock:
        if owner_id in self._user_processing_locks:
            return self._user_processing_locks[owner_id]
        with self._user_locks_manager:
            if owner_id not in self._user_processing_locks:
                self._user_processing_locks[owner_id] = threading.RLock()
            return self._user_processing_locks[owner_id]

    @property
    def episode_generator(self):
        return self._episode_generator

    @property
    def semantic_generator(self):
        return self._semantic_generator

    def build_world_memory_from_messages(
        self,
        owner_id: str,
        messages: List[Dict[str, Any]],
        batch_size: int = 40,
        enable_message_semantic_backfill: bool = True,
    ) -> Dict[str, Any]:
        start_time = time.time()
        user_lock = self._get_user_processing_lock(owner_id)
        with user_lock:
            message_objects = self._convert_dicts_to_messages(messages)
            with self.stats_lock:
                self.stats["messages_processed"] += len(message_objects)
            created_episodes = self._segment_and_create_episodes(
                owner_id=owner_id, messages=message_objects, batch_size=batch_size
            )
            message_blocks = self._build_message_blocks_from_episodes(
                owner_id=owner_id, created_episodes=created_episodes
            )
            self._persist_message_blocks(owner_id, message_blocks)
            for episode_info in created_episodes:
                episode_obj = episode_info.get("episode_object")
                if episode_obj is None:
                    continue
                self._async_generate_semantic_memories(owner_id, episode_obj)
            semantic_memories = self._semantic_repository.list_by_user(owner_id)
            semantic_memories = self._align_episode_semantics_to_messages(
                owner_id=owner_id,
                created_episodes=created_episodes,
                semantic_memories=semantic_memories,
                message_blocks=message_blocks,
            )
            semantic_memories = self._annotate_semantics_with_labels(owner_id=owner_id)
            atomic_entries = self._extract_atomic_entries_from_message_blocks(
                owner_id=owner_id, message_blocks=message_blocks
            )
            candidate_semantics = self._convert_atomic_entries_to_candidate_semantics(
                owner_id=owner_id, atomic_entries=atomic_entries, message_blocks=message_blocks
            )
            semantic_memories, accepted_candidate_semantics = (
                self._merge_candidate_semantics_by_message_gain(
                    owner_id=owner_id,
                    semantic_memories=semantic_memories,
                    candidate_semantics=candidate_semantics,
                )
            )
            if accepted_candidate_semantics:
                semantic_memories = self._annotate_specific_semantics_with_labels(
                    owner_id=owner_id,
                    semantic_memories=semantic_memories,
                    target_memory_ids={x.memory_id for x in accepted_candidate_semantics},
                )
            semantic_memories = self._backfill_missing_keywords_across_semantics(
                owner_id=owner_id, semantic_memories=semantic_memories
            )
            situations = self._rebuild_situations_from_keywords(
                owner_id=owner_id,
                created_episodes=created_episodes,
                semantic_memories=semantic_memories,
            )
            mental_records = self._build_mental_memory(
                owner_id=owner_id,
                situations=situations,
                label_cluster_threshold=getattr(self.config, "mental_label_cluster_threshold", 0.8),
            )
            with self.stats_lock:
                self.stats["episodes_created"] += len(created_episodes)
                self.stats["semantic_memories_created"] += len(semantic_memories)
                self.stats["total_processing_time"] += time.time() - start_time
            return {
                "episodes_created": len(created_episodes),
                "message_blocks_created": len(message_blocks),
                "semantic_memories_created": len(semantic_memories),
                "accepted_atomic_semantics": len(accepted_candidate_semantics),
                "situations_created": len(situations),
                "mental_units_created": len(mental_records) if mental_records else 0,
            }

    def _build_message_blocks_from_episodes(
        self, owner_id: str, created_episodes: List[Dict[str, Any]]
    ) -> List[MessageBlock]:
        blocks = []
        for episode_info in created_episodes:
            episode_obj = episode_info.get("episode_object")
            if episode_obj is None:
                continue
            original_messages = getattr(episode_obj, "original_messages", []) or []
            block_messages = []
            source_message_ids = []
            source_message_indices = []
            for idx, msg_data in enumerate(original_messages):
                if hasattr(msg_data, "to_dict"):
                    msg_dict = msg_data.to_dict()
                else:
                    msg_dict = msg_data
                block_messages.append(msg_dict)
                source_message_ids.append(
                    msg_dict.get("message_id", f"{episode_obj.episode_id}_{idx}")
                )
                source_message_indices.append(idx)
            content = "\n".join(
                [
                    f"[ID:{i}] [{m.get('timestamp', '')}] {m.get('role', '')}: {m.get('content', '')}"
                    for i, m in enumerate(block_messages)
                ]
            )
            block = MessageBlock(
                user_id=owner_id,
                episode_id=episode_obj.episode_id,
                source_message_ids=source_message_ids,
                source_message_indices=source_message_indices,
                messages=block_messages,
                content=content,
                timestamp_start=episode_obj.time_range_start,
                timestamp_end=episode_obj.time_range_end,
                metadata={"episode_id": episode_obj.episode_id},
            )
            blocks.append(block)
        return blocks

    def _persist_message_blocks(self, owner_id: str, message_blocks: List[MessageBlock]) -> None:
        if not message_blocks:
            return
        file_path = self._message_block_dir / f"{owner_id}_message_blocks.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for block in message_blocks:
                f.write(json.dumps(block.to_dict(), ensure_ascii=False) + "\n")

    def _extract_atomic_entries_from_message_blocks(
        self, owner_id: str, message_blocks: List[MessageBlock]
    ) -> List[AtomicMemoryEntry]:
        all_entries = []
        for block in message_blocks:
            entries = self._extract_atomic_entries_from_single_block(owner_id, block)
            all_entries.extend(entries)
        return all_entries

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
        with self._api_llm_scope(reasoning_effort="medium", min_completion_tokens=16384):
            resp = self.llm_client.generate_json_response(
                prompt=prompt,
                temperature=0.0,
                max_tokens=16384,
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

    def _pick_best_message_for_semantic_text(
        self, semantic_text: str, block: Optional[MessageBlock]
    ):
        if block is None or not block.messages:
            return (None, None, None)
        best_idx = None
        best_score = -1.0
        sem_emb = self.embedding_client.embed_text(semantic_text)
        for idx, msg in enumerate(block.messages):
            content = msg.get("content", "")
            if not content:
                continue
            msg_emb = self.embedding_client.embed_text(content)
            score = self.embedding_client.cosine_similarity(sem_emb, msg_emb)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            return (None, None, None)
        msg = block.messages[best_idx]
        msg_id = msg.get("message_id")
        if not msg_id and best_idx < len(block.source_message_ids):
            msg_id = block.source_message_ids[best_idx]
        return (msg_id, best_idx, msg.get("content"))

    def _align_episode_semantics_to_messages(
        self,
        owner_id: str,
        created_episodes: List[Dict[str, Any]],
        semantic_memories: List[SemanticMemory],
        message_blocks: List[MessageBlock],
    ) -> List[SemanticMemory]:
        episode_to_block = {}
        for episode_info, block in zip(created_episodes, message_blocks):
            ep = episode_info.get("episode_object")
            if ep is not None:
                episode_to_block[ep.episode_id] = block
        for sem in semantic_memories:
            source_episodes = getattr(sem, "source_episodes", []) or []
            if not source_episodes:
                continue
            ep_id = source_episodes[0]
            block = episode_to_block.get(ep_id)
            if block is None:
                continue
            msg_id, msg_idx, msg_content = self._pick_best_message_for_semantic_text(
                semantic_text=sem.content, block=block
            )
            sem.matched_message_id = msg_id
            sem.matched_message_index = msg_idx
            sem.matched_message_content = msg_content
            sem.matched_block_id = block.block_id
            sem.source_message_ids = [msg_id] if msg_id else []
            sem.source_message_indices = [msg_idx] if msg_idx is not None else []
        self._semantic_repository.overwrite_all(owner_id, semantic_memories)
        return semantic_memories

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
            sem = SemanticMemory(
                user_id=owner_id,
                content=entry.lossless_restatement,
                knowledge_type="knowledge",
                created_at=entry.timestamp or datetime.now(),
                timestamp=entry.timestamp,
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

    def _merge_candidate_semantics_by_message_gain(
        self,
        owner_id: str,
        semantic_memories: List[SemanticMemory],
        candidate_semantics: List[SemanticMemory],
    ):
        existing_message_ids = set()
        for sem in semantic_memories:
            if getattr(sem, "matched_message_id", None):
                existing_message_ids.add(sem.matched_message_id)
        accepted = []
        for cand in candidate_semantics:
            msg_id = getattr(cand, "matched_message_id", None)
            if not msg_id:
                continue
            if msg_id in existing_message_ids:
                continue
            cand_id = self._semantic_repository.save(cand)
            cand.memory_id = cand_id
            semantic_memories.append(cand)
            accepted.append(cand)
            existing_message_ids.add(msg_id)
        if accepted:
            self._semantic_repository.overwrite_all(owner_id, semantic_memories)
        return (semantic_memories, accepted)

    def _annotate_specific_semantics_with_labels(
        self, owner_id: str, semantic_memories: List[SemanticMemory], target_memory_ids: set
    ) -> List[SemanticMemory]:
        if not semantic_memories or not target_memory_ids:
            return semantic_memories
        label_inventory = list(self.default_labels)
        for sem in semantic_memories:
            if sem.memory_id not in target_memory_ids:
                continue
            if getattr(sem, "label_annotations", None):
                continue
            resp = self._extract_label_annotations_for_semantic(
                semantic=sem, label_inventory=label_inventory
            )
            existing_labels_raw = resp.get("existing_labels", {}) or {}
            new_labels_raw = resp.get("new_labels", {}) or {}
            annotations = []
            for bucket in (existing_labels_raw, new_labels_raw):
                for label, keywords in bucket.items():
                    norm_label = self._normalize_label(label)
                    if not norm_label:
                        continue
                    clean_keywords = []
                    seen = set()
                    for kw in keywords or []:
                        if not isinstance(kw, str):
                            continue
                        kw = kw.strip()
                        if not kw or kw.lower() in seen:
                            continue
                        seen.add(kw.lower())
                        clean_keywords.append(kw)
                    annotations.append(
                        SemanticLabelAnnotation(
                            label=norm_label,
                            keywords=clean_keywords,
                            is_new_label=False,
                            metadata={"source": "existing_label_inventory"},
                        )
                    )
            sem.label_annotations = annotations
            sem = self._merge_semantic_label_annotations_by_label(sem)
        self._semantic_repository.overwrite_all(owner_id, semantic_memories)
        return semantic_memories

    def _collect_global_keyword_label_map(
        self, semantic_memories: List[SemanticMemory]
    ) -> Dict[str, List[str]]:
        keyword_to_labels = defaultdict(set)
        for sem in semantic_memories:
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
                    keyword_to_labels[kw].add(label)
        return {kw: sorted(list(labels)) for kw, labels in keyword_to_labels.items()}

    def _merge_semantic_label_annotations_by_label(
        self, semantic: SemanticMemory
    ) -> SemanticMemory:
        merged = {}
        order = []
        for ann in getattr(semantic, "label_annotations", []) or []:
            label = self._normalize_label(getattr(ann, "label", ""))
            if not label:
                continue
            if label not in merged:
                merged[label] = {
                    "keywords": [],
                    "seen": set(),
                    "is_new_label": bool(getattr(ann, "is_new_label", False)),
                    "metadata": getattr(ann, "metadata", {}) or {},
                }
                order.append(label)
            for kw in getattr(ann, "keywords", []) or []:
                if not isinstance(kw, str):
                    continue
                kw = kw.strip()
                if not kw:
                    continue
                if kw.lower() in merged[label]["seen"]:
                    continue
                merged[label]["seen"].add(kw.lower())
                merged[label]["keywords"].append(kw)
        semantic.label_annotations = [
            SemanticLabelAnnotation(
                label=label,
                keywords=merged[label]["keywords"],
                is_new_label=merged[label]["is_new_label"],
                metadata=merged[label]["metadata"],
            )
            for label in order
        ]
        return semantic

    def _backfill_missing_keywords_across_semantics(
        self, owner_id: str, semantic_memories: List[SemanticMemory]
    ) -> List[SemanticMemory]:
        if not semantic_memories:
            return semantic_memories
        keyword_to_labels = self._collect_global_keyword_label_map(semantic_memories)
        for sem in semantic_memories:
            content = sem.content or ""
            if not content:
                continue
            existing_keywords = set()
            for ann in getattr(sem, "label_annotations", []) or []:
                label = self._normalize_label(getattr(ann, "label", ""))
                if not label or label in SITUATION_EXCLUDED_LABELS:
                    continue
                for kw in getattr(ann, "keywords", []) or []:
                    if isinstance(kw, str) and kw.strip():
                        existing_keywords.add(kw.strip().lower())
            additions: List[SemanticLabelAnnotation] = []
            for keyword, labels in keyword_to_labels.items():
                if not keyword:
                    continue
                valid_labels = [
                    lbl
                    for lbl in labels
                    if self._normalize_label(lbl) not in SITUATION_EXCLUDED_LABELS
                ]
                if not valid_labels:
                    continue
                if keyword.lower() in existing_keywords:
                    continue
                if self._keyword_occurs_as_term(keyword, content):
                    chosen_label = valid_labels[0]
                    additions.append(
                        SemanticLabelAnnotation(
                            label=chosen_label,
                            keywords=[keyword],
                            is_new_label=False,
                            metadata={"source": "existing_label_inventory"},
                        )
                    )
            if additions:
                sem.label_annotations.extend(additions)
            sem = self._merge_semantic_label_annotations_by_label(sem)
        self._semantic_repository.overwrite_all(owner_id, semantic_memories)
        return semantic_memories

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
            return (block_id, msg_idx, getattr(sem, "created_at", datetime.max))

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
                ts = sem.timestamp or sem.created_at
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

    def _keyword_occurs_as_term(self, keyword: str, text: str) -> bool:
        if not keyword or not text:
            return False
        keyword = keyword.strip()
        text = text.strip()
        if not keyword or not text:
            return False
        pattern = "(?<!\\w)" + re.escape(keyword) + "(?!\\w)"
        return re.search(pattern, text, flags=re.IGNORECASE) is not None

    def _convert_dicts_to_messages(self, messages: List[Dict[str, Any]]) -> List[Message]:
        message_objects = []
        for msg_data in messages:
            timestamp = None
            if "timestamp" in msg_data:
                timestamp_value = msg_data["timestamp"]
                if isinstance(timestamp_value, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp_value.replace("Z", "+00:00"))
                    except ValueError:
                        timestamp = None
                elif isinstance(timestamp_value, datetime):
                    timestamp = timestamp_value
            if timestamp:
                message = Message(
                    role=msg_data["role"],
                    content=msg_data["content"],
                    timestamp=timestamp,
                    metadata=msg_data.get("metadata", {}),
                )
            else:
                message = Message(
                    role=msg_data["role"],
                    content=msg_data["content"],
                    metadata=msg_data.get("metadata", {}),
                )
            message_objects.append(message)
        return message_objects

    def _segment_and_create_episodes(
        self, owner_id: str, messages: List[Message], batch_size: int = 40
    ) -> List[Dict[str, Any]]:
        """Segment LoCoMo messages into episode-sized blocks.

        The previous implementation kept expanding the LLM segmentation window
        when the model returned no valid breakpoints.  With stricter/remote API
        models, a malformed or empty breakpoint response could therefore grow
        from one batch to the whole scenario and create one huge episode/message
        block.  This version is session-aware and has a deterministic fallback:

        1. Infer contiguous LoCoMo sessions from message metadata
           (locomo_session_key/session_key/dataset_timestamp) or timestamp.
        2. Ask the LLM for breakpoints inside each session/window.
        3. If no valid breakpoint is returned before the session/window ends,
           create that session/window as one fallback episode instead of
           expanding to the whole scenario.
        """
        created: List[Dict[str, Any]] = []
        if not messages:
            return created
        session_fallback_enabled = bool(
            getattr(self.config, "enable_session_fallback_episode_segmentation", True)
        )
        session_spans = self._infer_contiguous_session_spans(messages)
        self._debug_episode_segmentation_event(
            owner_id,
            {
                "event": "segmentation_start",
                "total_messages": len(messages),
                "batch_size": batch_size,
                "session_count": len(session_spans),
                "session_spans": [
                    {"start": s, "end": e, "count": e - s, "session_key": str(k)}
                    for s, e, k in session_spans
                ],
            },
        )
        for session_start, session_end, session_key in session_spans:
            cursor = session_start
            while cursor < session_end:
                window_start = cursor
                window_end = min(cursor + batch_size, session_end)
                current_window = messages[window_start:window_end]
                breakpoints = self._detect_episode_breakpoints(
                    current_window,
                    owner_id=owner_id,
                    absolute_start_index=window_start,
                    session_key=str(session_key),
                )
                if breakpoints:
                    episodes_data: List[Dict[str, Any]] = []
                    local_start = 0
                    for bp in breakpoints:
                        if bp < local_start or bp >= len(current_window):
                            continue
                        block = current_window[local_start : bp + 1]
                        if block:
                            episodes_data.append(
                                {
                                    "buffer_messages": block,
                                    "boundary_reason": f"Session-aware LLM episode segmentation | session={session_key} | local={local_start}-{bp}",
                                }
                            )
                        local_start = bp + 1
                    if local_start < len(current_window):
                        tail = current_window[local_start:]
                        if window_end >= session_end:
                            if tail:
                                episodes_data.append(
                                    {
                                        "buffer_messages": tail,
                                        "boundary_reason": f"Session-aware final segmented tail | session={session_key} | local={local_start}-{len(current_window) - 1}",
                                    }
                                )
                            created.extend(self._batch_create_episodes(owner_id, episodes_data))
                            cursor = window_end
                        else:
                            if episodes_data:
                                created.extend(self._batch_create_episodes(owner_id, episodes_data))
                            cursor = cursor + local_start
                    else:
                        created.extend(self._batch_create_episodes(owner_id, episodes_data))
                        cursor = window_end
                    self._debug_episode_segmentation_event(
                        owner_id,
                        {
                            "event": "llm_breakpoints_applied",
                            "session_key": str(session_key),
                            "window_absolute_start": window_start,
                            "window_absolute_end": window_end,
                            "window_message_count": len(current_window),
                            "breakpoints": breakpoints,
                            "episodes_data_count": len(episodes_data),
                        },
                    )
                    continue
                episodes_data = [
                    {
                        "buffer_messages": current_window,
                        "boundary_reason": f"Fallback unsplit session window because LLM returned no valid breakpoints | session={session_key} | absolute={window_start}-{window_end - 1}",
                    }
                ]
                created.extend(self._batch_create_episodes(owner_id, episodes_data))
                self._debug_episode_segmentation_event(
                    owner_id,
                    {
                        "event": "fallback_unsplit_session_window",
                        "session_key": str(session_key),
                        "window_absolute_start": window_start,
                        "window_absolute_end": window_end,
                        "window_message_count": len(current_window),
                        "reason": "no_valid_breakpoints",
                    },
                )
                cursor = window_end
        self._debug_episode_segmentation_event(
            owner_id,
            {
                "event": "segmentation_done",
                "created_episode_count": len(created),
                "created_message_counts": [
                    int(x.get("message_count", 0)) for x in created if isinstance(x, dict)
                ],
            },
        )
        return created

    def _infer_message_session_key(self, message: Message, index: int) -> str:
        """Infer a stable contiguous LoCoMo session/conversation key."""
        metadata = getattr(message, "metadata", {}) or {}
        for key in (
            "locomo_session_key",
            "session_key",
            "session_id",
            "conversation_id",
            "dataset_session_key",
            "dataset_timestamp",
        ):
            value = metadata.get(key)
            if value is not None and str(value).strip():
                return f"{key}:{str(value).strip()}"
        timestamp = getattr(message, "timestamp", None)
        if timestamp is not None:
            try:
                return f"timestamp:{timestamp.isoformat()}"
            except Exception:
                return f"timestamp:{str(timestamp)}"
        return f"index_chunk:{index // max(1, int(getattr(self.config, 'fallback_session_chunk_size', 20)))}"

    def _infer_contiguous_session_spans(
        self, messages: List[Message]
    ) -> List[Tuple[int, int, str]]:
        if not messages:
            return []
        spans: List[Tuple[int, int, str]] = []
        start = 0
        current_key = self._infer_message_session_key(messages[0], 0)
        for idx in range(1, len(messages)):
            key = self._infer_message_session_key(messages[idx], idx)
            if key != current_key:
                spans.append((start, idx, current_key))
                start = idx
                current_key = key
        spans.append((start, len(messages), current_key))
        return spans

    def _episode_segmentation_debug_enabled(self) -> bool:
        value = os.getenv("WM_DEBUG_SEGMENTATION", "").strip().lower()
        if value in {"1", "true", "yes", "y", "on"}:
            return True
        return bool(getattr(self.config, "debug_episode_segmentation", False))

    def _debug_episode_segmentation_event(self, owner_id: str, payload: Dict[str, Any]) -> None:
        if not self._episode_segmentation_debug_enabled():
            return
        try:
            payload = dict(payload)
            payload.setdefault("owner_id", owner_id)
            payload.setdefault("ts", datetime.now().isoformat())
            msg = json.dumps(payload, ensure_ascii=False)
            debug_dir = Path(self.config.storage_path) / "debug_segmentation"
            debug_dir.mkdir(parents=True, exist_ok=True)
            with open(
                debug_dir / f"{owner_id}_segmentation_debug.jsonl", "a", encoding="utf-8"
            ) as f:
                f.write(msg + "\n")
        except Exception as exc:
            logger.warning(f"Failed to write segmentation debug event for {owner_id}: {exc}")

    def _parse_episode_segmentation_response(self, raw_response: Any) -> Dict[str, Any]:
        """Robustly parse breakpoint JSON from either dict/list/text output."""
        if isinstance(raw_response, dict):
            return raw_response
        if isinstance(raw_response, list):
            return {"breakpoints": raw_response, "reasoning": "raw list response"}
        text = str(raw_response or "").strip()
        if not text:
            return {"breakpoints": [], "reasoning": "empty response"}
        text = re.sub("<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        if text.startswith("```"):
            text = re.sub("^```(?:json)?\\s*", "", text, flags=re.IGNORECASE)
            text = re.sub("\\s*```$", "", text).strip()
        candidates = [text]
        for start_char, end_char in (("{", "}"), ("[", "]")):
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end != -1 and (end > start):
                candidates.append(text[start : end + 1])
            elif start != -1 and end == -1:
                candidates.append(text[start:] + end_char)
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
                if isinstance(parsed, list):
                    return {"breakpoints": parsed, "reasoning": "parsed list response"}
            except Exception:
                continue
        return {"breakpoints": [], "reasoning": "failed_to_parse_segmentation_response"}

    def _detect_episode_breakpoints(
        self,
        messages: List[Message],
        owner_id: Optional[str] = None,
        absolute_start_index: int = 0,
        session_key: Optional[str] = None,
    ) -> List[int]:
        numbered_lines = []
        for idx, msg in enumerate(messages):
            numbered_lines.append(f"{idx}. [{msg.timestamp.isoformat()}] {msg.role}: {msg.content}")
        conversation = "\n".join(numbered_lines)
        prompt = self.prompts.get_episode_segmentation_prompt(conversation)
        raw_response: Any = None
        response: Dict[str, Any]
        use_raw_debug_call = bool(
            self._episode_segmentation_debug_enabled()
            and hasattr(self.llm_client, "generate_text_response")
        )
        try:
            if use_raw_debug_call:
                raw_response = self.llm_client.generate_text_response(
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=1024,
                    api_reasoning_effort="minimal",
                    api_min_completion_tokens=4096,
                )
                response = self._parse_episode_segmentation_response(raw_response)
            else:
                response = self.llm_client.generate_json_response(
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=1024,
                    default_response={"breakpoints": [], "reasoning": ""},
                    api_reasoning_effort="minimal",
                    api_min_completion_tokens=4096,
                )
                response = self._parse_episode_segmentation_response(response)
        except Exception as exc:
            response = {"breakpoints": [], "reasoning": f"exception: {type(exc).__name__}: {exc}"}
        breakpoints = response.get("breakpoints", []) if isinstance(response, dict) else []
        if not isinstance(breakpoints, list):
            breakpoints = []
        cleaned = []
        invalid = []
        for x in breakpoints:
            try:
                v = int(x)
                if 0 <= v < len(messages):
                    cleaned.append(v)
                else:
                    invalid.append(x)
            except Exception:
                invalid.append(x)
        cleaned = sorted(set(cleaned))
        if owner_id is not None:
            self._debug_episode_segmentation_event(
                owner_id,
                {
                    "event": "detect_episode_breakpoints",
                    "session_key": session_key,
                    "absolute_start_index": absolute_start_index,
                    "message_count": len(messages),
                    "raw_response": raw_response if raw_response is not None else None,
                    "parsed_response": response,
                    "cleaned_breakpoints": cleaned,
                    "invalid_breakpoints": invalid,
                    "conversation_preview": conversation[:2000],
                },
            )
        return cleaned

    def _batch_create_episodes(
        self, owner_id: str, episodes_data: List[Dict]
    ) -> List[Dict[str, Any]]:
        if not episodes_data:
            return []
        created_episodes = []
        if len(episodes_data) == 1:
            episode_data = episodes_data[0]
            episode_info = self._create_episode_from_messages(
                owner_id, episode_data["buffer_messages"], episode_data["boundary_reason"]
            )
            if episode_info:
                created_episodes.append(episode_info)
        else:
            with ThreadPoolExecutor(
                max_workers=min(len(episodes_data), self.config.max_workers)
            ) as executor:
                future_to_data = {
                    executor.submit(
                        self._create_episode_from_messages,
                        owner_id,
                        ep_data["buffer_messages"],
                        ep_data["boundary_reason"],
                    ): ep_data
                    for ep_data in episodes_data
                }
                for future in as_completed(future_to_data):
                    try:
                        episode_info = future.result()
                        if episode_info:
                            created_episodes.append(episode_info)
                    except Exception as e:
                        logger.error(f"Error creating episode in batch: {e}")
                        raise
        created_episodes.sort(
            key=lambda x: (
                x["episode_object"].time_range_start.isoformat()
                if x.get("episode_object") and x["episode_object"].time_range_start
                else ""
            )
        )
        return created_episodes

    def _create_episode_from_messages(self, owner_id, messages, boundary_reason):
        if not messages:
            return None
        with self._api_llm_scope(reasoning_effort="minimal", min_completion_tokens=4096):
            episode = self.episode_generator.generate_episode(owner_id, messages, boundary_reason)
        episode.time_range_start = messages[0].timestamp
        episode.time_range_end = messages[-1].timestamp
        self._episode_repository.save(episode)
        return dict(
            episode_id=episode.episode_id,
            title=episode.title,
            message_count=episode.message_count,
            boundary_reason=boundary_reason,
            episode_object=episode,
        )

    def _async_generate_semantic_memories(self, owner_id, new_episode):
        episodes = self._episode_repository.list_by_user(owner_id)
        existing = self._semantic_repository.list_by_user(owner_id)
        with self._api_llm_scope(reasoning_effort="medium", min_completion_tokens=16384):
            generated = self.semantic_generator.check_and_generate_semantic_memories(
                user_id=owner_id,
                new_episode=new_episode,
                existing_episodes=[e for e in episodes if e.episode_id != new_episode.episode_id],
                existing_semantic_memories=existing,
            )
        message_embeddings = [
            (msg, self.embedding_client.embed_text(msg.content))
            for msg in new_episode.get_messages()
        ]
        saved = []
        embeddings = {}
        for memory in generated:
            vector = self.embedding_client.embed_text(memory.content)
            if self._is_duplicate_semantic_memory(
                owner_id,
                memory,
                existing_memories=existing,
                embedding_cache=embeddings,
                new_embedding=vector,
            ):
                continue
            memory.timestamp = self._align_semantic_timestamp(memory.content, message_embeddings)
            self._semantic_repository.save(memory)
            embeddings[memory.memory_id] = vector
            existing.append(memory)
            saved.append(memory)
        return saved

    def _align_semantic_timestamp(
        self, semantic_text: str, message_embeddings: List[Tuple[Message, List[float]]]
    ) -> Optional[datetime]:
        if not message_embeddings:
            return None
        try:
            sem_emb = self.embedding_client.embed_text(semantic_text)
            best_msg = None
            best_score = -1.0
            for msg, emb in message_embeddings:
                try:
                    score = self.embedding_client.cosine_similarity(sem_emb, emb)
                except Exception:
                    score = 0.0
                if score > best_score:
                    best_score = score
                    best_msg = msg
            return best_msg.timestamp if best_msg else None
        except Exception:
            return None

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
        }
        return alias_map.get(label.lower(), label)

    def _tokenize_label_for_overlap(self, label: str) -> List[str]:
        if not label:
            return []
        label = self._normalize_label(label)
        text = label.lower()
        text = text.replace("&", " and ")
        parts = [x.strip() for x in text.split() if x.strip()]
        stopwords = {"and", "of", "the", "a", "an"}
        tokens = [x for x in parts if x not in stopwords]
        return tokens

    def _find_word_overlap_merge_target(
        self, proposed_label: str, label_inventory: List[str]
    ) -> str:
        proposed_norm = self._normalize_label(proposed_label)
        if not proposed_norm:
            return ""
        proposed_tokens = set(self._tokenize_label_for_overlap(proposed_norm))
        if not proposed_tokens:
            return ""
        for existing_label in label_inventory:
            existing_norm = self._normalize_label(existing_label)
            if not existing_norm:
                continue
            existing_tokens = set(self._tokenize_label_for_overlap(existing_norm))
            if not existing_tokens:
                continue
            if proposed_tokens.issubset(existing_tokens) or existing_tokens.issubset(
                proposed_tokens
            ):
                return existing_norm
        return ""

    def _update_label_inventory(
        self, current_inventory: List[str], proposed_new_labels: List[str]
    ) -> List[str]:
        normalized_existing = {}
        for label in current_inventory:
            norm = self._normalize_label(label)
            if norm:
                normalized_existing[norm.lower()] = norm
        updated = list(normalized_existing.values())
        for label in proposed_new_labels:
            norm = self._normalize_label(label)
            if not norm:
                continue
            if norm.lower() in normalized_existing:
                continue
            updated.append(norm)
            normalized_existing[norm.lower()] = norm
        return updated

    def _find_most_similar_existing_label(
        self, proposed_label: str, label_inventory: List[str]
    ) -> Dict[str, Any]:
        norm_label = self._normalize_label(proposed_label)
        if not norm_label:
            return {"label": "", "similarity": 0.0}
        if not label_inventory:
            return {"label": "", "similarity": 0.0}
        try:
            proposed_emb = self.embedding_client.embed_text(norm_label)
        except Exception:
            return {"label": label_inventory[0] if label_inventory else "", "similarity": 0.0}
        best_label = ""
        best_sim = -1.0
        for existing_label in label_inventory:
            existing_norm = self._normalize_label(existing_label)
            if not existing_norm:
                continue
            try:
                existing_emb = self.embedding_client.embed_text(existing_norm)
                sim = self.embedding_client.cosine_similarity(proposed_emb, existing_emb)
            except Exception:
                sim = 0.0
            if sim > best_sim:
                best_sim = sim
                best_label = existing_norm
        return {"label": best_label, "similarity": float(best_sim if best_sim >= 0 else 0.0)}

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
            with self._api_llm_scope(reasoning_effort="minimal", min_completion_tokens=4096):
                resp = self.llm_client.generate_json_response(
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=256,
                    default_response={"merge_to_existing": True, "reason": ""},
                )
            return {
                "merge_to_existing": bool(resp.get("merge_to_existing", True)),
                "reason": str(resp.get("reason", "")).strip(),
            }
        except Exception:
            return {"merge_to_existing": True, "reason": "fallback_merge_on_exception"}

    def _merge_keywords_into_label_bucket(
        self, bucket: Dict[str, List[str]], label: str, keywords: List[str]
    ) -> None:
        label = self._normalize_label(label)
        if not label:
            return
        if label not in bucket:
            bucket[label] = []
        existing_lower = {x.strip().lower() for x in bucket[label] if isinstance(x, str)}
        for kw in keywords:
            if not isinstance(kw, str):
                continue
            kw = kw.strip()
            if not kw:
                continue
            if kw.lower() in existing_lower:
                continue
            bucket[label].append(kw)
            existing_lower.add(kw.lower())

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
            with self._api_llm_scope(reasoning_effort="minimal", min_completion_tokens=4096):
                resp = self.llm_client.generate_json_response(
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=1024,
                    default_response={"existing_labels": {}, "new_labels": {}},
                )
            return resp
        except Exception:
            return {"existing_labels": {}, "new_labels": {}}

    def _annotate_semantics_with_labels(self, owner_id: str) -> List[SemanticMemory]:
        semantic_memories = self._semantic_repository.list_by_user(owner_id)
        if not semantic_memories:
            return []
        label_inventory = list(self.default_labels)
        for sem in semantic_memories:
            if getattr(sem, "label_annotations", None):
                continue
            resp = self._extract_label_annotations_for_semantic(
                semantic=sem, label_inventory=label_inventory
            )
            existing_labels_raw = resp.get("existing_labels", {})
            new_labels_raw = resp.get("new_labels", {})
            if not isinstance(existing_labels_raw, dict):
                existing_labels_raw = {}
            if not isinstance(new_labels_raw, dict):
                new_labels_raw = {}
            existing_inventory_before_update = {self._normalize_label(x) for x in label_inventory}
            merged_existing_labels: Dict[str, List[str]] = {}
            for label, keywords in existing_labels_raw.items():
                norm_label = self._normalize_label(label)
                if not norm_label:
                    continue
                if not isinstance(keywords, list):
                    keywords = []
                clean_keywords = []
                seen_kw = set()
                for kw in keywords:
                    if not isinstance(kw, str):
                        continue
                    kw = kw.strip()
                    if not kw:
                        continue
                    if kw.lower() in seen_kw:
                        continue
                    seen_kw.add(kw.lower())
                    clean_keywords.append(kw)
                self._merge_keywords_into_label_bucket(
                    merged_existing_labels, norm_label, clean_keywords
                )
            accepted_new_labels: Dict[str, List[str]] = {}
            for proposed_label, keywords in new_labels_raw.items():
                norm_label = self._normalize_label(proposed_label)
                if not norm_label:
                    continue
                if not isinstance(keywords, list):
                    keywords = []
                clean_keywords = []
                seen_kw = set()
                for kw in keywords:
                    if not isinstance(kw, str):
                        continue
                    kw = kw.strip()
                    if not kw:
                        continue
                    if kw.lower() in seen_kw:
                        continue
                    seen_kw.add(kw.lower())
                    clean_keywords.append(kw)
                word_overlap_target = self._find_word_overlap_merge_target(
                    proposed_label=norm_label, label_inventory=label_inventory
                )
                if word_overlap_target:
                    self._merge_keywords_into_label_bucket(
                        merged_existing_labels, word_overlap_target, clean_keywords
                    )
                    continue
                similarity_info = self._find_most_similar_existing_label(
                    proposed_label=norm_label, label_inventory=label_inventory
                )
                candidate_existing_label = similarity_info.get("label", "")
                should_merge = False
                if candidate_existing_label:
                    judge_result = self._judge_whether_new_label_can_merge_into_existing(
                        semantic_content=sem.content,
                        new_label=norm_label,
                        new_label_keywords=clean_keywords,
                        candidate_existing_label=candidate_existing_label,
                    )
                    should_merge = bool(judge_result.get("merge_to_existing", False))
                if should_merge and candidate_existing_label:
                    self._merge_keywords_into_label_bucket(
                        merged_existing_labels, candidate_existing_label, clean_keywords
                    )
                else:
                    accepted_new_labels[norm_label] = clean_keywords
            proposed_new_labels = list(accepted_new_labels.keys())
            label_inventory = self._update_label_inventory(label_inventory, proposed_new_labels)
            annotations: List[SemanticLabelAnnotation] = []
            seen_labels = set()
            for label, keywords in merged_existing_labels.items():
                norm_label = self._normalize_label(label)
                if not norm_label:
                    continue
                if norm_label in seen_labels:
                    continue
                seen_labels.add(norm_label)
                annotations.append(
                    SemanticLabelAnnotation(
                        label=norm_label,
                        keywords=keywords,
                        is_new_label=False,
                        metadata={"source": "existing_label_inventory"},
                    )
                )
            for label, keywords in accepted_new_labels.items():
                norm_label = self._normalize_label(label)
                if not norm_label:
                    continue
                if norm_label in seen_labels:
                    continue
                seen_labels.add(norm_label)
                is_new = norm_label not in existing_inventory_before_update
                annotations.append(
                    SemanticLabelAnnotation(
                        label=norm_label,
                        keywords=keywords,
                        is_new_label=is_new,
                        metadata={"source": "new_label_proposal"},
                    )
                )
            sem.label_annotations = annotations
        self._semantic_repository.overwrite_all(owner_id, semantic_memories)
        return semantic_memories

    def _find_top_k_similar_existing_labels(
        self, target_label: str, existing_labels: List[str], top_k: int = 8
    ) -> List[str]:
        norm_target = self._normalize_label(target_label)
        if not norm_target or not existing_labels:
            return []
        try:
            target_emb = self.embedding_client.embed_text(norm_target)
        except Exception:
            return existing_labels[:top_k]
        scored = []
        for label in existing_labels:
            norm_label = self._normalize_label(label)
            if not norm_label:
                continue
            try:
                emb = self.embedding_client.embed_text(norm_label)
                sim = self.embedding_client.cosine_similarity(target_emb, emb)
            except Exception:
                sim = 0.0
            scored.append((sim, norm_label))
        scored.sort(key=lambda x: x[0], reverse=True)
        result = []
        seen = set()
        for _, label in scored:
            if label in seen:
                continue
            seen.add(label)
            result.append(label)
            if len(result) >= top_k:
                break
        return result

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
            with self._api_llm_scope(reasoning_effort="minimal", min_completion_tokens=4096):
                resp = self.llm_client.generate_json_response(
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=256,
                    default_response={"merge": False, "target_label": "", "reason": ""},
                )
            return {
                "merge": bool(resp.get("merge", False)),
                "target_label": self._normalize_label(resp.get("target_label", "")),
                "reason": str(resp.get("reason", "")).strip(),
            }
        except Exception:
            return {"merge": False, "target_label": "", "reason": "fallback_keep_on_exception"}

    def _finalize_sparse_new_labels(
        self, owner_id: str, semantic_memories: List[SemanticMemory]
    ) -> List[SemanticMemory]:
        if not semantic_memories:
            return semantic_memories
        label_to_semantic_ids: Dict[str, set] = defaultdict(set)
        label_to_keywords: Dict[str, List[str]] = defaultdict(list)
        label_is_new: Dict[str, bool] = {}
        existing_labels = set()
        for sem in semantic_memories:
            for ann in getattr(sem, "label_annotations", []) or []:
                norm_label = self._normalize_label(ann.label)
                if not norm_label:
                    continue
                label_to_semantic_ids[norm_label].add(sem.memory_id)
                label_is_new[norm_label] = label_is_new.get(norm_label, False) or bool(
                    ann.is_new_label
                )
                if ann.is_new_label:
                    existing_lower = {x.lower() for x in label_to_keywords[norm_label]}
                    for kw in ann.keywords:
                        if not isinstance(kw, str):
                            continue
                        kw = kw.strip()
                        if not kw:
                            continue
                        if kw.lower() in existing_lower:
                            continue
                        label_to_keywords[norm_label].append(kw)
                        existing_lower.add(kw.lower())
                else:
                    existing_labels.add(norm_label)
        sparse_new_labels = []
        for label, is_new in label_is_new.items():
            if not is_new:
                continue
            semantic_count = len(label_to_semantic_ids.get(label, set()))
            if semantic_count < 3:
                sparse_new_labels.append(label)
        if not sparse_new_labels:
            return semantic_memories
        for new_label in sparse_new_labels:
            candidate_existing_labels = self._find_top_k_similar_existing_labels(
                target_label=new_label, existing_labels=sorted(list(existing_labels)), top_k=8
            )
            if not candidate_existing_labels:
                continue
            judge_result = self._judge_sparse_new_label_final_merge(
                target_label=new_label,
                collected_keywords=label_to_keywords.get(new_label, []),
                candidate_existing_labels=candidate_existing_labels,
            )
            should_merge = bool(judge_result.get("merge", False))
            merge_target = self._normalize_label(judge_result.get("target_label", ""))
            if not should_merge or not merge_target:
                continue
            if merge_target not in existing_labels:
                continue
            for sem in semantic_memories:
                anns = getattr(sem, "label_annotations", []) or []
                new_label_keywords_for_this_sem = []
                kept_annotations = []
                target_annotation = None
                for ann in anns:
                    ann_label = self._normalize_label(ann.label)
                    if ann_label == new_label and ann.is_new_label:
                        for kw in ann.keywords:
                            if isinstance(kw, str) and kw.strip():
                                new_label_keywords_for_this_sem.append(kw.strip())
                        continue
                    if ann_label == merge_target:
                        target_annotation = ann
                        continue
                    kept_annotations.append(ann)
                if not new_label_keywords_for_this_sem and target_annotation is None:
                    sem.label_annotations = kept_annotations + (
                        [target_annotation] if target_annotation else []
                    )
                    continue
                if target_annotation is None and new_label_keywords_for_this_sem:
                    target_annotation = SemanticLabelAnnotation(
                        label=merge_target,
                        keywords=[],
                        is_new_label=False,
                        metadata={"source": "existing_label_inventory"},
                    )
                if target_annotation is not None:
                    existing_kw_lower = {
                        x.strip().lower() for x in target_annotation.keywords if isinstance(x, str)
                    }
                    for kw in new_label_keywords_for_this_sem:
                        if kw.lower() in existing_kw_lower:
                            continue
                        target_annotation.keywords.append(kw)
                        existing_kw_lower.add(kw.lower())
                    target_annotation.is_new_label = False
                final_anns = kept_annotations + ([target_annotation] if target_annotation else [])
                sem.label_annotations = final_anns
        self._semantic_repository.overwrite_all(owner_id, semantic_memories)
        return semantic_memories

    def _build_semantic_to_situation_ids(
        self, situations: List[SituationMemory]
    ) -> Dict[str, List[str]]:
        mapping = defaultdict(set)
        for situation in situations:
            for agg in situation.aggregates:
                for semantic_id in agg.semantic_ids:
                    mapping[semantic_id].add(situation.situation_id)
        return {
            semantic_id: sorted(list(situation_ids))
            for semantic_id, situation_ids in mapping.items()
        }

    def _build_mental_memory(
        self, owner_id: str, situations: List[SituationMemory], label_cluster_threshold: float = 0.8
    ) -> Optional[List[MentalMemory]]:
        if not situations:
            self._mental_storage.overwrite_all(owner_id, [])
            return None
        semantic_memories = self._annotate_semantics_with_labels(owner_id=owner_id)
        if not semantic_memories:
            self._mental_storage.overwrite_all(owner_id, [])
            return None
        semantic_memories = self._finalize_sparse_new_labels(
            owner_id=owner_id, semantic_memories=semantic_memories
        )
        semantic_to_situation_ids = self._build_semantic_to_situation_ids(situations)
        label_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for sem in semantic_memories:
            timestamps = []
            if getattr(sem, "timestamp", None):
                if isinstance(sem.timestamp, datetime):
                    timestamps.append(sem.timestamp.isoformat())
                else:
                    timestamps.append(str(sem.timestamp))
            elif getattr(sem, "created_at", None):
                if isinstance(sem.created_at, datetime):
                    timestamps.append(sem.created_at.isoformat())
                else:
                    timestamps.append(str(sem.created_at))
            for ann in getattr(sem, "label_annotations", []) or []:
                label = self._normalize_label(ann.label)
                if not label:
                    continue
                label_buckets[label].append(
                    {
                        "semantic_id": sem.memory_id,
                        "semantic_content": sem.content,
                        "keywords": ann.keywords or [],
                        "situation_ids": semantic_to_situation_ids.get(sem.memory_id, []),
                        "timestamps": timestamps,
                        "is_new_label": ann.is_new_label,
                    }
                )
        mental_unit_records: List[MentalMemory] = []
        for label in sorted(label_buckets.keys()):
            items = label_buckets[label]
            unit_is_new_label = any((bool(item.get("is_new_label", False)) for item in items))
            semantic_ids = []
            all_situation_ids = set()
            all_timestamps = []
            semantic_links: List[MentalSemanticLink] = []
            for item in items:
                semantic_ids.append(item["semantic_id"])
                all_situation_ids.update(item["situation_ids"])
                all_timestamps.extend(item["timestamps"])
                semantic_links.append(
                    MentalSemanticLink(
                        semantic_id=item["semantic_id"],
                        semantic_content=item["semantic_content"],
                        keywords=item["keywords"],
                        situation_ids=item["situation_ids"],
                        timestamps=item["timestamps"],
                        metadata={"is_new_label": item["is_new_label"]},
                    )
                )
            unit = MentalCluster(
                cluster_summary=label,
                detailed_summary=f"All semantic memories annotated with label '{label}'",
                label=label,
                is_new_label=unit_is_new_label,
                situation_ids=sorted(list(all_situation_ids)),
                semantic_ids=semantic_ids,
                semantic_links=semantic_links,
                timestamps=all_timestamps,
                metadata={"unit_type": "label_bucket", "semantic_count": len(semantic_ids)},
            )
            mental_record = MentalMemory(
                user_id=owner_id,
                content=label,
                clusters=[unit],
                metadata={"label": label, "semantic_count": len(semantic_ids)},
            )
            mental_unit_records.append(mental_record)
        self._mental_storage.overwrite_all(owner_id, mental_unit_records)
        return mental_unit_records

    def _is_duplicate_semantic_memory(
        self,
        owner_id: str,
        memory: SemanticMemory,
        existing_memories: Optional[List[SemanticMemory]] = None,
        embedding_cache: Optional[Dict[str, List[float]]] = None,
        new_embedding: Optional[List[float]] = None,
    ) -> bool:
        try:
            return self._is_duplicate_semantic_memory_fallback(
                owner_id,
                memory,
                existing_memories=existing_memories,
                embedding_cache=embedding_cache,
                new_embedding=new_embedding,
            )
        except Exception as e:
            logger.error(f"Error checking semantic duplication: {e}")
            return False

    def _is_duplicate_semantic_memory_fallback(
        self,
        owner_id: str,
        memory: SemanticMemory,
        existing_memories: Optional[List[SemanticMemory]] = None,
        embedding_cache: Optional[Dict[str, List[float]]] = None,
        new_embedding: Optional[List[float]] = None,
    ) -> bool:
        try:
            if existing_memories is None:
                existing_memories = self._semantic_repository.list_by_user(owner_id)
            if not existing_memories:
                return False
            embeddings_lookup = embedding_cache if embedding_cache is not None else {}
            if new_embedding is not None:
                memory_embedding = new_embedding
            else:
                memory_embedding = self.embedding_client.embed_text(memory.content)
            for existing in existing_memories:
                if existing.knowledge_type == memory.knowledge_type:
                    existing_embedding = embeddings_lookup.get(existing.memory_id)
                    if existing_embedding is None:
                        existing_embedding = self.embedding_client.embed_text(existing.content)
                        if embedding_cache is not None:
                            embeddings_lookup[existing.memory_id] = existing_embedding
                    similarity = self.embedding_client.cosine_similarity(
                        memory_embedding, existing_embedding
                    )
                    if similarity > self.config.semantic_similarity_threshold:
                        return True
            return False
        except Exception as e:
            logger.error(f"Error checking semantic memory duplication: {e}")
            return False
