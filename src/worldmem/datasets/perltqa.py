"""
PerLTQA dataset adapter for WorldMem.

Place this file under `src/core/perltqa_adapter.py` and use it from the
PerLTQA runner.  The adapter builds the same five JSONL layers used by the
LoCoMo world-memory pipeline:

1. message_blocks
2. episodes
3. semantic
4. situation
5. mental

It is intentionally implemented as an adapter around `MemorySystem` rather
than replacing the existing LoCoMo `build_world_memory_from_messages` entry.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from worldmem.construction.builder import DEFAULT_SEMANTIC_LABELS
from worldmem.models import Episode, MessageBlock, SemanticLabelAnnotation, SemanticMemory

logger = logging.getLogger(__name__)
GPT5_PERLTQA_COMPLETION_TOKENS = 32768
PERLTQA_EXTRA_LABELS = ["Appearance", "Achievement", "Awards & Role Models"]
PERLTQA_LABEL_ALIASES = {
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
PROFILE_FIELD_LABELS: Dict[str, List[str]] = {
    "Protagonist": ["Character"],
    "Gender": ["Identity"],
    "Nickname": ["Identity"],
    "Title": ["Career", "Identity"],
    "Age": ["Identity"],
    "Occupation": ["Career"],
    "Nationality": ["Identity"],
    "Physical Characteristics": ["Appearance"],
    "Hobbies": ["Preferences", "Habits & Routines"],
    "Achievements": ["Achievement"],
    "Ethnic Background": ["Identity"],
    "Education Background": ["Education"],
    "Employer": ["Career", "Organization"],
    "Awards and Role Models": ["Awards & Role Models"],
    "Awards & Role Models": ["Awards & Role Models"],
}
SOURCE_PROFILE = "profile"
SOURCE_SOCIAL = "social_relationship"
SOURCE_EVENTS = "events"


def _normalize_perltqa_label(memory_system: Any, label: str) -> str:
    if not label:
        return ""
    if hasattr(memory_system, "_normalize_label"):
        label = memory_system._normalize_label(label)
    label = " ".join(str(label).strip().split())
    return PERLTQA_LABEL_ALIASES.get(label.lower(), label)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _is_non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and (not value.strip()):
        return False
    if isinstance(value, (list, tuple, dict)) and (not value):
        return False
    return True


def _safe_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _naive_datetime(dt: Optional[datetime]) -> Optional[datetime]:
    """Normalize PerLTQA timestamps to timezone-naive datetimes.

    The dataset has one logical timestamp for events/dialogues, not separate
    start/end instants.  Some strings may parse as offset-aware (for example
    with Z or +00:00), while generated/profile records may be naive.  Keep the
    same visible wall-clock date/time and strip tzinfo so all later comparisons
    are safe and deterministic.
    """
    if dt is None:
        return None
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


def _parse_datetime(value: Any) -> Optional[datetime]:
    """Parse common PerLTQA date/time formats without raising."""
    if isinstance(value, datetime):
        return _naive_datetime(value)
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        return _naive_datetime(datetime.fromisoformat(normalized))
    except Exception:
        pass
    formats = [
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%B %d, %Y",
        "%b %d, %Y",
        "%Y-%m-%d",
        "%Y/%m/%d",
    ]
    for fmt in formats:
        try:
            return _naive_datetime(datetime.strptime(text, fmt))
        except Exception:
            continue
    return None


def _message_timestamp(value: Any) -> str:
    """Return ISO timestamp when available; otherwise keep the requested empty placeholder."""
    dt = _parse_datetime(value)
    return dt.isoformat() if dt else ""


def _timestamp_range_from_messages(
    messages: Sequence[Dict[str, Any]],
) -> Tuple[Optional[datetime], Optional[datetime]]:
    timestamps: List[datetime] = []
    for msg in messages:
        ts = _parse_datetime(msg.get("timestamp"))
        if ts is not None:
            timestamps.append(ts)
    if not timestamps:
        return (None, None)
    return (min(timestamps), max(timestamps))


def _split_profile_description(text: str) -> List[str]:
    """
    Split `profile_description` into sentence-level source messages.

    The split keeps markdown headings such as `**Protagonist: Wang Xiaoming**`
    as independent source messages, then sentence-splits normal paragraphs.
    """
    text = (text or "").strip()
    if not text:
        return []
    units: List[str] = []
    paragraphs = [p.strip() for p in re.split("\\n+", text) if p.strip()]
    for paragraph in paragraphs:
        if paragraph.startswith("**") and paragraph.endswith("**") and (len(paragraph) < 160):
            units.append(paragraph)
            continue
        pieces = re.split("(?<=[.!?])\\s+", paragraph)
        for piece in pieces:
            piece = piece.strip()
            if piece:
                units.append(piece)
    return units or [text]


def _make_message(
    *,
    role: str,
    content: str,
    source: str,
    character_id: str,
    character_name: str,
    timestamp: Any = "",
    original_speaker: Optional[str] = None,
    index: int = 0,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metadata = {
        "source": source,
        "character_id": character_id,
        "character_name": character_name,
        "original_speaker": original_speaker or role,
        "dataset_timestamp": str(timestamp or ""),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return {
        "message_id": f"perltqa_{character_id}_{source}_{index}_{uuid.uuid4().hex[:8]}",
        "role": role or character_name or character_id,
        "content": content,
        "timestamp": _message_timestamp(timestamp),
        "metadata": metadata,
    }


def _make_block(
    *,
    owner_id: str,
    episode_id: str,
    messages: Sequence[Dict[str, Any]],
    source: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> MessageBlock:
    source_message_ids = [m.get("message_id", "") for m in messages]
    source_message_indices = list(range(len(messages)))
    timestamp_start, timestamp_end = _timestamp_range_from_messages(messages)
    content = "\n".join(
        (
            f"[ID:{idx}] [{m.get('timestamp', '')}] {m.get('role', '')}: {m.get('content', '')}"
            for idx, m in enumerate(messages)
        )
    )
    block_metadata = {"source": source, "episode_id": episode_id}
    if metadata:
        block_metadata.update(metadata)
    return MessageBlock(
        user_id=owner_id,
        episode_id=episode_id,
        source_message_ids=source_message_ids,
        source_message_indices=source_message_indices,
        messages=list(messages),
        content=content,
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
        metadata=block_metadata,
    )


def _make_episode(
    *,
    owner_id: str,
    title: str,
    content: str,
    messages: Sequence[Dict[str, Any]],
    source: str,
    timestamp: Any = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Episode:
    timestamp_start, timestamp_end = _timestamp_range_from_messages(messages)
    ep_timestamp = _parse_datetime(timestamp) or timestamp_start or datetime.now()
    ep_metadata = {"source": source}
    if metadata:
        ep_metadata.update(metadata)
    return Episode(
        user_id=owner_id,
        title=title or source,
        content=content,
        original_messages=list(messages),
        message_count=len(messages),
        boundary_reason=f"PerLTQA pre-segmented source: {source}",
        timestamp=ep_timestamp,
        time_range_start=timestamp_start,
        time_range_end=timestamp_end,
        metadata=ep_metadata,
        tags=["perltqa", source],
    )


def _dedupe_strings(values: Iterable[Any]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        item = value.strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _make_annotations(
    label_to_keywords: Dict[str, Sequence[str]],
    *,
    source: str,
    memory_system: Any,
    is_new_label: bool = False,
) -> List[SemanticLabelAnnotation]:
    annotations: List[SemanticLabelAnnotation] = []
    for label, keywords in label_to_keywords.items():
        norm_label = _normalize_perltqa_label(memory_system, label)
        clean_keywords = _dedupe_strings(keywords)
        if not norm_label or not clean_keywords:
            continue
        annotations.append(
            SemanticLabelAnnotation(
                label=norm_label,
                keywords=clean_keywords,
                is_new_label=is_new_label,
                metadata={"source": source},
            )
        )
    return annotations


def _merge_extra_annotations(
    memory_system: Any, semantic: SemanticMemory, annotations: List[SemanticLabelAnnotation]
) -> SemanticMemory:
    if annotations:
        semantic.label_annotations.extend(annotations)
    if hasattr(memory_system, "_merge_semantic_label_annotations_by_label"):
        semantic = memory_system._merge_semantic_label_annotations_by_label(semantic)
    return semantic


def _json_list_response(resp: Any) -> List[Dict[str, Any]]:
    if isinstance(resp, dict):
        if isinstance(resp.get("memories"), list):
            resp = resp["memories"]
        elif isinstance(resp.get("items"), list):
            resp = resp["items"]
        else:
            resp = [resp]
    if not isinstance(resp, list):
        return []
    return [x for x in resp if isinstance(x, dict)]


def _api_llm_scope(
    memory_system: Any,
    *,
    reasoning_effort: str = "minimal",
    min_completion_tokens: int = GPT5_PERLTQA_COMPLETION_TOKENS,
):
    if hasattr(memory_system, "_api_llm_scope"):
        return memory_system._api_llm_scope(
            reasoning_effort=reasoning_effort, min_completion_tokens=min_completion_tokens
        )
    client = getattr(memory_system, "llm_client", None)
    if client is not None and hasattr(client, "api_call_scope"):
        return client.api_call_scope(
            reasoning_effort=reasoning_effort, min_completion_tokens=min_completion_tokens
        )
    return contextlib.nullcontext()


def _extract_event_semantics_from_episode(
    memory_system: Any,
    *,
    owner_id: str,
    episode: Episode,
    block: MessageBlock,
    event_record: Dict[str, Any],
) -> List[SemanticMemory]:
    """Thinking-mode semantic extraction from the event-level episode memory."""
    prompt = """
You are extracting semantic memories for a PerLTQA personal long-term memory dataset.

[Character Memory Owner]
{0}

[Event Record]
Summary: {1}
Theme: {2}
Creation Time: {3}
Content:
{4}

[Grounding Dialogue Block]
{5}

Task:
Extract all valuable persistent semantic memories from the event record.

Requirements:
1. Extract atomic but self-contained facts. Split unrelated facts into separate semantic memories.
2. Every semantic memory must be explicit about who did/experienced/liked/planned what. Avoid pronouns.
3. Preserve exact names, dates, locations, organizations, objects, preferences, relationships, goals, and event details.
4. Use the dialogue block only for grounding and source alignment. If a semantic is clearly supported by one or more dialogue lines, include their integer IDs from [ID:n].
5. Skip greetings and generic social gestures.
6. Return JSON only. Do not include explanations.

Output format:
[
  {{
    \"content\": \"self-contained semantic memory\",
    \"knowledge_type\": \"knowledge\",
    \"source_message_indices\": [0, 1]
  }}
]
""".format(
        owner_id,
        event_record.get("summary", ""),
        event_record.get("Theme", ""),
        event_record.get("Creation Time", ""),
        event_record.get("content", episode.content),
        block.content,
    )
    try:
        with _api_llm_scope(
            memory_system,
            reasoning_effort="medium",
            min_completion_tokens=GPT5_PERLTQA_COMPLETION_TOKENS,
        ):
            resp = memory_system.llm_client.generate_json_response(
                prompt=prompt,
                temperature=0.0,
                max_tokens=GPT5_PERLTQA_COMPLETION_TOKENS,
                default_response=[],
                enable_thinking=True,
            )
    except TypeError:
        with _api_llm_scope(
            memory_system,
            reasoning_effort="medium",
            min_completion_tokens=GPT5_PERLTQA_COMPLETION_TOKENS,
        ):
            resp = memory_system.llm_client.generate_json_response(
                prompt=prompt,
                temperature=0.0,
                max_tokens=GPT5_PERLTQA_COMPLETION_TOKENS,
                default_response=[],
            )
    except Exception as exc:
        logger.warning(
            "PerLTQA event semantic extraction failed for episode %s: %s", episode.episode_id, exc
        )
        return []
    semantics: List[SemanticMemory] = []
    for item in _json_list_response(resp):
        content = _clean_text(item.get("content") or item.get("memory") or item.get("semantic"))
        if not content:
            continue
        idxs: List[int] = []
        for raw_idx in item.get("source_message_indices", []) or []:
            try:
                idx = int(raw_idx)
            except Exception:
                continue
            if 0 <= idx < len(block.messages):
                idxs.append(idx)
        idxs = sorted(set(idxs))
        if idxs:
            matched_idx = idxs[0]
            matched_msg = block.messages[matched_idx]
            matched_message_id = matched_msg.get("message_id")
            matched_message_content = matched_msg.get("content")
            source_message_ids = [block.messages[i].get("message_id", "") for i in idxs]
        else:
            matched_message_id, matched_idx, matched_message_content = (
                memory_system._pick_best_message_for_semantic_text(
                    semantic_text=content, block=block
                )
            )
            source_message_ids = [matched_message_id] if matched_message_id else []
            idxs = [matched_idx] if matched_idx is not None else []
        event_ts = _parse_datetime(event_record.get("Creation Time")) or episode.timestamp
        sem = SemanticMemory(
            user_id=owner_id,
            content=content,
            knowledge_type=item.get("knowledge_type", "knowledge") or "knowledge",
            created_at=event_ts or datetime.now(),
            timestamp=event_ts,
            source_episodes=[episode.episode_id],
            matched_message_id=matched_message_id,
            matched_message_index=matched_idx,
            matched_message_content=matched_message_content,
            matched_block_id=block.block_id,
            source_message_ids=source_message_ids,
            source_message_indices=idxs,
            confidence=1.0,
            metadata={
                "source": SOURCE_EVENTS,
                "generation_method": "perltqa_event_episode_semantic",
                "event_id": episode.metadata.get("event_id", ""),
                "dialogue_id": episode.metadata.get("dialogue_id", ""),
                "theme": event_record.get("Theme", ""),
                "creation_time": event_record.get("Creation Time", ""),
            },
            tags=["perltqa", SOURCE_EVENTS, "episode_semantic"],
        )
        semantics.append(sem)
    return semantics


def _build_profile_layers(
    memory_system: Any,
    *,
    owner_id: str,
    character_id: str,
    character_name: str,
    character_data: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[MessageBlock], List[SemanticMemory]]:
    profile = character_data.get("profile") or {}
    description = character_data.get("profile_description") or ""
    description_sentences = _split_profile_description(description)
    messages = [
        _make_message(
            role=character_name or owner_id,
            content=sentence,
            source=SOURCE_PROFILE,
            character_id=character_id,
            character_name=character_name,
            timestamp="",
            original_speaker=character_name or owner_id,
            index=i,
            extra_metadata={"profile_description_index": i},
        )
        for i, sentence in enumerate(description_sentences)
    ]
    episode = _make_episode(
        owner_id=owner_id,
        title=f"{character_name or owner_id} profile description",
        content=description,
        messages=messages,
        source=SOURCE_PROFILE,
        timestamp="",
        metadata={"character_id": character_id, "character_name": character_name},
    )
    block = _make_block(
        owner_id=owner_id,
        episode_id=episode.episode_id,
        messages=messages,
        source=SOURCE_PROFILE,
        metadata={"character_id": character_id, "character_name": character_name},
    )
    memory_system._episode_repository.save(episode)
    semantics: List[SemanticMemory] = []
    for key, value in profile.items():
        if not _is_non_empty(value):
            continue
        content = f"{character_name or owner_id}'s {key} is {_clean_text(value)}"
        labels = PROFILE_FIELD_LABELS.get(key, [])
        if not labels:
            labels = ["Identity"]
        label_to_keywords = {label: [content] for label in labels}
        annotations = _make_annotations(
            label_to_keywords,
            source="perltqa_profile_rule",
            memory_system=memory_system,
            is_new_label=False,
        )
        matched_message_id, matched_idx, matched_content = (
            memory_system._pick_best_message_for_semantic_text(semantic_text=content, block=block)
        )
        source_message_ids = [matched_message_id] if matched_message_id else []
        source_message_indices = [matched_idx] if matched_idx is not None else []
        sem = SemanticMemory(
            user_id=owner_id,
            content=content,
            knowledge_type="knowledge",
            created_at=datetime.now(),
            timestamp=None,
            source_episodes=[episode.episode_id],
            matched_message_id=matched_message_id,
            matched_message_index=matched_idx,
            matched_message_content=matched_content,
            matched_block_id=block.block_id,
            source_message_ids=source_message_ids,
            source_message_indices=source_message_indices,
            confidence=1.0,
            metadata={
                "source": SOURCE_PROFILE,
                "generation_method": "perltqa_profile_rule",
                "profile_key": key,
                "character_id": character_id,
                "character_name": character_name,
            },
            tags=["perltqa", SOURCE_PROFILE],
            label_annotations=annotations,
        )
        memory_system._semantic_repository.save(sem)
        semantics.append(sem)
    return ([{"episode_object": episode}], [block], semantics)


def _format_social_relationship(rel: Dict[str, Any]) -> str:
    name = _clean_text(rel.get("Supporting Characters"))
    description = _clean_text(rel.get("Description"))
    relationship = _clean_text(rel.get("Relationship"))
    pieces: List[str] = []
    if name:
        pieces.append(name)
    if description:
        if pieces:
            pieces[-1] = pieces[-1] + ", " + description
        else:
            pieces.append(description)
    if relationship:
        pieces.append(f"Relationship: {relationship}")
    return " ".join(pieces).strip()


def _build_social_relationship_layers(
    memory_system: Any,
    *,
    owner_id: str,
    character_id: str,
    character_name: str,
    character_data: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[MessageBlock], List[SemanticMemory]]:
    relationships = character_data.get("social_relationship") or {}
    if not isinstance(relationships, dict) or not relationships:
        return ([], [], [])
    messages: List[Dict[str, Any]] = []
    relation_items = sorted(relationships.items(), key=lambda kv: str(kv[0]))
    for i, (rel_id, rel) in enumerate(relation_items):
        if not isinstance(rel, dict):
            continue
        content = _format_social_relationship(rel)
        if not content:
            continue
        supporting_name = (
            _clean_text(rel.get("Supporting Characters")) or character_name or owner_id
        )
        messages.append(
            _make_message(
                role=supporting_name,
                content=content,
                source=SOURCE_SOCIAL,
                character_id=character_id,
                character_name=character_name,
                timestamp="",
                original_speaker=supporting_name,
                index=i,
                extra_metadata={
                    "relationship_id": rel_id,
                    "supporting_character": supporting_name,
                    "relationship": _clean_text(rel.get("Relationship")),
                },
            )
        )
    if not messages:
        return ([], [], [])
    episode_content = "\n".join((m["content"] for m in messages))
    episode = _make_episode(
        owner_id=owner_id,
        title=f"{character_name or owner_id} social relationships",
        content=episode_content,
        messages=messages,
        source=SOURCE_SOCIAL,
        timestamp="",
        metadata={"character_id": character_id, "character_name": character_name},
    )
    block = _make_block(
        owner_id=owner_id,
        episode_id=episode.episode_id,
        messages=messages,
        source=SOURCE_SOCIAL,
        metadata={"character_id": character_id, "character_name": character_name},
    )
    memory_system._episode_repository.save(episode)
    semantics: List[SemanticMemory] = []
    for idx, msg in enumerate(messages):
        rel_meta = msg.get("metadata", {}) or {}
        supporting_name = _clean_text(rel_meta.get("supporting_character"))
        relationship = _clean_text(rel_meta.get("relationship"))
        label_to_keywords: Dict[str, List[str]] = {}
        if supporting_name:
            label_to_keywords["Character"] = [supporting_name]
        if relationship:
            label_to_keywords["Relationship"] = [relationship]
        annotations = _make_annotations(
            label_to_keywords,
            source="perltqa_social_rule",
            memory_system=memory_system,
            is_new_label=False,
        )
        sem = SemanticMemory(
            user_id=owner_id,
            content=msg["content"],
            knowledge_type="knowledge",
            created_at=datetime.now(),
            timestamp=None,
            source_episodes=[episode.episode_id],
            matched_message_id=msg.get("message_id"),
            matched_message_index=idx,
            matched_message_content=msg.get("content"),
            matched_block_id=block.block_id,
            source_message_ids=[msg.get("message_id", "")],
            source_message_indices=[idx],
            confidence=1.0,
            metadata={
                "source": SOURCE_SOCIAL,
                "generation_method": "perltqa_social_rule",
                "relationship_id": rel_meta.get("relationship_id", ""),
                "character_id": character_id,
                "character_name": character_name,
            },
            tags=["perltqa", SOURCE_SOCIAL],
            label_annotations=annotations,
        )
        memory_system._semantic_repository.save(sem)
        semantics.append(sem)
    return ([{"episode_object": episode}], [block], semantics)


def _split_event_content_sentences(text: str) -> List[str]:
    """Split one PerLTQA event paragraph into sentence-level source messages."""
    return _split_profile_description(text)


def _iter_event_messages(
    event_id: str, event_record: Dict[str, Any], *, character_id: str, character_name: str
) -> List[Dict[str, Any]]:
    """Treat each event sentence as a bottom-level source message."""
    content = _clean_text(event_record.get("content", ""))
    sentences = _split_event_content_sentences(content)
    creation_time = event_record.get("Creation Time", "") or ""
    messages: List[Dict[str, Any]] = []
    for i, sentence in enumerate(sentences):
        if not _clean_text(sentence):
            continue
        messages.append(
            _make_message(
                role=character_name or character_id,
                content=sentence,
                source=SOURCE_EVENTS,
                character_id=character_id,
                character_name=character_name,
                timestamp=creation_time,
                original_speaker=character_name or character_id,
                index=i,
                extra_metadata={
                    "source_subtype": "event",
                    "event_id": event_id,
                    "dialogue_id": "",
                    "event_message_index": i,
                    "summary": event_record.get("summary", ""),
                    "theme": event_record.get("Theme", ""),
                    "creation_time": creation_time,
                    "characters": event_record.get("Characters", []),
                },
            )
        )
    return messages


def _parse_dialogue_line(line: str) -> Tuple[str, str]:
    raw = str(line).strip()
    if not raw:
        return ("", "")
    if ":" in raw:
        speaker, text = raw.split(":", 1)
        speaker = speaker.strip()
        if 0 < len(speaker) <= 80:
            return (speaker, text.strip())
    return ("", raw)


def _iter_dialogue_messages(
    dialogue_id: str, dialogue: Dict[str, Any], *, character_id: str, character_name: str
) -> List[Dict[str, Any]]:
    contents = dialogue.get("contents") or {}
    if not isinstance(contents, dict):
        return []
    messages: List[Dict[str, Any]] = []
    running_index = 0
    for ts_key in sorted(contents.keys(), key=lambda x: str(x)):
        lines = contents.get(ts_key) or []
        if isinstance(lines, str):
            lines = [lines]
        for line in lines:
            speaker, text = _parse_dialogue_line(str(line))
            role = speaker or character_name or character_id
            messages.append(
                _make_message(
                    role=role,
                    content=text,
                    source=SOURCE_EVENTS,
                    character_id=character_id,
                    character_name=character_name,
                    timestamp=ts_key,
                    original_speaker=role,
                    index=running_index,
                    extra_metadata={
                        "source_subtype": "dialogue",
                        "dialogue_id": dialogue_id,
                        "event_id": dialogue.get("events", ""),
                        "raw_line": str(line),
                    },
                )
            )
            running_index += 1
    return messages


def _build_event_dialogue_layers(
    memory_system: Any,
    *,
    owner_id: str,
    character_id: str,
    character_name: str,
    character_data: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[MessageBlock], List[SemanticMemory]]:
    """Build PerLTQA events and dialogues as two independent sources.

    New scheme:
    * events are not paired with dialogues.  Each event paragraph is an Episode;
      its sentence split is the bottom-level MessageBlock.
    * dialogues are not paired with events.  Each dialogue group is an Episode;
      its individual utterances are the bottom-level MessageBlock.
    * no semantic memories are extracted from Episode content.  All event/dialogue
      semantics are extracted once from the bottom-level message blocks by the
      caller's atomic-message extraction path.

    This preserves the original five-layer JSONL schema while making every
    semantic point directly to one matched source message.
    """
    raw_events = character_data.get("events") or {}
    raw_dialogues = character_data.get("dialogues") or {}
    events = raw_events if isinstance(raw_events, dict) else {}
    dialogues = raw_dialogues if isinstance(raw_dialogues, dict) else {}
    if not events and (not dialogues):
        return ([], [], [])
    created_episodes: List[Dict[str, Any]] = []
    blocks: List[MessageBlock] = []
    semantics: List[SemanticMemory] = []
    for event_id, event_record_raw in sorted(events.items(), key=lambda kv: str(kv[0])):
        event_record = event_record_raw if isinstance(event_record_raw, dict) else {}
        content = _clean_text(event_record.get("content", ""))
        creation_time = event_record.get("Creation Time", "") or ""
        messages = _iter_event_messages(
            str(event_id), event_record, character_id=character_id, character_name=character_name
        )
        title = _clean_text(event_record.get("summary")) or f"PerLTQA event {event_id}"
        episode = _make_episode(
            owner_id=owner_id,
            title=title,
            content=content,
            messages=messages,
            source=SOURCE_EVENTS,
            timestamp=creation_time,
            metadata={
                "character_id": character_id,
                "character_name": character_name,
                "source_subtype": "event",
                "event_id": str(event_id),
                "dialogue_id": "",
                "summary": event_record.get("summary", ""),
                "theme": event_record.get("Theme", ""),
                "creation_time": creation_time,
                "last_accessed_time": event_record.get("Last Accessed Time", ""),
                "characters": event_record.get("Characters", []),
            },
        )
        block = _make_block(
            owner_id=owner_id,
            episode_id=episode.episode_id,
            messages=messages,
            source=SOURCE_EVENTS,
            metadata={
                "character_id": character_id,
                "character_name": character_name,
                "source_subtype": "event",
                "event_id": str(event_id),
                "dialogue_id": "",
                "creation_time": creation_time,
            },
        )
        memory_system._episode_repository.save(episode)
        created_episodes.append({"episode_object": episode})
        blocks.append(block)
    for dialogue_id, dialogue_raw in sorted(dialogues.items(), key=lambda kv: str(kv[0])):
        dialogue = dialogue_raw if isinstance(dialogue_raw, dict) else {}
        messages = _iter_dialogue_messages(
            str(dialogue_id), dialogue, character_id=character_id, character_name=character_name
        )
        event_id = str(dialogue.get("events", "") or "")
        episode_content_lines: List[str] = []
        for msg in messages:
            ts = msg.get("timestamp", "")
            role = msg.get("role", "")
            content = msg.get("content", "")
            prefix = f"[{ts}] " if ts else ""
            episode_content_lines.append(
                f"{prefix}{role}: {content}" if role else f"{prefix}{content}"
            )
        episode_content = "\n".join(episode_content_lines)
        first_ts = messages[0].get("timestamp", "") if messages else ""
        episode = _make_episode(
            owner_id=owner_id,
            title=f"PerLTQA dialogue {dialogue_id}",
            content=episode_content,
            messages=messages,
            source=SOURCE_EVENTS,
            timestamp=first_ts,
            metadata={
                "character_id": character_id,
                "character_name": character_name,
                "source_subtype": "dialogue",
                "event_id": event_id,
                "dialogue_id": str(dialogue_id),
            },
        )
        block = _make_block(
            owner_id=owner_id,
            episode_id=episode.episode_id,
            messages=messages,
            source=SOURCE_EVENTS,
            metadata={
                "character_id": character_id,
                "character_name": character_name,
                "source_subtype": "dialogue",
                "event_id": event_id,
                "dialogue_id": str(dialogue_id),
            },
        )
        memory_system._episode_repository.save(episode)
        created_episodes.append({"episode_object": episode})
        blocks.append(block)
    return (created_episodes, blocks, semantics)


def _annotation_response_to_annotations(
    memory_system: Any, resp: Dict[str, Any], *, existing_inventory: Sequence[str]
) -> List[SemanticLabelAnnotation]:
    annotations: List[SemanticLabelAnnotation] = []
    inventory_norm = {memory_system._normalize_label(x).lower() for x in existing_inventory if x}
    for bucket_name, is_new_default in (("existing_labels", False), ("new_labels", True)):
        bucket = resp.get(bucket_name, {}) if isinstance(resp, dict) else {}
        if not isinstance(bucket, dict):
            continue
        for label, keywords in bucket.items():
            norm_label = _normalize_perltqa_label(memory_system, label)
            if not norm_label:
                continue
            if not isinstance(keywords, list):
                keywords = []
            clean_keywords = _dedupe_strings(keywords)
            if not clean_keywords:
                continue
            is_new = is_new_default and norm_label.lower() not in inventory_norm
            annotations.append(
                SemanticLabelAnnotation(
                    label=norm_label,
                    keywords=clean_keywords,
                    is_new_label=is_new,
                    metadata={"source": "perltqa_agent_label_annotation"},
                )
            )
    return annotations


def _annotate_non_profile_semantics(
    memory_system: Any,
    *,
    owner_id: str,
    semantic_memories: List[SemanticMemory],
    skip_sources: Sequence[str] = (SOURCE_PROFILE,),
) -> List[SemanticMemory]:
    if not semantic_memories:
        return []
    label_inventory = list(DEFAULT_SEMANTIC_LABELS)
    skip_source_set = set(skip_sources)
    for sem in semantic_memories:
        source = (sem.metadata or {}).get("source", "")
        if source in skip_source_set:
            sem = _merge_extra_annotations(memory_system, sem, [])
            continue
        try:
            with _api_llm_scope(
                memory_system,
                reasoning_effort="minimal",
                min_completion_tokens=GPT5_PERLTQA_COMPLETION_TOKENS,
            ):
                resp = memory_system._extract_label_annotations_for_semantic(
                    semantic=sem, label_inventory=label_inventory
                )
        except Exception as exc:
            logger.warning(
                "PerLTQA label annotation failed for semantic %s: %s", sem.memory_id, exc
            )
            sem = _merge_extra_annotations(memory_system, sem, [])
            continue
        new_annotations = _annotation_response_to_annotations(
            memory_system,
            resp if isinstance(resp, dict) else {},
            existing_inventory=label_inventory,
        )
        sem = _merge_extra_annotations(memory_system, sem, new_annotations)
        proposed_new_labels = [ann.label for ann in new_annotations if ann.is_new_label]
        if proposed_new_labels and hasattr(memory_system, "_update_label_inventory"):
            label_inventory = memory_system._update_label_inventory(
                label_inventory, proposed_new_labels
            )
    memory_system._semantic_repository.overwrite_all(owner_id, semantic_memories)
    return semantic_memories


def _mark_candidate_semantics_with_perltqa_metadata(
    candidates: List[SemanticMemory], *, block_by_id: Dict[str, MessageBlock]
) -> None:
    for cand in candidates:
        block = block_by_id.get(cand.matched_block_id or "")
        block_meta = block.metadata if block else {}
        event_id = block_meta.get("event_id", "") if block_meta else ""
        dialogue_id = block_meta.get("dialogue_id", "") if block_meta else ""
        source_subtype = block_meta.get("source_subtype", "") if block_meta else ""
        episode_id = block.episode_id if block else None
        if episode_id and episode_id not in cand.source_episodes:
            cand.source_episodes.append(episode_id)
        cand.metadata.update(
            {
                "source": SOURCE_EVENTS,
                "source_subtype": source_subtype,
                "generation_method": "perltqa_event_dialogue_message_atomic_candidate",
                "event_id": event_id,
                "dialogue_id": dialogue_id,
                "character_name": block_meta.get("character_name", "") if block_meta else "",
                "creation_time": block_meta.get("creation_time", "") if block_meta else "",
            }
        )
        if "perltqa" not in cand.tags:
            cand.tags.append("perltqa")
        if SOURCE_EVENTS not in cand.tags:
            cand.tags.append(SOURCE_EVENTS)


def build_perltqa_character_memory(
    memory_system: Any,
    *,
    owner_id: str,
    character_data: Dict[str, Any],
    character_id: str = "",
    character_name: str = "",
    enable_dialogue_atomic_backfill: bool = True,
    annotate_agent_labels: bool = True,
) -> Dict[str, Any]:
    """
    Build all five memory layers for one PerLTQA character.

    Parameters
    ----------
    memory_system:
        An initialized `worldmem.construction.builder.MemoryBuilder` instance.
    owner_id:
        Memory namespace / file prefix. Each PerLTQA character should get a
        separate owner_id.
    character_data:
        One character object from `perltmem_en_v2.json`.
    character_id:
        Dataset character key if available.
    character_name:
        Protagonist name. If omitted, it is inferred from `profile.Protagonist`.
    enable_dialogue_atomic_backfill:
        Run thinking-mode message-block atomic extraction for both event-sentence
        blocks and dialogue-utterance blocks.  Events and dialogues are extracted
        independently and no episode-level semantic extraction is performed.
    annotate_agent_labels:
        Use the normal non-thinking label/keyword agent for non-profile
        semantics after deterministic profile/social annotations have been set.
    persist_lancedb:
        Also append all built layers to the existing LanceDB tables.
    """
    profile = character_data.get("profile") or {}
    if not character_name:
        character_name = _clean_text(profile.get("Protagonist")) or owner_id
    if not character_id:
        character_id = owner_id
    created_episodes: List[Dict[str, Any]] = []
    message_blocks: List[MessageBlock] = []
    semantic_memories: List[SemanticMemory] = []
    profile_eps, profile_blocks, profile_semantics = _build_profile_layers(
        memory_system,
        owner_id=owner_id,
        character_id=character_id,
        character_name=character_name,
        character_data=character_data,
    )
    created_episodes.extend(profile_eps)
    message_blocks.extend(profile_blocks)
    semantic_memories.extend(profile_semantics)
    social_eps, social_blocks, social_semantics = _build_social_relationship_layers(
        memory_system,
        owner_id=owner_id,
        character_id=character_id,
        character_name=character_name,
        character_data=character_data,
    )
    created_episodes.extend(social_eps)
    message_blocks.extend(social_blocks)
    semantic_memories.extend(social_semantics)
    event_eps, event_blocks, event_semantics = _build_event_dialogue_layers(
        memory_system,
        owner_id=owner_id,
        character_id=character_id,
        character_name=character_name,
        character_data=character_data,
    )
    created_episodes.extend(event_eps)
    message_blocks.extend(event_blocks)
    semantic_memories.extend(event_semantics)
    memory_system._persist_message_blocks(owner_id, message_blocks)
    accepted_backfill_semantics: List[SemanticMemory] = []
    if enable_dialogue_atomic_backfill and event_blocks:
        extractable_event_blocks = [
            block
            for block in event_blocks
            if _clean_text(getattr(block, "content", "")) and getattr(block, "messages", None)
        ]
        if extractable_event_blocks:
            atomic_entries = memory_system._extract_atomic_entries_from_message_blocks(
                owner_id=owner_id, message_blocks=extractable_event_blocks
            )
            candidate_semantics = memory_system._convert_atomic_entries_to_candidate_semantics(
                owner_id=owner_id,
                atomic_entries=atomic_entries,
                message_blocks=extractable_event_blocks,
            )
            block_by_id = {block.block_id: block for block in extractable_event_blocks}
            _mark_candidate_semantics_with_perltqa_metadata(
                candidate_semantics, block_by_id=block_by_id
            )
            for cand in candidate_semantics:
                memory_system._semantic_repository.save(cand)
                semantic_memories.append(cand)
                accepted_backfill_semantics.append(cand)
            if accepted_backfill_semantics:
                memory_system._semantic_repository.overwrite_all(owner_id, semantic_memories)
    if annotate_agent_labels:
        semantic_memories = _annotate_non_profile_semantics(
            memory_system,
            owner_id=owner_id,
            semantic_memories=semantic_memories,
            skip_sources=(SOURCE_PROFILE,),
        )
    else:
        for sem in semantic_memories:
            sem = _merge_extra_annotations(memory_system, sem, [])
        memory_system._semantic_repository.overwrite_all(owner_id, semantic_memories)
    semantic_memories = memory_system._backfill_missing_keywords_across_semantics(
        owner_id=owner_id, semantic_memories=semantic_memories
    )
    situations = memory_system._rebuild_situations_from_keywords(
        owner_id=owner_id, created_episodes=created_episodes, semantic_memories=semantic_memories
    )
    mental_records = (
        memory_system._build_mental_memory(
            owner_id=owner_id,
            situations=situations,
            label_cluster_threshold=getattr(
                memory_system.config, "mental_label_cluster_threshold", 0.8
            ),
        )
        or []
    )
    return {
        "owner_id": owner_id,
        "character_id": character_id,
        "character_name": character_name,
        "episodes_created": len(created_episodes),
        "message_blocks_created": len(message_blocks),
        "semantic_memories_created": len(semantic_memories),
        "profile_semantics": len(profile_semantics),
        "social_semantics": len(social_semantics),
        "event_episode_semantics": len(event_semantics),
        "event_dialogue_message_semantics": len(accepted_backfill_semantics),
        "accepted_dialogue_atomic_semantics": len(accepted_backfill_semantics),
        "situations_created": len(situations),
        "mental_units_created": len(mental_records),
    }
