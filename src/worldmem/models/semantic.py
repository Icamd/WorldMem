"""
Semantic Memory Data Model
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class SemanticLabelAnnotation:
    label: str = ""
    keywords: List[str] = field(default_factory=list)
    is_new_label: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "keywords": self.keywords,
            "is_new_label": self.is_new_label,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticLabelAnnotation":
        return cls(
            label=data.get("label", ""),
            keywords=data.get("keywords", []),
            is_new_label=data.get("is_new_label", False),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SemanticMemory:
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    content: str = ""
    knowledge_type: str = "knowledge"
    created_at: datetime = field(default_factory=datetime.now)
    timestamp: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    source_episodes: List[str] = field(default_factory=list)
    matched_message_id: Optional[str] = None
    matched_message_index: Optional[int] = None
    matched_message_content: Optional[str] = None
    matched_block_id: Optional[str] = None
    source_message_ids: List[str] = field(default_factory=list)
    source_message_indices: List[int] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    revision_count: int = 1
    label_annotations: List[SemanticLabelAnnotation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "content": self.content,
            "knowledge_type": self.knowledge_type,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "source_episodes": self.source_episodes,
            "matched_message_id": self.matched_message_id,
            "matched_message_index": self.matched_message_index,
            "matched_message_content": self.matched_message_content,
            "matched_block_id": self.matched_block_id,
            "source_message_ids": self.source_message_ids,
            "source_message_indices": self.source_message_indices,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "tags": self.tags,
            "revision_count": self.revision_count,
            "label_annotations": [x.to_dict() for x in self.label_annotations],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticMemory":
        created_at = data.get("created_at")
        timestamp = data.get("timestamp")
        updated_at = data.get("updated_at")
        return cls(
            memory_id=data.get("memory_id", str(uuid.uuid4())),
            user_id=data.get("user_id", ""),
            content=data.get("content", ""),
            knowledge_type=data.get("knowledge_type", "knowledge"),
            created_at=datetime.fromisoformat(created_at) if created_at else datetime.now(),
            timestamp=datetime.fromisoformat(timestamp) if timestamp else None,
            updated_at=datetime.fromisoformat(updated_at) if updated_at else None,
            source_episodes=data.get("source_episodes", []),
            matched_message_id=data.get("matched_message_id"),
            matched_message_index=data.get("matched_message_index"),
            matched_message_content=data.get("matched_message_content"),
            matched_block_id=data.get("matched_block_id"),
            source_message_ids=data.get("source_message_ids", []),
            source_message_indices=data.get("source_message_indices", []),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            revision_count=data.get("revision_count", 1),
            label_annotations=[
                SemanticLabelAnnotation.from_dict(x) for x in data.get("label_annotations", [])
            ],
        )

    def update_content(self, new_content: str, additional_episodes: List[str] = None) -> None:
        self.content = new_content
        self.updated_at = datetime.now()
        self.revision_count += 1
        if additional_episodes:
            all_episodes = set(self.source_episodes)
            all_episodes.update(additional_episodes)
            self.source_episodes = list(all_episodes)

    def add_source_episode(self, episode_id: str) -> None:
        if episode_id not in self.source_episodes:
            self.source_episodes.append(episode_id)

    def remove_source_episode(self, episode_id: str) -> None:
        if episode_id in self.source_episodes:
            self.source_episodes.remove(episode_id)

    def add_tag(self, tag: str) -> None:
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> None:
        if tag in self.tags:
            self.tags.remove(tag)
