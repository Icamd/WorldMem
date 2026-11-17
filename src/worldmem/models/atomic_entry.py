import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class AtomicMemoryEntry:
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    block_id: str = ""
    lossless_restatement: str = ""
    keywords: List[str] = field(default_factory=list)
    timestamp: Optional[datetime] = None
    location: Optional[str] = None
    persons: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    topic: Optional[str] = None
    source_message_ids: List[str] = field(default_factory=list)
    source_message_indices: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "user_id": self.user_id,
            "block_id": self.block_id,
            "lossless_restatement": self.lossless_restatement,
            "keywords": self.keywords,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "location": self.location,
            "persons": self.persons,
            "entities": self.entities,
            "topic": self.topic,
            "source_message_ids": self.source_message_ids,
            "source_message_indices": self.source_message_indices,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AtomicMemoryEntry":
        ts = data.get("timestamp")
        return cls(
            entry_id=data.get("entry_id", str(uuid.uuid4())),
            user_id=data.get("user_id", ""),
            block_id=data.get("block_id", ""),
            lossless_restatement=data.get("lossless_restatement", ""),
            keywords=data.get("keywords", []),
            timestamp=datetime.fromisoformat(ts) if ts else None,
            location=data.get("location"),
            persons=data.get("persons", []),
            entities=data.get("entities", []),
            topic=data.get("topic"),
            source_message_ids=data.get("source_message_ids", []),
            source_message_indices=data.get("source_message_indices", []),
            metadata=data.get("metadata", {}),
        )
