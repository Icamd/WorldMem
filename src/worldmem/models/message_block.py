import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class MessageBlock:
    block_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    episode_id: Optional[str] = None
    source_message_ids: List[str] = field(default_factory=list)
    source_message_indices: List[int] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    content: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    timestamp_start: Optional[datetime] = None
    timestamp_end: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_id": self.block_id,
            "user_id": self.user_id,
            "episode_id": self.episode_id,
            "source_message_ids": self.source_message_ids,
            "source_message_indices": self.source_message_indices,
            "messages": self.messages,
            "content": self.content,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "timestamp_start": self.timestamp_start.isoformat() if self.timestamp_start else None,
            "timestamp_end": self.timestamp_end.isoformat() if self.timestamp_end else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MessageBlock":
        def _dt(x):
            return datetime.fromisoformat(x) if x else None

        return cls(
            block_id=data.get("block_id", str(uuid.uuid4())),
            user_id=data.get("user_id", ""),
            episode_id=data.get("episode_id"),
            source_message_ids=data.get("source_message_ids", []),
            source_message_indices=data.get("source_message_indices", []),
            messages=data.get("messages", []),
            content=data.get("content", ""),
            created_at=_dt(data.get("created_at")) or datetime.now(),
            timestamp_start=_dt(data.get("timestamp_start")),
            timestamp_end=_dt(data.get("timestamp_end")),
            metadata=data.get("metadata", {}),
        )
