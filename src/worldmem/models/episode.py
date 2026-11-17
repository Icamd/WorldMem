"""
Episode Data Model
Episode data model
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .message import Message


@dataclass
class Episode:
    """Episode data model"""

    title: str
    content: str
    original_messages: List[Dict[str, Any]]
    message_count: int
    boundary_reason: str
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    timestamp: datetime = field(default_factory=datetime.now)
    time_range_start: Optional[datetime] = None
    time_range_end: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "user_id": self.user_id,
            "title": self.title,
            "content": self.content,
            "original_messages": self.original_messages,
            "message_count": self.message_count,
            "boundary_reason": self.boundary_reason,
            "created_at": self.created_at.isoformat(),
            "timestamp": self.timestamp.isoformat(),
            "time_range_start": self.time_range_start.isoformat()
            if self.time_range_start
            else None,
            "time_range_end": self.time_range_end.isoformat() if self.time_range_end else None,
            "metadata": self.metadata,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        timestamp = data.get("timestamp", data.get("created_at"))
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        time_range_start = data.get("time_range_start")
        if isinstance(time_range_start, str):
            time_range_start = datetime.fromisoformat(time_range_start)
        time_range_end = data.get("time_range_end")
        if isinstance(time_range_end, str):
            time_range_end = datetime.fromisoformat(time_range_end)
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        return cls(
            episode_id=data.get("episode_id", str(uuid.uuid4())),
            user_id=data.get("user_id", ""),
            title=data["title"],
            content=data["content"],
            original_messages=data["original_messages"],
            message_count=data["message_count"],
            boundary_reason=data["boundary_reason"],
            created_at=created_at if created_at else datetime.now(),
            timestamp=timestamp if timestamp else datetime.now(),
            time_range_start=time_range_start,
            time_range_end=time_range_end,
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
        )

    @classmethod
    def from_messages(
        cls,
        messages: List[Message],
        user_id: str,
        title: str = "",
        content: str = "",
        boundary_reason: str = "",
        timestamp: Optional[datetime] = None,
    ) -> "Episode":
        if timestamp is None:
            timestamp = datetime.now()
        time_range_start = messages[0].timestamp if messages else None
        time_range_end = messages[-1].timestamp if messages else None
        return cls(
            user_id=user_id,
            title=title,
            content=content,
            original_messages=[msg.to_dict(clean_metadata=True) for msg in messages],
            message_count=len(messages),
            boundary_reason=boundary_reason,
            timestamp=timestamp,
            time_range_start=time_range_start,
            time_range_end=time_range_end,
        )

    def get_messages(self) -> List[Message]:
        return [Message.from_dict(msg_data) for msg_data in self.original_messages]

    def get_conversation_text(self) -> str:
        lines = []
        for msg_data in self.original_messages:
            role = msg_data.get("role", "unknown")
            content = msg_data.get("content", "")
            timestamp = msg_data.get("timestamp")
            if timestamp:
                lines.append(f"[{timestamp}] {role}: {content}")
            else:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)
