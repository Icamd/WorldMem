import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class SituationAggregate:
    aggregate_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    episode_id: str = ""
    content: str = ""
    semantic_ids: List[str] = field(default_factory=list)
    timestamp_start: Optional[datetime] = None
    timestamp_end: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aggregate_id": self.aggregate_id,
            "episode_id": self.episode_id,
            "content": self.content,
            "semantic_ids": self.semantic_ids,
            "timestamp_start": self.timestamp_start.isoformat() if self.timestamp_start else None,
            "timestamp_end": self.timestamp_end.isoformat() if self.timestamp_end else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SituationAggregate":
        ts_start = data.get("timestamp_start")
        if isinstance(ts_start, str):
            ts_start = datetime.fromisoformat(ts_start)
        ts_end = data.get("timestamp_end")
        if isinstance(ts_end, str):
            ts_end = datetime.fromisoformat(ts_end)
        return cls(
            aggregate_id=data.get("aggregate_id", str(uuid.uuid4())),
            episode_id=data.get("episode_id", ""),
            content=data.get("content", ""),
            semantic_ids=data.get("semantic_ids", []),
            timestamp_start=ts_start,
            timestamp_end=ts_end,
        )


@dataclass
class SituationMemory:
    situation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    summary: str = ""
    routing_summary: str = ""
    final_summary: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    aggregates: List[SituationAggregate] = field(default_factory=list)
    timestamp_start: Optional[datetime] = None
    timestamp_end: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.routing_summary and self.summary:
            self.routing_summary = self.summary
        if not self.summary and self.routing_summary:
            self.summary = self.routing_summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "situation_id": self.situation_id,
            "user_id": self.user_id,
            "summary": self.summary,
            "routing_summary": self.routing_summary,
            "final_summary": self.final_summary,
            "created_at": self.created_at.isoformat(),
            "aggregates": [agg.to_dict() for agg in self.aggregates],
            "timestamp_start": self.timestamp_start.isoformat() if self.timestamp_start else None,
            "timestamp_end": self.timestamp_end.isoformat() if self.timestamp_end else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SituationMemory":
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        timestamp_start = data.get("timestamp_start")
        if isinstance(timestamp_start, str):
            timestamp_start = datetime.fromisoformat(timestamp_start)
        timestamp_end = data.get("timestamp_end")
        if isinstance(timestamp_end, str):
            timestamp_end = datetime.fromisoformat(timestamp_end)
        routing_summary = data.get("routing_summary", "")
        summary = data.get("summary", "")
        final_summary = data.get("final_summary")
        if not routing_summary and summary:
            routing_summary = summary
        return cls(
            situation_id=data.get("situation_id", str(uuid.uuid4())),
            user_id=data.get("user_id", ""),
            summary=summary or routing_summary,
            routing_summary=routing_summary or summary,
            final_summary=final_summary,
            created_at=created_at if created_at else datetime.now(),
            aggregates=[SituationAggregate.from_dict(x) for x in data.get("aggregates", [])],
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            metadata=data.get("metadata", {}),
        )
