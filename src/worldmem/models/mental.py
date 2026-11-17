import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class MentalSemanticLink:
    semantic_id: str = ""
    semantic_content: str = ""
    keywords: List[str] = field(default_factory=list)
    situation_ids: List[str] = field(default_factory=list)
    timestamps: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "semantic_id": self.semantic_id,
            "semantic_content": self.semantic_content,
            "keywords": self.keywords,
            "situation_ids": self.situation_ids,
            "timestamps": self.timestamps,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MentalSemanticLink":
        return cls(
            semantic_id=data.get("semantic_id", ""),
            semantic_content=data.get("semantic_content", ""),
            keywords=data.get("keywords", []),
            situation_ids=data.get("situation_ids", []),
            timestamps=data.get("timestamps", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MentalCluster:
    cluster_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    cluster_summary: str = ""
    detailed_summary: str = ""
    label: str = ""
    is_new_label: bool = False
    situation_ids: List[str] = field(default_factory=list)
    semantic_ids: List[str] = field(default_factory=list)
    semantic_links: List[MentalSemanticLink] = field(default_factory=list)
    timestamps: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "cluster_summary": self.cluster_summary,
            "detailed_summary": self.detailed_summary,
            "label": self.label,
            "is_new_label": self.is_new_label,
            "situation_ids": self.situation_ids,
            "semantic_ids": self.semantic_ids,
            "semantic_links": [x.to_dict() for x in self.semantic_links],
            "timestamps": self.timestamps,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MentalCluster":
        return cls(
            cluster_id=data.get("cluster_id", str(uuid.uuid4())),
            cluster_summary=data.get("cluster_summary", ""),
            detailed_summary=data.get("detailed_summary", ""),
            label=data.get("label", ""),
            is_new_label=data.get("is_new_label", False),
            situation_ids=data.get("situation_ids", []),
            semantic_ids=data.get("semantic_ids", []),
            semantic_links=[
                MentalSemanticLink.from_dict(x) for x in data.get("semantic_links", [])
            ],
            timestamps=data.get("timestamps", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MentalMemory:
    mental_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    content: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    clusters: List[MentalCluster] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mental_id": self.mental_id,
            "user_id": self.user_id,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "clusters": [c.to_dict() for c in self.clusters],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MentalMemory":
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        return cls(
            mental_id=data.get("mental_id", str(uuid.uuid4())),
            user_id=data.get("user_id", ""),
            content=data.get("content", ""),
            created_at=created_at if created_at else datetime.now(),
            clusters=[MentalCluster.from_dict(x) for x in data.get("clusters", [])],
            metadata=data.get("metadata", {}),
        )
