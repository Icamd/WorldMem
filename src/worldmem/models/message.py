"""
Message Data Models
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict


@dataclass
class Message:
    """Message data model"""

    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self, clean_metadata: bool = True) -> Dict[str, Any]:
        """
        Convert to dictionary

        Args:
            clean_metadata: Whether to clean redundant information from metadata
        """
        metadata = self.metadata
        if clean_metadata:
            metadata = self._clean_metadata(self.metadata)
        return {
            "message_id": self.message_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": metadata,
        }

    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean redundant information from metadata

        Args:
            metadata: Original metadata

        Returns:
            Cleaned metadata with redundant fields removed
        """
        if not metadata:
            return {}
        redundant_fields = {"original_text", "timestamp"}
        valuable_fields = {
            "original_speaker",
            "dataset_timestamp",
            "image_url",
            "blip_caption",
            "search_query",
            "has_multimodal_content",
        }
        cleaned_metadata = {}
        for key, value in metadata.items():
            if key not in redundant_fields:
                if value is not None or key in valuable_fields:
                    cleaned_metadata[key] = value
        return cleaned_metadata

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary"""
        return cls(
            message_id=data.get("message_id", str(uuid.uuid4())),
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"])
            if isinstance(data["timestamp"], str)
            else data["timestamp"],
            metadata=data.get("metadata", {}),
        )

    def __str__(self) -> str:
        return f"[{self.role}] {self.content[:50]}..."
