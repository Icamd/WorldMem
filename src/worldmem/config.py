"""Portable configuration; credentials are read from the environment by clients."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class MemoryConfig:
    storage_path: str = "outputs/memories"
    llm_model: str = "Qwen/Qwen3-8B"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    language: str = "en"
    episode_min_messages: int = 1
    episode_max_messages: int = 40
    enable_prediction_correction: bool = True
    extract_semantic_per_episode: bool = False
    semantic_similarity_threshold: float = 1.0
    mental_label_cluster_threshold: float = 0.78
    mental_semantic_cluster_threshold: float = 0.8
    max_statements_for_prediction: int = 10
    statement_similarity_threshold: float = 0.7
    prediction_temperature: float = 0.3
    max_workers: int = 1
    construction_regular_min_tokens: int = 4096
    construction_semantic_min_tokens: int = 16384

    def __post_init__(self):
        if self.max_workers < 1 or self.episode_min_messages < 1:
            raise ValueError("max_workers and episode_min_messages must be positive")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
