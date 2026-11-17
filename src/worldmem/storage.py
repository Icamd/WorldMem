"""The canonical storage format is one JSONL file per owner and record type."""

import json
import os
import re
import threading
import uuid
from pathlib import Path

LAYERS = ("message_blocks", "episodes", "semantic", "situation", "mental")


def validate_owner(owner):
    if not re.fullmatch("[\\w .-]+", owner, flags=re.UNICODE) or owner in {".", ".."}:
        raise ValueError("owner must be a nonempty file-safe identifier without path separators")
    return owner


def layer_path(root, owner, layer):
    validate_owner(owner)
    if layer not in LAYERS:
        raise ValueError(f"Unknown memory layer: {layer}")
    return Path(root) / layer / f"{owner}_{layer}.jsonl"


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    """Replace a result/checkpoint atomically within its parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class JsonlRepository:
    def __init__(self, root, layer, record_type, id_field):
        self.root, self.layer = (root, layer)
        self.record_type, self.id_field = (record_type, id_field)
        self._lock = threading.RLock()

    def save(self, item):
        path = layer_path(self.root, item.user_id, self.layer)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
        return getattr(item, self.id_field)

    def list_by_user(self, owner):
        with self._lock:
            return [
                self.record_type.from_dict(item)
                for item in read_jsonl(layer_path(self.root, owner, self.layer))
            ]

    list_user_items = list_by_user

    def overwrite_all(self, owner, items):
        path = layer_path(self.root, owner, self.layer)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with self._lock:
            try:
                with temporary.open("w", encoding="utf-8") as stream:
                    for item in items:
                        stream.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
