from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


def sanitize_owner_id(text: str, fallback: str) -> str:
    text = (text or "").strip() or fallback
    text = re.sub("\\s+", "_", text)
    text = re.sub("[^A-Za-z0-9_\\-\\.]+", "_", text)
    text = re.sub("_+", "_", text).strip("_")
    return text or fallback


def get_character_name(character_data: Dict[str, Any], fallback: str) -> str:
    profile = character_data.get("profile") or {}
    name = str(profile.get("Protagonist") or "").strip()
    return name or fallback


def make_owner_id(character_key: str, idx: int, character_name: str, fmt: str) -> str:
    key = sanitize_owner_id(str(character_key or idx), f"character_{idx}")
    name = sanitize_owner_id(character_name, f"character_{idx}")
    if fmt == "name":
        return name
    if fmt == "index":
        return f"character_{idx}"
    if fmt == "key":
        return key
    if fmt == "name_index":
        return f"{name}_{idx}"
    return f"{name}_{key}"


def iter_perltqa_characters(data: Any) -> List[Tuple[str, Dict[str, Any]]]:
    """Normalize known PerLTQA JSON layouts to [(character_key, character_obj)]."""
    character_markers = {
        "profile",
        "profile_description",
        "social_relationship",
        "events",
        "dialogues",
    }
    if isinstance(data, list):
        out = []
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                key = str(item.get("id") or item.get("character_id") or idx)
                out.append((key, item))
        return out
    if isinstance(data, dict):
        if character_markers.intersection(data.keys()):
            return [(str(data.get("id") or data.get("character_id") or "0"), data)]
        out = []
        for key, value in data.items():
            if isinstance(value, dict):
                out.append((str(key), value))
        return out
    raise ValueError("Unsupported PerLTQA JSON layout: expected dict or list")
