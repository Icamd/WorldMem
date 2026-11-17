from datetime import datetime
from typing import Any, Dict, List


def parse_timestamp(value: str) -> datetime:
    """Parse dataset timestamps such as '1:56 pm on 8 May, 2023'."""
    value = " ".join(value.split())
    if " on " not in value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now()
    time_part, date_part = value.split(" on ")
    time_part = time_part.lower().strip()
    hour = 0
    minute = 0
    is_pm = "pm" in time_part
    time_part = time_part.replace("pm", "").replace("am", "").strip()
    if ":" in time_part:
        hour_str, minute_str = time_part.split(":", 1)
        hour = int(hour_str)
        minute = int(minute_str)
    else:
        hour = int(time_part)
    if is_pm and hour != 12:
        hour += 12
    if not is_pm and hour == 12:
        hour = 0
    months = {
        "january": 1,
        "jan": 1,
        "february": 2,
        "feb": 2,
        "march": 3,
        "mar": 3,
        "april": 4,
        "apr": 4,
        "may": 5,
        "june": 6,
        "jun": 6,
        "july": 7,
        "jul": 7,
        "august": 8,
        "aug": 8,
        "september": 9,
        "sep": 9,
        "october": 10,
        "oct": 10,
        "november": 11,
        "nov": 11,
        "december": 12,
        "dec": 12,
    }
    parts = date_part.replace(",", "").split()
    day = 1
    month = 1
    year = datetime.now().year
    for part in parts:
        lower = part.lower()
        if lower in months:
            month = months[lower]
        elif part.isdigit():
            num = int(part)
            if num > 31:
                year = num
            else:
                day = num
    return datetime(year=year, month=month, day=day, hour=hour, minute=minute)


def build_messages(conversation: Dict[str, Any], session_limit: int = None) -> List[Dict[str, Any]]:
    speaker_a = conversation.get("speaker_a", "speaker_a")
    speaker_b = conversation.get("speaker_b", "speaker_b")
    special_keys = {"speaker_a", "speaker_b"}
    messages: List[Dict[str, Any]] = []
    session_count = 0
    for key, chats in conversation.items():
        if key in special_keys or key.endswith("_date_time"):
            continue
        if session_limit and session_count >= session_limit:
            break
        session_count += 1
        timestamp_raw = conversation.get(f"{key}_date_time")
        timestamp = parse_timestamp(timestamp_raw) if timestamp_raw else datetime.now()
        for chat in chats or []:
            speaker = chat.get("speaker", speaker_a)
            text = chat.get("text", "")
            parts = [text]
            if chat.get("blip_caption"):
                parts.append(f"[Image: {chat['blip_caption']}]")
            if chat.get("query"):
                parts.append(f"[Search: {chat['query']}]")
            role = speaker if speaker in (speaker_a, speaker_b) else "user"
            messages.append(
                {
                    "role": role,
                    "content": " ".join(parts),
                    "timestamp": timestamp.isoformat(),
                    "metadata": {
                        "original_speaker": speaker,
                        "dataset_timestamp": timestamp_raw,
                        "blip_caption": chat.get("blip_caption"),
                        "search_query": chat.get("query"),
                    },
                }
            )
    return messages
