
import json
import re
from typing import Any, List, Optional, Union

import json5
import json_repair


def safe_json_parse(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return json5.loads(text)
    except Exception:
        pass
    try:
        return json.loads(json_repair.repair_json(text))
    except Exception:
        return {}


def extract_json_block(text: str) -> str:
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group(0) if match else text


def extract_array_fallback(text: str) -> List[str]:
    match = re.search(r"\[([\s\S]*?)\]", text)
    if not match:
        return []
    items = re.split(r",\s*", match.group(1))
    return [i.strip(" \"'") for i in items if i.strip()]


def validate_list(items: Any) -> List[Any]:
    if not isinstance(items, list):
        return []
    cleaned = []
    for item in items:
        if isinstance(item, str) and item.strip():
            cleaned.append(item.strip())
        elif isinstance(item, dict):
            cleaned.append(item)
    return cleaned


def parse_with_fallbacks(raw_text: str, key: Optional[str] = None) -> Union[List[str], dict]:
    content = re.sub(r"```(?:json)?|```", "", raw_text).strip()

    data = safe_json_parse(content)
    if key and key in data:
        return validate_list(data[key])
    if not key and data:
        return data

    data = safe_json_parse(extract_json_block(content))
    if key and key in data:
        return validate_list(data[key])
    if not key and data:
        return data

    if key:
        fallback = extract_array_fallback(content)
        if fallback:
            return validate_list(fallback)

    return [] if key else {}
