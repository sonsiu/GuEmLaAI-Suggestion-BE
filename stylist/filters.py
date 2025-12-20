"""
Candidate filtering helpers (score, attributes, category).
"""


def filter_wardrobe_by_score(candidates: list):
    before = len(candidates or [])
    filtered = []
    for c in (candidates or []):
        try:
            score = float(c.get("score", 0))
        except Exception:
            filtered.append(c)
            continue
        if score >= 0:
            filtered.append(c)

    removed = before - len(filtered)
    if removed > 0:
        print(f"?? Filtered out {removed} candidates with score < 0")
    return filtered


def filter_wardrobe_by_formality(user_query: str, candidates: list):
    """
    Filters out casual items if user query mentions 'formal'.
    You can expand this logic for other styles later (e.g., sporty, elegant, etc.).
    """
    normalized_query = user_query.lower().strip()

    if "formal" in normalized_query:
        filtered = [item for item in candidates if item["text"].get("purpose", "").lower() != "casual"]
        print(f"?? Filtered out {len(candidates) - len(filtered)} casual items (formal query detected).")
        return filtered

    if "casual" in normalized_query:
        filtered = [item for item in candidates if item["text"].get("purpose", "").lower() != "formal"]
        print(f"?? Filtered out {len(candidates) - len(filtered)} formal items (casual query detected).")
        return filtered

    return candidates


def filter_wardrobe_by_attributes(attributes: dict, candidates: list):
    """
    Uses parsed attributes (formality/season/occasion) instead of plain string matching.
    Applies a soft filter for season and a hard filter for conflicting formality.
    """
    attrs = attributes or {}
    formality = str(attrs.get("formality", "")).lower()
    season_pref = [s.lower() for s in (attrs.get("season") or []) if isinstance(s, str)]
    occasion = str(attrs.get("occasion", "")).lower()

    filtered = []
    for c in candidates or []:
        item = c.get("text", {}) if isinstance(c, dict) else {}
        purpose = str(item.get("purpose", "")).lower()
        score = float(c.get("score", 0))

        if formality.startswith("formal") and purpose == "casual":
            continue
        if formality.startswith("casual") and purpose == "formal":
            continue

        item_seasons = [s.lower() for s in (item.get("season") or []) if isinstance(s, str)]
        bonus = 0.0
        if season_pref:
            if item_seasons and any(s in item_seasons for s in season_pref):
                bonus += 0.05
            elif item_seasons:
                bonus -= 0.05

        if occasion:
            desc = (item.get("desc") or "").lower()
            if occasion in desc:
                bonus += 0.02

        filtered.append({**c, "score": score + bonus})

    return filtered if filtered else candidates


def filter_wardrobe_by_category(category_prefix: str, candidates: list):
    """
    Filters candidates by normalized category prefix.
    """
    match category_prefix:
        case "top_":
            return [item for item in candidates if item["text"].get("category", "").lower().startswith("top_")]
        case "bottom_":
            return [item for item in candidates if item["text"].get("category", "").lower().startswith("bottom_")]
        case "outerwear_":
            return [item for item in candidates if item["text"].get("category", "").lower().startswith("outerwear_")]
        case "bag_":
            return [item for item in candidates if item["text"].get("category", "").lower().startswith("bag_")]
        case "accessory_":
            return [item for item in candidates if item["text"].get("category", "").lower().startswith("accessory_")]
        case "footwear_":
            return [item for item in candidates if item["text"].get("category", "").lower().startswith("footwear_")]
        case _:
            print(f"?? Unknown category prefix '{category_prefix}', no filtering applied.")
            return candidates


__all__ = [
    "filter_wardrobe_by_score",
    "filter_wardrobe_by_formality",
    "filter_wardrobe_by_attributes",
    "filter_wardrobe_by_category",
]
