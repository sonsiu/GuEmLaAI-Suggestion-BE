"""
Wardrobe data loading and normalization utilities.
"""
import json
import os
import requests

from category_synonyms import category_synonyms
from .config import API_URL


def normalize_wardrobe_items(wardrobe_items):
    """
    Build richer, bilingual descriptions for better semantic recall.
    Includes purpose, desc, season/weather hints, and category synonyms.
    """
    normalized = []
    for item in wardrobe_items:
        if not isinstance(item, dict):
            normalized.append(str(item))
            continue

        cat = item.get("category", "")
        syns = category_synonyms(cat)
        size = item.get("size", "")
        colors = ", ".join(item.get("color", []))
        seasons = ", ".join(item.get("season", []))
        purpose = item.get("purpose", "")
        desc = item.get("desc", "")

        purpose_tags = []
        if purpose == "formal":
            purpose_tags = ["formal", "smart", "business", "office", "trang trong"]
        elif purpose == "casual":
            purpose_tags = ["casual", "everyday", "relaxed", "thoai mai"]
        elif purpose == "all_purpose":
            purpose_tags = ["versatile", "multi-occasion", "linh hoat"]

        season_tags = []
        lower_seasons = seasons.lower()
        if "summer" in lower_seasons or "spring" in lower_seasons:
            season_tags += ["warm weather", "hot", "mua he", "mua xuan"]
        if "winter" in lower_seasons or "autumn" in lower_seasons:
            season_tags += ["cool weather", "cold", "mua dong", "mua thu"]
        if "all year" in lower_seasons:
            season_tags += ["all seasons", "quanh nam"]

        text = (
            f"id: {item.get('id', '')}; "
            f"{syns}; "
            f"size {size}; "
            f"colors: {colors}; "
            f"season: {seasons}; "
            f"purpose: {purpose} {' '.join(purpose_tags)}; "
            f"desc: {desc}"
        )
        if season_tags:
            text += f"; weather tags: {' '.join(season_tags)}"

        normalized.append(text.strip())
    return normalized


def load_wardrobe_for_user(user_id: str):
    """
    Dynamically fetch wardrobe JSON for the given user_id from the C# API.
    Falls back to local cache if unavailable.
    """
    url = API_URL.format(userId=user_id)
    print(f"?? Fetching wardrobe data for user {user_id} from {url} ...")

    try:
        response = requests.get(url, verify=False, timeout=10)
        response.raise_for_status()
        data = response.json()
        wardrobe_items = data["wardrobe"]

        with open(f"wardrobe_{user_id}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"? Loaded {len(wardrobe_items)} wardrobe items from API.")
        return wardrobe_items

    except requests.RequestException as e:
        print(f"? Failed to fetch wardrobe from API: {e}")
        cache_file = f"wardrobe_{user_id}.json"
        if os.path.exists(cache_file):
            print(f"?? Loading cached wardrobe for user {user_id}")
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data["wardrobe"]
        else:
            raise RuntimeError(f"No wardrobe data available for user {user_id}")


__all__ = ["normalize_wardrobe_items", "load_wardrobe_for_user"]
