"""
Deterministic guardrails and prompt validation utilities.
"""
import typing as t

from .config import BLACKLIST_TOKENS, WHITELIST_TOKENS
from .embeddings import embedding_func


def isPromptLengthValid(user_query: str, minLength: int, maxLength: int = 500) -> t.Tuple[bool, str]:
    """
    Enforce minimum detail and cap overly long prompts to avoid injection/noise.
    Returns (ok, message_if_not_ok).
    """
    text = user_query.strip()
    if len(text) < minLength:
        return False, "Please provide a more detailed request-for example, mention the occasion, season, or style you want."
    if len(text) > maxLength:
        return False, f"Your request is too long. Please shorten it to under {maxLength} characters so I can help accurately."
    return True, ""


def passes_keyword_filters(text: str) -> bool:
    """Deterministic guard: block obvious off-topic; allow if any fashion token appears."""
    t_lower = text.lower()
    if any(b in t_lower for b in BLACKLIST_TOKENS):
        return False
    if any(w in t_lower for w in WHITELIST_TOKENS):
        return True
    return False


def isPromptRelevant(user_query: str, wardrobe_items: list, index) -> bool:
    q_emb = embedding_func(user_query).astype("float32")

    D, I = index.search(q_emb, k=1)
    top_score = float(D[0][0])
    top_idx = int(I[0][0])

    item = wardrobe_items[top_idx]
    if isinstance(item, dict):
        desc = item.get("category", "")
        if item.get("color"):
            desc += f" ({', '.join(item['color'])})"
        if item.get("season"):
            desc += f" - for {', '.join(item['season'])}"
        top_item = desc.strip()
    else:
        top_item = str(item)

    print(f"{top_score:.4f}: {top_item}")

    threshold = 0.45
    return top_score >= threshold


__all__ = ["isPromptLengthValid", "passes_keyword_filters", "isPromptRelevant"]
