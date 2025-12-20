"""
LLM helpers, intent parsing, and lightweight formality inference.
"""
import json
import typing as t
import requests

from .config import OPENROUTER_KEY
from .config import QUERY_SCORE_AGGREGATION  # for re-export compatibility
from .config import TOP_K_EMBED  # for re-export compatibility
from .config import TOP_K_RERANK  # for re-export compatibility
from .config import DEVICE  # for re-export compatibility
from .config import EMBED_MODEL_NAME, EMBED_MODEL_PATH  # for re-export compatibility
from .config import WORKING_DIR, WARDROBE_API_BASE, API_URL  # re-export compatibility
from .config import USE_CUDA  # re-export compatibility
from .config import WHITELIST_TOKENS, BLACKLIST_TOKENS  # re-export compatibility
from .config import ensure_working_dir  # re-export compatibility


def llm_call_openrouter(messages: list, model="gpt-4o-mini", max_tokens=400, temperature=0.7) -> str:
    key = OPENROUTER_KEY
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if not r.ok:
        print(f"OpenRouter response status={r.status_code}, body={r.text}")
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def llm_func_system_user(system_prompt: str, user_prompt: str, model="gpt-4o-mini", max_tokens=500, temperature=0.7) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    if OPENROUTER_KEY:
        return llm_call_openrouter(messages, model=model, max_tokens=max_tokens, temperature=temperature)
    else:
        raise RuntimeError("No LLM API key found. Set OPENROUTER_API_KEY in environment or .env")


def infer_formality(user_query: str, parsed_attributes: dict) -> str:
    """
    Lightweight, deterministic formality classifier to reduce prompt load on the LLM.
    Uses existing attribute hints first, then falls back to keyword cues.
    """
    if not isinstance(parsed_attributes, dict):
        parsed_attributes = {}

    existing = str(parsed_attributes.get("formality", "")).lower().strip()
    if existing in {"formal", "casual"}:
        return existing

    q = user_query.lower()
    formal_signals = [
        "bảo vệ", "luận văn", "dự án", "thesis", "defense", "presentation", "interview", "phỏng vấn", "meeting", "hội nghị", "conference", "client", "ceremony", "đám cưới", "graduation", "kỷ yếu", "pitch",
    ]
    casual_signals = [
        "đi chơi", "hang out", "picnic", "du lịch", "travel", "barbecue", "bbq", "cafe", "cà phê", "thoải mái", "casual", "dã ngoại",
    ]

    if any(token in q for token in formal_signals):
        return "formal"
    if any(token in q for token in casual_signals):
        return "casual"
    return "any"


def analyze_intent_and_relevance(user_query: str) -> t.Tuple[bool, dict]:
    """
    Guardrail + richer intent parsing.
    Returns (relevant, intent_payload) where payload contains:
      - rewrite_en: English rewrite of the intent
      - rewrite_vi: Vietnamese rewrite of the intent
      - keywords: compressed keyword-only intent
      - attributes: dict with formality/season/occasion (best-effort)
    """
    system_prompt = (
        "You are a guardrail and intent normalizer for a wardrobe stylist app. "
        "Decide if the query is about clothing/outfits/personal styling. "
        "If relevant, produce:\n"
        "- rewrite_en: one short English sentence capturing occasion, season/weather, and style tone.\n"
        "- rewrite_vi: the same, but in Vietnamese.\n"
        "- keywords: only key terms (comma separated), no filler words.\n"
        "- attributes: JSON with season (array like [\"summer\"], empty if unknown) and occasion (short string). Leave formality empty; it will be inferred separately.\n"
        "Respond ONLY with JSON. If not relevant, set relevant=false and keep other fields minimal/empty."
    )
    user_prompt = (
        f'User query: "{user_query}"\n'
        "Return JSON exactly. Example:\n"
        '{\"relevant\": true, \"rewrite_en\": \"Smart summer outfit for an office day, breathable fabrics.\", \"rewrite_vi\": \"Trang phuc gọn gàng cho ngày hè đi làm, vải thoáng mát.\", \"keywords\": \"summer, office, smart, breathable\", \"attributes\": {\"formality\": \"\", \"season\": [\"summer\"], \"occasion\": \"office\"}}\n'
        '{\"relevant\": false, \"rewrite_en\": \"\", \"rewrite_vi\": \"\", \"keywords\": \"\", \"attributes\": {\"formality\": \"\", \"season\": [], \"occasion\": \"\"}}'
    )
    try:
        raw = llm_func_system_user(system_prompt, user_prompt, max_tokens=220, temperature=0.2)
        raw_clean = raw.strip().strip("`").strip()
        parsed = json.loads(raw_clean)
        relevant = bool(parsed.get("relevant"))
        if not relevant:
            return False, {}

        payload = {
            "rewrite_en": (parsed.get("rewrite_en") or "").strip(),
            "rewrite_vi": (parsed.get("rewrite_vi") or "").strip(),
            "keywords": (parsed.get("keywords") or "").strip(),
            "attributes": parsed.get("attributes") or {},
        }
        attrs = payload["attributes"] or {}
        attrs["formality"] = infer_formality(user_query, attrs)
        payload["attributes"] = attrs
        return True, payload
    except Exception as e:
        print(f"Guardrail intent LLM failed to parse: {e}")
        return True, {
            "rewrite_en": user_query.strip(),
            "rewrite_vi": "",
            "keywords": "",
            "attributes": {"formality": infer_formality(user_query, {}), "season": [], "occasion": ""},
        }


__all__ = [
    "llm_call_openrouter",
    "llm_func_system_user",
    "infer_formality",
    "analyze_intent_and_relevance",
    "OPENROUTER_KEY",
    "QUERY_SCORE_AGGREGATION",
    "TOP_K_EMBED",
    "TOP_K_RERANK",
    "DEVICE",
    "EMBED_MODEL_NAME",
    "EMBED_MODEL_PATH",
    "WORKING_DIR",
    "WARDROBE_API_BASE",
    "API_URL",
    "USE_CUDA",
    "WHITELIST_TOKENS",
    "BLACKLIST_TOKENS",
    "ensure_working_dir",
]
