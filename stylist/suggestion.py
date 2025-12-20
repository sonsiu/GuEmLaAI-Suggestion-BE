"""
High-level workflow for suggesting outfits from a wardrobe.
"""
import re

from .embeddings import aggregate_multi_query_candidates
from .filters import (
    filter_wardrobe_by_attributes,
    filter_wardrobe_by_category,
    filter_wardrobe_by_score,
)
from .formatter import format_candidates
from .guardrails import isPromptLengthValid, passes_keyword_filters
from .intent import analyze_intent_and_relevance, llm_func_system_user
from .outfits import (
    build_deterministic_outfits,
    parse_outfits_from_llm,
    validate_and_fill_outfits,
)


def suggest_outfit_from_wardrobe(user_query: str, wardrobe_items: list, index):
    ok_length, length_msg = isPromptLengthValid(user_query, 20, 300)
    if not ok_length:
        return {"selected_ids": [], "suggestion": length_msg}

    if not passes_keyword_filters(user_query):
        return {"selected_ids": [], "suggestion": "I don't have an answer for this question. Please ask questions about outfit suggestion."}

    relevant, intent_payload = analyze_intent_and_relevance(user_query)
    if not relevant:
        return {"selected_ids": [], "suggestion": "I don't have an answer for this question. Please ask questions about outfit suggestion."}
    intent_payload = intent_payload or {}

    print(f"dY\"? User query: {user_query}\n")
    print(f"[STEP 1] Analyze intent for better embedding...")
    print(intent_payload)
    print("\n")

    print(f"[STEP 2] Local embedding of query variants...\n")
    query_variants = [
        user_query,
        intent_payload.get("rewrite_en", ""),
        intent_payload.get("rewrite_vi", ""),
        intent_payload.get("keywords", ""),
    ]
    candidates = aggregate_multi_query_candidates(
        query_variants, index, wardrobe_items
    )

    print(f"[STEP 3] Local FAISS search merged across {len([q for q in query_variants if q.strip()])} query views...")

    candidates = filter_wardrobe_by_attributes(intent_payload.get("attributes"), candidates)
    candidates = filter_wardrobe_by_score(candidates)

    for i, c in enumerate(candidates[:25], 1):
        print(f" {i:2d}. (score={c['score']:.4f}) {c['text']}")

    top_candidates = filter_wardrobe_by_category("top_", candidates)
    bottom_candidates = filter_wardrobe_by_category("bottom_", candidates)
    outerwear_candidates = filter_wardrobe_by_category("outerwear_", candidates)
    bag_candidates = filter_wardrobe_by_category("bag_", candidates)
    accessory_candidates = filter_wardrobe_by_category("accessory_", candidates)
    footwear_candidates = filter_wardrobe_by_category("footwear_", candidates)

    print(f"\n[STEP 4] Preparing LLM prompt...")

    context_sections = [
        format_candidates("Tops", top_candidates),
        format_candidates("Bottoms", bottom_candidates),
        format_candidates("Outerwear", outerwear_candidates),
        format_candidates("Bags", bag_candidates),
        format_candidates("Accessories", accessory_candidates),
        format_candidates("Footwear", footwear_candidates),
    ]
    context_block = "\n".join(context_sections)

    system_prompt = "You are a friendly personal stylist. Use the provided wardrobe items to recommend suitable outfits."
    user_prompt = (
        f"Below are the wardrobe categories and the most relevant items for each category:\n"
        f"{context_block}\n\n"
        f'User asked: "{user_query}"\n\n'
        "Choose 2 distinct outfits. Each outfit must be returned as a single array of integer item IDs (for example: [3, 7, 12, 15]). Follow these rules exactly:\n"
        "- Each outfit MUST include at minimum: 1 bottom (pants or skirt), 1 top (shirt, blouse, etc.), and 1 footwear (shoes, sandals, boots).\n"
        "- Prefer a full 4-item outfit when possible: top + bottom + footwear + outerwear (jacket/coat) OR accessory OR bag. The 4th item only choose one of outerwear OR accessory OR bag.\n"
        "- If a full 4-item outfit is not possible (missing relevant items), return the minimal valid outfit (at least top, bottom, footwear).\n"
        "- Do NOT include duplicate categories in the same outfit (for example: do not include two bottoms or two footwear items in one outfit).\n"
        "- Choose items from the provided lists with highest relevance for the user's request. If an exact category is missing, choose the closest matching item.\n"
        "- The two outfits must be meaningfully different (different style, formality, or variant) while both satisfying the user's request.\n"
        "- Return ONLY the two arrays, one per line, with no explanations, no extra text, and no additional characters.\n\n"
        "Required output format:\n"
        "[array_of_ids_for_outfit_1]\n"
        "[array_of_ids_for_outfit_2]\n\n"
        "Example:\n"
        "[3, 7, 12, 15]\n"
        "[2, 6, 9, 14]"
    )

    suggestion = ""
    suggestion = llm_func_system_user(system_prompt, user_prompt, max_tokens=500, temperature=0.6)
    print(user_prompt)

    print("\n" + "="*80)
    print("?o\" LLM Outfit Suggestion ?o\"")
    print("="*80)
    print(suggestion)
    print("="*80)

    parsed = parse_outfits_from_llm(suggestion)

    candidates_by_cat = {
        "top": top_candidates,
        "bottom": bottom_candidates,
        "outerwear": outerwear_candidates,
        "accessory": accessory_candidates + bag_candidates,
        "footwear": footwear_candidates,
    }

    validated = validate_and_fill_outfits(parsed, wardrobe_items, candidates_by_cat) if parsed else []

    if not parsed or not any(validated):
        print("?s??,? LLM output invalid or incomplete ?? using deterministic fallback to build outfits.")
        fallback = build_deterministic_outfits(wardrobe_items, candidates_by_cat)
        selected_outfits = [o for o in fallback if o]
    else:
        selected_outfits = []
        fallback = build_deterministic_outfits(wardrobe_items, candidates_by_cat)
        fb_iter = iter(fallback)
        for v in validated:
            if v:
                selected_outfits.append(v)
            else:
                selected_outfits.append(next(fb_iter, []))

    if len(selected_outfits) < 2:
        more = build_deterministic_outfits(wardrobe_items, candidates_by_cat)
        for o in more:
            if o and len(selected_outfits) < 2:
                selected_outfits.append(o)

    final_selected = []
    for o in selected_outfits[:2]:
        if not o:
            final_selected.append([])
        else:
            final_selected.append([int(x) for x in o][:4])

    cut_suggestion = re.split(r"\[[^\]]*\]", suggestion)[0].strip()
    return {"selected_ids": final_selected, "suggestion": cut_suggestion}


__all__ = ["suggest_outfit_from_wardrobe"]
