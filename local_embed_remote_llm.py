# lightrag_wardrobe_stylist_v3.py
"""
Wardrobe Stylist - Pure Local Embedding + Reranking, Remote LLM Only for Final Suggestion
"""
import os
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import requests
import typing as t
import torch
import json
import re
import ast
import time;

from category_synonyms import category_synonyms

WHITELIST_TOKENS = [
    "outfit", "clothes", "clothing", "wardrobe", "attire", "wear", "wearing",
    "style", "fashion", "look", "fit",
    "shirt", "tshirt", "tee", "polo", "hoodie", "sweater", "cardigan", "jacket", "coat", "blazer", "vest",
    "pants", "trousers", "jeans", "shorts", "skirt", "dress", "suit", "tie", "bow tie",
    "shoes", "sneakers", "boots", "loafers", "sandals", "heels", "footwear",
    "bag", "belt",
    "formal", "casual", "business", "smart casual", "business casual",
    "wedding", "ceremony", "party", "interview", "presentation", "meeting", "office", "work",
    # Vietnamese tokens
    "mặc", "áo", "quần", "váy", "giày", "dép", "áo sơ mi", "quần tây", "áo khoác", "áo vest", "suit",
    "trang phục", "phong cách", "đi tiệc", "phỏng vấn", "đi chơi", "đi làm", "đi làm việc", "đi họp", "phòng họp",
    "bảo vệ đồ án", "bảo vệ luận văn", "thuyết trình", "thuyết trình khách hàng",
    "tiệc cưới", "dự cưới", "đám cưới", "đồ cưới", "dạ tiệc",
    "dã ngoại", "đi biển", "đi du lịch"
]

BLACKLIST_TOKENS = [
    "http://", "https://", "www.", ".com", ".net", ".org",
    "select ", "insert ", "update ", "delete ", "drop ", "union ",
    "sql ", "ssh ", "ftp ", "sudo", "rm -", "pip ", "npm ", "curl ",
    "password", "bank", "loan", "crypto", "bitcoin", "eth", "server",
    "politics", "election", "president", "prime minister"
]

load_dotenv()

# -----------------------
# Config
# -----------------------
EMBED_MODEL_NAME = "BAAI/bge-m3"
EMBED_MODEL_PATH = "./models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"

USE_CUDA = torch.cuda.is_available() and os.getenv("USE_CUDA", "1") == "1"
os.environ["HF_HUB_OFFLINE"] = "1"
DEVICE = "cuda" if USE_CUDA else "cpu"
TOP_K_EMBED = 25   
TOP_K_RERANK = 7   
QUERY_SCORE_AGGREGATION = "max"  # "max" or "mean" when merging multi-view query scores
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
WORKING_DIR = "./test_gu_em_la_ai"
WARDROBE_API_BASE = os.getenv("WARDROBE_API_BASE", "https://localhost:7016")
API_URL = f"{WARDROBE_API_BASE}/item-descriptions/{{userId}}/raw"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# -----------------------
# Load models LOCALLY
# -----------------------
print("Loading embedding model:", EMBED_MODEL_NAME, "device:", DEVICE)
start_time = time.time()
embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)
# embed_model = SentenceTransformer(EMBED_MODEL_PATH, device=DEVICE)
end_time = time.time()
elapsed = end_time - start_time
print(f"✅ Model loaded successfully in {elapsed:.2f} seconds.")
# -----------------------
# Local Embedding Function (NO LLM)
# -----------------------
def embedding_func(texts: t.Union[str, t.List[str]]) -> np.ndarray:
    if isinstance(texts, str):
        texts = [texts]
    embs = embed_model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=16,
        show_progress_bar=False
    )
    arr = np.array(embs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms
    return arr

def aggregate_multi_query_candidates(
    query_texts: t.List[str],
    index,
    wardrobe_items: list,
    top_k: int = TOP_K_EMBED,
    mode: str = QUERY_SCORE_AGGREGATION,
) -> list:
    """
    Embed multiple query variants and merge their FAISS scores.
    mode: "max" keeps the best score per item across queries; "mean" averages them.
    """
    if not query_texts:
        return []

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for qt in query_texts:
        key = qt.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(qt)

    if not deduped:
        return []

    q_embs = embedding_func(deduped).astype("float32")
    combined: dict[int, dict] = {}

    for q_emb in q_embs:
        D, I = index.search(q_emb.reshape(1, -1), k=top_k)
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(wardrobe_items):
                continue
            item = wardrobe_items[idx]
            entry = combined.setdefault(
                idx, {"id": item.get("id"), "scores": [], "text": item}
            )
            entry["scores"].append(float(score))

    results = []
    for entry in combined.values():
        scores = entry["scores"] or [0.0]
        if mode == "mean":
            agg_score = float(sum(scores) / len(scores))
        else:
            agg_score = float(max(scores))
        results.append({"id": entry["id"], "score": agg_score, "text": entry["text"]})

    results.sort(key=lambda x: x["score"], reverse=True)
    return results

# -----------------------
# Build FAISS Index
# -----------------------
def build_faiss_index(wardrobe_items, working_dir=WORKING_DIR):
    index_file = os.path.join(WORKING_DIR, "wardrobe_faiss.index")
    embeddings_file = os.path.join(WORKING_DIR, "wardrobe_embeddings.npy")

    # with open("wardrobe.json", "r", encoding="utf-8") as f:
    #         data = json.load(f)
    # wardrobe_items = data["wardrobe"]

    # Check if indexes changed
    rebuild = True
    if os.path.exists(index_file) and os.path.exists(embeddings_file):
        try:
            existing_embeddings = np.load(embeddings_file)
            if existing_embeddings.shape[0] == len(wardrobe_items):
                rebuild = False
        except Exception:
            rebuild = True

        # ===========================rebuild only for debug reasons, remove after done debugging================================
        wardrobe_texts = normalize_wardrobe_items(wardrobe_items)
        wardrobe_embeddings = embedding_func(wardrobe_texts)
        dim = wardrobe_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(wardrobe_embeddings)
        faiss.write_index(index, index_file)
        np.save(embeddings_file, wardrobe_embeddings)
        # ==========================================================================================    
        
    if not rebuild:
        # os.path.exists(index_file) and os.path.exists(embeddings_file):
        print(f"\n✅ Loading existing FAISS index from {WORKING_DIR}")
        index = faiss.read_index(index_file)
        wardrobe_embeddings = existing_embeddings
        print(f"   Loaded {index.ntotal} wardrobe items (dim={wardrobe_embeddings.shape[1]})")
    else:
        print(f"\n📦 Building FAISS index")

        # Embed human-readable normalized texts (not raw dicts) to get meaningful semantic vectors
        wardrobe_texts = normalize_wardrobe_items(wardrobe_items)
        wardrobe_embeddings = embedding_func(wardrobe_texts)
        dim = wardrobe_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(wardrobe_embeddings)

        # Save for future use
        faiss.write_index(index, index_file)
        np.save(embeddings_file, wardrobe_embeddings)
        print(f"   ✅ Indexed {index.ntotal} wardrobe items (dim={dim})")
        print(f"   💾 Saved to {WORKING_DIR}")

    return index, wardrobe_embeddings
# -----------------------
# LLM Function
# -----------------------
def llm_call_openrouter(messages: list, model="gpt-4o-mini", max_tokens=400, temperature=0.7) -> str:
    key = OPENROUTER_KEY
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
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
        "bảo vệ", "luận văn", "đồ án", "thesis", "defense", "presentation",
        "interview", "phỏng vấn", "meeting", "hội nghị", "conference",
        "client", "ceremony", "đám cưới", "graduation", "kỷ yếu", "pitch"
    ]
    casual_signals = [
        "đi chơi", "hang out", "picnic", "du lịch", "travel", "barbecue",
        "bbq", "cafe", "cà phê", "thoải mái", "casual", "dã ngoại"
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
        # Fallback: treat as relevant and reuse the raw query to avoid blocking the flow
        return True, {
            "rewrite_en": user_query.strip(),
            "rewrite_vi": "",
            "keywords": "",
            "attributes": {"formality": infer_formality(user_query, {}), "season": [], "occasion": ""},
        }

# -----------------------
# Main Workflow
# -----------------------
def suggest_outfit_from_wardrobe(user_query: str, wardrobe_items: list, index):

    ok_length, length_msg = isPromptLengthValid(user_query, 20, 300)
    if not ok_length:
        return {"selected_ids": [], "suggestion": length_msg}

    if not passes_keyword_filters(user_query):
        return {"selected_ids": [], "suggestion": "I don't have an answer for this question. Please ask questions about outfit suggestion."}

    # STEP 1: LLM guardrail + normalization of intent
    relevant, intent_payload = analyze_intent_and_relevance(user_query)
    if not relevant:
        return {"selected_ids": [], "suggestion": "I don't have an answer for this question. Please ask questions about outfit suggestion."}
    intent_payload = intent_payload or {}

    print(f"dY\"? User query: {user_query}\n")
    print(f"[STEP 1] Analyze intent for better embedding...")
    print(intent_payload)
    print("\n")
    
    # STEP 2: Multi-view embedding of queries (raw + EN rewrite + VI rewrite + keywords)
    print(f"[STEP 2] Local embedding of query variants...\n")
    query_variants = [
        user_query,
        intent_payload.get("rewrite_en", ""),
        intent_payload.get("rewrite_vi", ""),
        intent_payload.get("keywords", ""),
    ]
    candidates = aggregate_multi_query_candidates(
        query_variants, index, wardrobe_items, top_k=TOP_K_EMBED, mode=QUERY_SCORE_AGGREGATION
    )

    print(f"[STEP 3] Local FAISS search merged across {len([q for q in query_variants if q.strip()])} query views...")

    # =============filter wardrobe by parsed attributes====================================================
    candidates = filter_wardrobe_by_attributes(intent_payload.get("attributes"), candidates)

    # =============filter wardrobe by score================================================================
    candidates = filter_wardrobe_by_score(candidates)
    
    for i, c in enumerate(candidates[:TOP_K_EMBED], 1):
        print(f" {i:2d}. (score={c['score']:.4f}) {c['text']}")
    
    # =============filter wardrobe by category============================================================
    top_candidates = filter_wardrobe_by_category("top_", candidates)
    bottom_candidates = filter_wardrobe_by_category("bottom_", candidates)
    outerwear_candidates = filter_wardrobe_by_category("outerwear_", candidates)
    bag_candidates = filter_wardrobe_by_category("bag_", candidates)
    accessory_candidates = filter_wardrobe_by_category("accessory_", candidates)
    footwear_candidates = filter_wardrobe_by_category("footwear_", candidates)
    # =====================================================================================================

    # STEP 4: Build context for LLM
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
    
    # STEP 5: ONLY NOW call LLM (single API call)
    suggestion = ""
    suggestion = llm_func_system_user(system_prompt, user_prompt, max_tokens=500, temperature=0.6)
    print(user_prompt)

    print("\n" + "="*80)
    print("?o\" LLM Outfit Suggestion ?o\"")
    print("="*80)
    print(suggestion)
    print("="*80)
    
    # Parse LLM response safely and validate/fill outfits. If validation fails, build deterministic fallback outfits.
    parsed = parse_outfits_from_llm(suggestion)

    candidates_by_cat = {
        "top": top_candidates,
        "bottom": bottom_candidates,
        "outerwear": outerwear_candidates,
        "accessory": accessory_candidates + bag_candidates,
        "footwear": footwear_candidates,
    }

    validated = validate_and_fill_outfits(parsed, wardrobe_items, candidates_by_cat) if parsed else []

    # If parsed result is missing or invalid, use deterministic fallback
    if not parsed or not any(validated):
        print("?s??,? LLM output invalid or incomplete ?? using deterministic fallback to build outfits.")
        fallback = build_deterministic_outfits(wardrobe_items, candidates_by_cat)
        selected_outfits = [o for o in fallback if o]
    else:
        # Use validated outfits, but if any outfit couldn't be fixed (empty), replace with fallback
        selected_outfits = []
        fallback = build_deterministic_outfits(wardrobe_items, candidates_by_cat)
        fb_iter = iter(fallback)
        for v in validated:
            if v:
                selected_outfits.append(v)
            else:
                # take next fallback
                selected_outfits.append(next(fb_iter, []))

    # Ensure we return exactly two outfits
    if len(selected_outfits) < 2:
        more = build_deterministic_outfits(wardrobe_items, candidates_by_cat)
        for o in more:
            if o and len(selected_outfits) < 2:
                selected_outfits.append(o)

    # Final normalization: ensure list of ints and trim to 4 items each
    final_selected = []
    for o in selected_outfits[:2]:
        if not o:
            final_selected.append([])
        else:
            final_selected.append([int(x) for x in o][:4])

    # For backward compatibility, also provide the raw bracket strings if available
    cut_suggestion = re.split(r"\[[^\]]*\]", suggestion)[0].strip()
    return {"selected_ids": final_selected, "suggestion": cut_suggestion}


def load_wardrobe_for_user(user_id: str):
    """
    Dynamically fetch wardrobe JSON for the given user_id from the C# API.
    Falls back to local cache if unavailable.
    """
    url = API_URL.format(userId=user_id)
    print(f"📡 Fetching wardrobe data for user {user_id} from {url} ...")

    try:
        response = requests.get(url, verify=False, timeout=10)
        response.raise_for_status()
        data = response.json()
        wardrobe_items = data["wardrobe"]

        # cache locally
        with open(f"wardrobe_{user_id}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Loaded {len(wardrobe_items)} wardrobe items from API.")
        return wardrobe_items

    except requests.RequestException as e:
        print(f"❌ Failed to fetch wardrobe from API: {e}")
        # fallback to local cache
        # cache_file = f"wardrobe_{user_id}.json"
        cache_file = f"wardrobe_{user_id}.json"
        if os.path.exists(cache_file):
            print(f"⚠️ Loading cached wardrobe for user {user_id}")
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data["wardrobe"]
        else:
            raise RuntimeError(f"No wardrobe data available for user {user_id}")
        
def isPromptRelevant(user_query: str, wardrobe_items: list, index) -> bool:
    q_emb = embedding_func(user_query).astype("float32")

    D, I = index.search(q_emb, k=1)  # only need the top 1
    top_score = float(D[0][0])
    top_idx = int(I[0][0])

    item = wardrobe_items[top_idx]
    if isinstance(item, dict):
        # Safely construct a readable string for debugging/logging
        desc = item.get("category", "")
        if item.get("color"):
            desc += f" ({', '.join(item['color'])})"
        if item.get("season"):
            desc += f" - for {', '.join(item['season'])}"
        top_item = desc.strip()
    else:
        top_item = str(item)

    # Print debug info
    print(f"{top_score:.4f}: {top_item}")

    # If the top score is lower than threshold (e.g., 0.5), it's likely irrelevant
    threshold = 0.45
    return top_score >= threshold
def isPromptLengthValid(user_query: str, minLength: int, maxLength: int = 500) -> t.Tuple[bool, str]:
    """
    Enforce minimum detail and cap overly long prompts to avoid injection/noise.
    Returns (ok, message_if_not_ok).
    """
    text = user_query.strip()
    if len(text) < minLength:
        return False, "Please provide a more detailed request—for example, mention the occasion, season, or style you want."
    if len(text) > maxLength:
        return False, f"Your request is too long. Please shorten it to under {maxLength} characters so I can help accurately."
    return True, ""

def passes_keyword_filters(text: str) -> bool:
    """Deterministic guard: block obvious off-topic; allow if any fashion token appears."""
    t = text.lower()
    if any(b in t for b in BLACKLIST_TOKENS):
        return False
    if any(w in t for w in WHITELIST_TOKENS):
        return True
    return False

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

def filter_wardrobe_by_score(candidates: list):
    # Remove any candidates that have a negative similarity score.
    # Candidates are expected in the format: {"id": ..., "score": float, "text": ...}
    before = len(candidates or [])
    filtered = []
    for c in (candidates or []):
        try:
            score = float(c.get("score", 0))
        except Exception:
            # If score cannot be parsed, keep the candidate (safer)
            filtered.append(c)
            continue
        if score >= 0:
            filtered.append(c)

    removed = before - len(filtered)
    if removed > 0:
        print(f"🧾 Filtered out {removed} candidates with score < 0")
    return filtered

def filter_wardrobe_by_formality(user_query: str, candidates: list):
    """
    Filters out casual items if user query mentions 'formal'.
    You can expand this logic for other styles later (e.g., sporty, elegant, etc.).
    """
    normalized_query = user_query.lower().strip()

    # Case 1: User wants formal style → remove casual
    if "formal" in normalized_query:
        filtered = [item for item in candidates if item["text"].get("purpose", "").lower() != "casual"]
        print(f"🧹 Filtered out {len(candidates) - len(filtered)} casual items (formal query detected).")
        return filtered
    
     # Case 2: User wants casual style → remove formal
    if "casual" in normalized_query:
        filtered = [item for item in candidates if item["text"].get("purpose", "").lower() != "formal"]
        print(f"🧹 Filtered out {len(candidates) - len(filtered)} formal items (casual query detected).")
        return filtered

    # No explicit 'formal' or 'casual' in query: return the original candidates list
    return candidates

def filter_wardrobe_by_attributes(attributes: dict, candidates: list):
    """
    Uses parsed attributes (formality/season/occasion) instead of plain string matching.
    Applies a soft filter for season and a hard filter for conflicting formality.
    Rules:
      - If user wants casual => drop items tagged formal
      - If user wants formal => drop items tagged casual
      - Otherwise keep all purposes
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

        # Hard filter on formality conflicts per wardrobe purpose tag
        if formality.startswith("formal") and purpose == "casual":
            continue
        if formality.startswith("casual") and purpose == "formal":
            continue

        # Soft preference on season
        item_seasons = [s.lower() for s in (item.get("season") or []) if isinstance(s, str)]
        bonus = 0.0
        if season_pref:
            if item_seasons and any(s in item_seasons for s in season_pref):
                bonus += 0.05
            elif item_seasons:
                bonus -= 0.05

        # Simple occasion nudge (keep minimal to avoid over-filtering)
        if occasion:
            desc = (item.get("desc") or "").lower()
            if occasion in desc:
                bonus += 0.02

        filtered.append({**c, "score": score + bonus})

    # Avoid empty results if filters were too strict
    return filtered if filtered else candidates

def filter_wardrobe_by_category(category_prefix: str, candidates: list):
    """
    Filters out casual items if user query mentions 'formal'.
    You can expand this logic for other styles later (e.g., sporty, elegant, etc.).
    """

    match category_prefix:
        case "top_":
            filtered = [item for item in candidates if item["text"].get("category", "").lower().startswith("top_")]
            return filtered

        case "bottom_":
            filtered = [item for item in candidates if item["text"].get("category", "").lower().startswith("bottom_")]
            return filtered

        case "outerwear_":
            filtered = [item for item in candidates if item["text"].get("category", "").lower().startswith("outerwear_")]
            return filtered

        case "bag_":
            filtered = [item for item in candidates if item["text"].get("category", "").lower().startswith("bag_")]
            return filtered

        case "accessory_":
            filtered = [item for item in candidates if item["text"].get("category", "").lower().startswith("accessory_")]
            return filtered

        case "footwear_":
            filtered = [item for item in candidates if item["text"].get("category", "").lower().startswith("footwear_")]
            return filtered

        case _:
            print(f"⚠️ Unknown category prefix '{category_prefix}', no filtering applied.")
            return candidates
        
def format_candidates(title, items):
    if not items:
        return f"{title}:\n  (Không có món phù hợp)\n"
    lines = [f"  {c['text']} (score={c['score']:.4f})" for c in items]
    return f"{title}:\n" + "\n".join(lines) + "\n"


def _get_item_by_id(wardrobe_items: list, item_id: int):
    for it in wardrobe_items:
        try:
            if int(it.get("id")) == int(item_id):
                return it
        except Exception:
            continue
    return None


def _get_category_tag(item: dict) -> t.Optional[str]:
    """Map item['category'] to one of: 'top','bottom','footwear','outerwear','accessory' or None"""
    if not isinstance(item, dict):
        return None
    cat = item.get("category", "").lower()
    if cat.startswith("top_"):
        return "top"
    if cat.startswith("bottom_"):
        return "bottom"
    if cat.startswith("footwear_"):
        return "footwear"
    if cat.startswith("outerwear_"):
        return "outerwear"
    if cat.startswith("bag_") or cat.startswith("accessory_"):
        return "accessory"
    return None


def parse_outfits_from_llm(suggestion: str) -> list:
    """Safely parse bracketed arrays from LLM suggestion into list of int lists."""
    arrays = re.findall(r"\[[^\]]*\]", suggestion)
    parsed = []
    for a in arrays:
        try:
            val = ast.literal_eval(a)
            if isinstance(val, (list, tuple)):
                ints = [int(x) for x in val if isinstance(x, (int, float)) or (isinstance(x, str) and x.strip().isdigit())]
                parsed.append(ints)
        except Exception:
            continue
    return parsed


def validate_and_fill_outfits(outfits: list, wardrobe_items: list, candidates_by_cat: dict) -> list:
    """Ensure each outfit has at least top, bottom, footwear. Prefer 4-item outfits by adding outerwear or accessory when possible.
    outfits: list of lists of ids (ints)
    candidates_by_cat: dict with keys 'top','bottom','footwear','outerwear','accessory' -> list of candidate dicts with 'id'
    Returns a list of validated/fixed outfits (each a list of ints)."""
    # Precompute score lookups to break ties when multiple items share a category
    score_lookup = {}
    for cat, cand_list in candidates_by_cat.items():
        for cand in cand_list or []:
            try:
                cid = int(cand.get("id"))
                score = float(cand.get("score", 0))
            except Exception:
                continue
            score_lookup.setdefault(cat, {})[cid] = score

    fixed = []
    for outfit in outfits:
        # remove duplicates while preserving order
        seen = set()
        uniq = []
        for oid in outfit:
            try:
                iid = int(oid)
            except Exception:
                continue
            if iid in seen:
                continue
            seen.add(iid)
            uniq.append(iid)

        # map present categories, keeping the best (highest score) when duplicates appear
        present = {"top": None, "bottom": None, "footwear": None, "outerwear": None, "accessory": None}
        for iid in uniq:
            item = _get_item_by_id(wardrobe_items, iid)
            cat = _get_category_tag(item) if item else None
            if not cat:
                continue
            current = present.get(cat)
            if current is None:
                present[cat] = iid
            else:
                # Replace if this item has a better similarity score
                current_score = score_lookup.get(cat, {}).get(current, float("-inf"))
                new_score = score_lookup.get(cat, {}).get(iid, float("-inf"))
                if new_score > current_score:
                    present[cat] = iid

        # Fill missing required categories in order: bottom, top, footwear
        for req in ["bottom", "top", "footwear"]:
            if present[req] is None:
                for cand in candidates_by_cat.get(req, []):
                    try:
                        cid = int(cand.get("id"))
                    except Exception:
                        continue
                    if cid in present.values():
                        continue
                    if _get_item_by_id(wardrobe_items, cid):
                        present[req] = cid
                        break

        # Prefer to reach 4 items by adding outerwear then accessory, no duplicate categories
        for opt in ["outerwear", "accessory"]:
            if len([v for v in present.values() if v]) >= 4:
                break
            if present[opt] is None:
                for cand in candidates_by_cat.get(opt, []):
                    try:
                        cid = int(cand.get("id"))
                    except Exception:
                        continue
                    if cid in present.values():
                        continue
                    if _get_item_by_id(wardrobe_items, cid):
                        present[opt] = cid
                        break

        # Final check: must have required categories, and enforce single item per category
        if present["top"] and present["bottom"] and present["footwear"]:
            ordered = []
            for cat in ["top", "bottom", "footwear", "outerwear", "accessory"]:
                if present.get(cat):
                    ordered.append(present[cat])
            fixed.append(ordered[:4])
        else:
            fixed.append([])

    return fixed


def build_deterministic_outfits(wardrobe_items: list, candidates_by_cat: dict) -> list:
    """Build two deterministic outfits from top candidates as a fallback.
    Returns two outfits (lists of ints)."""
    def pick(cats, used):
        out = []
        for cat in cats:
            for cand in candidates_by_cat.get(cat, []):
                try:
                    cid = int(cand.get("id"))
                except Exception:
                    continue
                if cid not in used and _get_item_by_id(wardrobe_items, cid):
                    out.append(cid)
                    used.add(cid)
                    break
        return out

    # Build outfit A: top, bottom, footwear, outerwear, accessory
    used = set()
    outfit_a = pick(["top", "bottom", "footwear", "outerwear", "accessory"], used)
    # Build outfit B: pick next-best top/bottom/footwear and accessory
    outfit_b = pick(["top", "bottom", "footwear", "accessory", "outerwear"], used)

    # Ensure minimal validity; if any outfit missing required, try to fill from any category
    def ensure_minimal(outfit, used):
        present_cats = set()
        for iid in outfit:
            item = _get_item_by_id(wardrobe_items, iid)
            cat = _get_category_tag(item)
            if cat:
                present_cats.add(cat)
        for req in ["top", "bottom", "footwear"]:
            # Only try to fill required category if candidates exist for that category
            if req not in present_cats and candidates_by_cat.get(req):
                for cand in candidates_by_cat.get(req, []):
                    try:
                        cid = int(cand.get("id"))
                    except Exception:
                        continue
                    if cid not in used and _get_item_by_id(wardrobe_items, cid):
                        outfit.append(cid)
                        used.add(cid)
                        present_cats.add(req)
                        break
            # stop if we already reached 4 items
            if len(outfit) >= 4:
                return outfit
        return outfit

    outfit_a = ensure_minimal(outfit_a, used)
    outfit_b = ensure_minimal(outfit_b, used)

    return [outfit_a, outfit_b]
# -----------------------
# Example Usage
# -----------------------
if __name__ == "__main__":
    print("🎯 Wardrobe Stylist - Local Embed/Rerank, Remote LLM Only")
    print("="*80)
    
    try:
        # Example queries
        queries = [
            # "helo",
            # "thịt chó có ngon như người ta bảo không",
            "ngày mai tôi đi bảo vệ đồ án, tôi nên mặc gì?",
            # "Tôi nên mặc gì đến buổi phỏng vấn quan trọng của tôi?",
            # "Tôi nên mặc gì đi uống cà phê với bạn?",
            # "Tôi nên mặc gì đến buổi BBQ ngoài trời của bạn tôi?",
        ]
        
        wardrobe_items = load_wardrobe_for_user("21")
        # wardrobe_items_texts = normalize_wardrobe_items(wardrobe_items)

        for query in queries:
            # wardrobe_items = filter_wardrobe_by_formality(query, wardrobe_items)
            # print("NORMALIZED WARDROBE ITEMS TEXTS:", wardrobe_items_texts)
            index, _ = build_faiss_index(wardrobe_items)
            result = suggest_outfit_from_wardrobe(query, wardrobe_items, index)
            print("\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


