"""
Outfit parsing, validation, and deterministic fallbacks.
"""
import ast
import re
import typing as t


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
    """Ensure each outfit has at least top, bottom, footwear. Prefer 4-item outfits by adding outerwear or accessory when possible."""
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
                current_score = score_lookup.get(cat, {}).get(current, float("-inf"))
                new_score = score_lookup.get(cat, {}).get(iid, float("-inf"))
                if new_score > current_score:
                    present[cat] = iid

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
    """Build two deterministic outfits from top candidates as a fallback."""
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

    used = set()
    outfit_a = pick(["top", "bottom", "footwear", "outerwear", "accessory"], used)
    outfit_b = pick(["top", "bottom", "footwear", "accessory", "outerwear"], used)

    def ensure_minimal(outfit, used_set):
        present_cats = set()
        for iid in outfit:
            item = _get_item_by_id(wardrobe_items, iid)
            cat = _get_category_tag(item)
            if cat:
                present_cats.add(cat)
        for req in ["top", "bottom", "footwear"]:
            if req not in present_cats and candidates_by_cat.get(req):
                for cand in candidates_by_cat.get(req, []):
                    try:
                        cid = int(cand.get("id"))
                    except Exception:
                        continue
                    if cid not in used_set and _get_item_by_id(wardrobe_items, cid):
                        outfit.append(cid)
                        used_set.add(cid)
                        present_cats.add(req)
                        break
            if len(outfit) >= 4:
                return outfit
        return outfit

    outfit_a = ensure_minimal(outfit_a, used)
    outfit_b = ensure_minimal(outfit_b, used)

    return [outfit_a, outfit_b]


__all__ = [
    "parse_outfits_from_llm",
    "validate_and_fill_outfits",
    "build_deterministic_outfits",
]
