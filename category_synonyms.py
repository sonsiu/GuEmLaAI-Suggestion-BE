# Reusable category synonym expander for enriching wardrobe item text
def category_synonyms(cat: str) -> str:
    cat_lower = (cat or "").lower()
    syns = [cat_lower.replace("_", " ")]

    if "jeans" in cat_lower:
        syns += ["jeans", "denim pants", "casual trousers", "quan jean"]
    if "trousers" in cat_lower or "suit" in cat_lower:
        syns += ["dress pants", "trouser", "quan au", "quan tay"]
    if "short" in cat_lower:
        syns += ["shorts", "quan short"]
    if "t-shirt" in cat_lower or "tee" in cat_lower:
        syns += ["t shirt", "tee", "ao thun"]
    if "polo" in cat_lower:
        syns += ["polo shirt", "ao polo"]
    if "hoodie" in cat_lower:
        syns += ["hoodie", "ao hoodie", "ao khoac ni"]
    if "tank" in cat_lower or "vest" in cat_lower:
        syns += ["tank top", "sleeveless", "ao ba lo"]
    if "long sleeved" in cat_lower or "shirt" in cat_lower:
        syns += ["button down shirt", "dress shirt", "ao so mi"]
    if "cargo" in cat_lower:
        syns += ["cargo pants", "quan tui hop"]
    if "footwear" in cat_lower or "shoe" in cat_lower:
        syns += ["giay", "shoes", "footwear"]
    if "loafer" in cat_lower:
        syns += ["loafers", "dress shoes", "giay tay"]
    if "sneaker" in cat_lower:
        syns += ["sneakers", "running shoes", "giay the thao"]
    if "flip flop" in cat_lower:
        syns += ["flip flops", "sandals", "dep"]
    if "outerwear" in cat_lower or "jacket" in cat_lower:
        syns += ["outerwear", "jacket", "coat", "ao khoac"]
    if "bag" in cat_lower:
        syns += ["bag", "tui xach", "backpack"]
    if "accessory" in cat_lower or "cap" in cat_lower:
        syns += ["cap", "hat", "mu", "accessory"]

    # Deduplicate while preserving order
    seen = set()
    out = []
    for s in syns:
        if s not in seen and s:
            seen.add(s)
            out.append(s)
    return " | ".join(out)
