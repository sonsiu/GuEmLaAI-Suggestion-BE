"""
Embedding utilities and model initialization.
"""
import time
import typing as t
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder  # CrossEncoder kept for compatibility

from .config import (
    DEVICE,
    EMBED_MODEL_NAME,
    EMBED_MODEL_PATH,
    TOP_K_EMBED,
    QUERY_SCORE_AGGREGATION,
)

print("Loading embedding model:", EMBED_MODEL_NAME, "device:", DEVICE)
start_time = time.time()
embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)
# embed_model = SentenceTransformer(EMBED_MODEL_PATH, device=DEVICE)
end_time = time.time()
elapsed = end_time - start_time
print(f"? Model loaded successfully in {elapsed:.2f} seconds.")


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


__all__ = [
    "SentenceTransformer",
    "CrossEncoder",
    "embed_model",
    "embedding_func",
    "aggregate_multi_query_candidates",
]
