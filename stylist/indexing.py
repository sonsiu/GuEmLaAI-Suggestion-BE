"""
FAISS indexing helpers and cache management.
"""
import os
import re
import typing as t
import numpy as np
import faiss

from .config import WORKING_DIR
from .embeddings import embedding_func
from .wardrobe import normalize_wardrobe_items
from .config import ensure_working_dir


def _ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _sanitize_prefix(prefix: str) -> str:
    """Make a filesystem-safe prefix (Windows-safe: no colons, slashes, etc.)."""
    sanitized = re.sub(r"[^0-9A-Za-z._-]", "-", prefix)
    sanitized = re.sub(r"-{2,}", "-", sanitized).strip("-")
    return sanitized or "wardrobe"


def build_faiss_index(
    wardrobe_items,
    user_id: t.Union[str, int, None] = None,
    wardrobe_version: str | None = None,
    working_dir: str = WORKING_DIR,
):
    ensure_working_dir()

    prefix_parts = []
    if user_id is not None:
        prefix_parts.append(str(user_id))
    if wardrobe_version:
        prefix_parts.append(str(wardrobe_version))
    prefix = "_".join(prefix_parts) if prefix_parts else "wardrobe"

    safe_prefix = _sanitize_prefix(prefix)
    if safe_prefix != prefix:
        print(f"Sanitized prefix '{prefix}' -> '{safe_prefix}' for filesystem safety")

    index_file = os.path.join(working_dir, f"{safe_prefix}_faiss.index")
    embeddings_file = os.path.join(working_dir, f"{safe_prefix}_embeddings.npy")

    _ensure_dir(working_dir)

    # Decide whether to reuse cache or rebuild
    rebuild = True
    existing_embeddings = None
    if os.path.exists(index_file) and os.path.exists(embeddings_file):
        try:
            existing_embeddings = np.load(embeddings_file)
            if existing_embeddings.shape[0] == len(wardrobe_items):
                rebuild = False
        except Exception:
            rebuild = True

    index = None
    if not rebuild:
        try:
            print(f"\nLoading existing FAISS index from {index_file}")
            index = faiss.read_index(index_file)
            # Validate cache integrity and size match before trusting it
            if (
                index.ntotal != len(wardrobe_items)
                or existing_embeddings is None
                or existing_embeddings.shape[0] != index.ntotal
            ):
                print("Cache size mismatch detected; rebuilding FAISS index.")
                rebuild = True
        except Exception:
            print("Cached FAISS index corrupted; rebuilding.")
            rebuild = True

    if rebuild:
        print(f"\nBuilding FAISS index for prefix '{prefix}'")

        wardrobe_texts = normalize_wardrobe_items(wardrobe_items)
        wardrobe_embeddings = embedding_func(wardrobe_texts)
        dim = wardrobe_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(wardrobe_embeddings)

        faiss.write_index(index, index_file)
        np.save(embeddings_file, wardrobe_embeddings)
        print(f"   Indexed {index.ntotal} wardrobe items (dim={dim})")
        print(f"   Saved to {index_file} and {embeddings_file}")
    else:
        wardrobe_embeddings = existing_embeddings
        print(f"   Loaded {index.ntotal} wardrobe items (dim={wardrobe_embeddings.shape[1]})")

    return index, wardrobe_embeddings


__all__ = ["build_faiss_index"]
