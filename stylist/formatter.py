"""
Output formatting helpers for candidate display.
"""


def format_candidates(title, items):
    if not items:
        return f"{title}:\n  (Không có món phù hợp)\n"
    lines = [f"  {c['text']} (score={c['score']:.4f})" for c in items]
    return f"{title}:\n" + "\n".join(lines) + "\n"


__all__ = ["format_candidates"]
