# emotions.py
"""
Lightweight emotion scoring for text using the NRC Emotion Lexicon (NRCLex).
Outputs six emotions: anger, sadness, joy, fear, surprise, disgust.
All scores are normalized to sum to 1 when any emotion is found, else 0s.

Requires:  pip install nrclex
"""

from __future__ import annotations
from typing import Dict, List

try:
    # noinspection PyUnresolvedReferences
    from nrclex import NRCLex
except Exception as e:
    raise ImportError(
        "nrclex is not available in this Python environment. "
        "Activate your .venv311 and run:  pip install nrclex"
    ) from e


# We focus on these six (subset of NRC: anger, anticipation, disgust, fear, joy, sadness, surprise, trust)
EMO_KEYS: List[str] = ["anger", "sadness", "joy", "fear", "surprise", "disgust"]


def compute_emotions(text: str) -> Dict[str, float]:
    """
    Return a dict with keys EMO_KEYS and float scores in [0,1].
    If no emotions are found, returns zeros for all keys.
    """
    if not isinstance(text, str) or not text.strip():
        return {k: 0.0 for k in EMO_KEYS}

    # Build NRCLex doc
    doc = NRCLex(text)
    raw = doc.raw_emotion_scores or {}

    # Keep only our six emotions
    filtered = {k: float(raw.get(k, 0.0)) for k in EMO_KEYS}

    total = sum(filtered.values())
    if total <= 0:
        # Nothing matched the lexicon
        return filtered

    # Normalize so the scores sum to 1.0
    return {k: v / total for k, v in filtered.items()}


def dominant_emotion(emodict: Dict[str, float]) -> str:
    """
    Given a dict from compute_emotions(), return the label of the max emotion;
    returns 'none' if all zeros.
    """
    if not emodict or all(v == 0 for v in emodict.values()):
        return "none"
    return max(emodict, key=emodict.get)


def add_emotions_to_df(df, text_col: str = "content"):
    """
    Convenience helper: adds emotion columns + 'dominant_emotion' to a DataFrame.
    Returns a new DataFrame (does not modify in place).

    Example:
        df = add_emotions_to_df(df, text_col="content")
    """
    import pandas as pd  # local import to avoid hard dependency if used standalone

    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame.")

    # Compute per-row emotion dicts
    rows = [compute_emotions(str(t)) for t in df[text_col].fillna("")]
    emo_df = pd.DataFrame(rows, columns=EMO_KEYS)

    # Dominant label
    dom = [dominant_emotion(r) for r in rows]
    emo_df["dominant_emotion"] = dom

    # Concatenate with original
    out = pd.concat([df.reset_index(drop=True), emo_df.reset_index(drop=True)], axis=1)
    return out
