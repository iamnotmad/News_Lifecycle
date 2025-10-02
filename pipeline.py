# pipeline.py
"""
End-to-end data pipeline helpers:
- combine_and_clean(dfs)
- add_sentiment(df)
- add_emotions(df)
- aggregate_daily(df)
- aggregate_emotions_daily(df)

Input schema for each platform DataFrame (case-sensitive):
    platform, post_id, author, created_at, content,
    like_count, reply_count, share_count, url

'created_at' can be str or datetime; will be normalized to UTC ISO8601 (Z).
"""

from __future__ import annotations
import pandas as pd
from typing import List

# ---- Sentiment (VADER) ----
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---- Emotions (NRC) ----
from emotions import add_emotions_to_df, EMO_KEYS


# ---------- helpers ----------

REQUIRED_COLS = [
    "platform", "post_id", "author", "created_at", "content",
    "like_count", "reply_count", "share_count", "url"
]
_NUMERIC_DEFAULTS = {"like_count": 0, "reply_count": 0, "share_count": 0}


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all required columns exist with sensible defaults."""
    out = df.copy()

    for c in REQUIRED_COLS:
        if c not in out.columns:
            out[c] = None

    # numeric fills
    for c, v in _NUMERIC_DEFAULTS.items():
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(v).astype(int)

    # created_at to UTC ISO Z string
    dt = pd.to_datetime(out["created_at"], errors="coerce", utc=True)
    out["created_at"] = dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # content as str (avoid NaNs)
    out["content"] = out["content"].fillna("").astype(str)

    # platform, author safe strings
    for c in ["platform", "author"]:
        out[c] = out[c].fillna("").astype(str)

    # post_id as string
    out["post_id"] = out["post_id"].astype(str)

    # url as str
    out["url"] = out["url"].fillna("").astype(str)

    return out[REQUIRED_COLS]


# ---------- core pipeline steps ----------

def combine_and_clean(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Combine per-platform frames into a single unified frame and de-duplicate.
    Keys for de-duplication: (platform, post_id). Fallback de-dup by URL
    only for platforms where URL is unique per item (not YouTube comments unless &lc is present).
    """
    if not dfs:
        return pd.DataFrame(columns=REQUIRED_COLS)

    normed = [_ensure_columns(df) for df in dfs if df is not None and len(df)]
    if not normed:
        return pd.DataFrame(columns=REQUIRED_COLS)

    combined = pd.concat(normed, ignore_index=True)

    # Drop rows with invalid timestamps
    dt = pd.to_datetime(combined["created_at"], errors="coerce", utc=True)
    combined = combined.loc[~dt.isna()].copy()
    combined["created_at"] = dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Normalize content a bit
    combined["content"] = (
        combined["content"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Primary de-dup by platform+post_id
    combined = (
        combined.sort_values("created_at")
                .drop_duplicates(subset=["platform", "post_id"], keep="first")
    )

    # Conservative URL fallback (avoid collapsing distinct YouTube comments)
    # Keep only if URL looks unique (e.g., Reddit permalinks, YouTube with &lc=)
    mask_unique_url = ~(
        (combined["platform"].str.lower() == "youtube") &
        (~combined["url"].str.contains(r"[?&]lc=", regex=True))
    )
    dedupe_url = combined.loc[mask_unique_url]
    keep_url = dedupe_url.drop_duplicates(subset=["url"], keep="first").index
    combined = combined.loc[
        combined.index.difference(dedupe_url.index).union(keep_url)
    ]

    # Final schema/order
    combined = _ensure_columns(combined)
    return combined.reset_index(drop=True)



def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add VADER sentiment columns: sentiment_pos, sentiment_neu, sentiment_neg, sentiment_compound.
    """
    if df.empty:
        return df.copy()

    analyzer = SentimentIntensityAnalyzer()

    def _score(text: str):
        s = analyzer.polarity_scores(text or "")
        return pd.Series(
            [s["pos"], s["neu"], s["neg"], s["compound"]],
            index=["sentiment_pos", "sentiment_neu", "sentiment_neg", "sentiment_compound"],
        )

    out = df.copy()
    scores = out["content"].apply(_score)
    out = pd.concat([out, scores], axis=1)
    return out


def add_emotions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add NRC emotion columns (anger, sadness, joy, fear, surprise, disgust) +
    dominant_emotion column using emotions.add_emotions_to_df.
    """
    if df.empty:
        return df.copy()
    return add_emotions_to_df(df, text_col="content")


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily aggregates:
      - posts (count)
      - likes / replies / shares (sum)
      - sentiment averages (pos/neu/neg/compound)
    """
    if df.empty:
        cols = ["created_at", "posts", "likes", "replies", "shares",
                "sentiment_pos", "sentiment_neu", "sentiment_neg", "sentiment_compound"]
        return pd.DataFrame(columns=cols)

    idx = pd.to_datetime(df["created_at"], utc=True)

    daily = (
        df.assign(_dt=idx)
          .set_index("_dt")
          .groupby(pd.Grouper(freq="D"))
          .agg(
              posts=("post_id", "count"),
              likes=("like_count", "sum"),
              replies=("reply_count", "sum"),
              shares=("share_count", "sum"),
              sentiment_pos=("sentiment_pos", "mean"),
              sentiment_neu=("sentiment_neu", "mean"),
              sentiment_neg=("sentiment_neg", "mean"),
              sentiment_compound=("sentiment_compound", "mean"),
          )
          .reset_index()
          .rename(columns={"_dt": "created_at"})
    )

    # pretty date for plotting (keep ISO)
    daily["created_at"] = daily["created_at"].dt.strftime("%Y-%m-%d")
    return daily


def aggregate_emotions_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily average emotion intensities (mean across posts per day).
    Columns = ['date'] + EMO_KEYS
    """
    if df.empty:
        return pd.DataFrame(columns=["date"] + EMO_KEYS)

    idx = pd.to_datetime(df["created_at"], utc=True)
    daily_emo = (
        df.assign(_dt=idx)
          .set_index("_dt")
          .groupby(pd.Grouper(freq="D"))[EMO_KEYS]
          .mean()
          .reset_index()
          .rename(columns={"_dt": "date"})
    )
    daily_emo["date"] = daily_emo["date"].dt.strftime("%Y-%m-%d")
    return daily_emo
