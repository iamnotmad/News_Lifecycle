import os
import sys
import pandas as pd

from loaders.reddit_loader import fetch_reddit
from loaders.youtube_loader import fetch_youtube_comments          # or swap to videos loader
# from loaders.youtube_videos_loader import fetch_youtube_videos
from loaders.instagram_loader import fetch_instagram_hashtag

from pipeline import (
    combine_and_clean,
    add_sentiment,
    add_emotions,
    aggregate_daily,
    aggregate_emotions_daily,
)

QUERY = "ahmedabad crash"
START = "2025-06-12T00:00:00Z"
END   = "2025-07-21T00:00:00Z"

REDDIT_ID     = os.getenv("REDDIT_ID")
REDDIT_SECRET = os.getenv("REDDIT_SECRET")
REDDIT_UA     = os.getenv("REDDIT_UA", "NewsScraperApp/1.0")
YOUTUBE_KEY   = os.getenv("YOUTUBE_API_KEY")
IG_USER       = os.getenv("IG_USER")
IG_PASS       = os.getenv("IG_PASS")

os.makedirs("data", exist_ok=True)
dfs = []

# ---- Reddit
try:
    if REDDIT_ID and REDDIT_SECRET:
        print("Fetching Reddit…")
        reddit_df = fetch_reddit(QUERY, START, END, REDDIT_ID, REDDIT_SECRET, REDDIT_UA, limit=1200)
        print(f"  Reddit rows: {len(reddit_df)}")
        reddit_df.to_csv("data/reddit_raw.csv", index=False)
        dfs.append(reddit_df)
    else:
        print("Skipping Reddit: set REDDIT_ID/REDDIT_SECRET")
except Exception as e:
    print("Skipping Reddit due to error:", e)

# ---- YouTube (comments or videos)
try:
    if YOUTUBE_KEY:
        print("Fetching YouTube…")
        yt_df = fetch_youtube_comments(YOUTUBE_KEY, QUERY, START, END, max_videos=10, max_comments_per_video=300)
        #yt_df = fetch_youtube_videos(YOUTUBE_KEY, QUERY, START, END, max_videos=25)  # <- if comments 403 often
        print(f"  YouTube rows: {len(yt_df)}")
        yt_df.to_csv("data/youtube_raw.csv", index=False)
        dfs.append(yt_df)
    else:
        print("Skipping YouTube: set YOUTUBE_API_KEY")
except Exception as e:
    print("Skipping YouTube due to error:", e)

# ---- Instagram (optional; requires session/creds)
try:
    if IG_USER and IG_PASS:
        print("Fetching Instagram…")
        ig_df = fetch_instagram_hashtag(QUERY.replace(" ", ""), START, END, max_posts=300,
                                        login_user=IG_USER, login_pass=IG_PASS)
        print(f"  Instagram rows: {len(ig_df)}")
        ig_df.to_csv("data/instagram_raw.csv", index=False)
        dfs.append(ig_df)
    else:
        print("Skipping Instagram: no IG_USER/IG_PASS provided.")
except Exception as e:
    print("Skipping Instagram due to error:", e)

# ---- Combine & analyze
if not dfs:
    print("No dataframes returned from sources. Nothing to write.")
    sys.exit(0)

combined = combine_and_clean(dfs)
print("Combined rows:", len(combined))

if combined.empty:
    print("Combined dataset is empty after cleaning. Nothing to write.")
    sys.exit(0)

combined = add_sentiment(combined)
combined = add_emotions(combined)

# ---- Save outputs
# ---- Save outputs ----
# Full dataset with emotions
combined.to_csv("data/combined_with_emotions.csv", index=False, encoding="utf-8-sig")

# Slim dataset for inspection / sharing
combined[[
    "platform","post_id","author","created_at","content",
    "like_count","reply_count","share_count","url",
    "sentiment_pos","sentiment_neu","sentiment_neg","sentiment_compound"
]].to_csv("data/combined.csv", index=False, encoding="utf-8-sig")

# Daily aggregates
daily = aggregate_daily(combined)
daily.to_csv("data/daily.csv", index=False, encoding="utf-8-sig")

# Daily emotion aggregates
daily_emo = aggregate_emotions_daily(combined)
daily_emo.to_csv("data/daily_emotions.csv", index=False, encoding="utf-8-sig")

# ---- Report summary ----
print("Wrote:")
for f in ["combined.csv","combined_with_emotions.csv","daily.csv","daily_emotions.csv"]:
    p = os.path.join("data", f)
    try:
        nrows = len(pd.read_csv(p, encoding="utf-8-sig"))
        print(f"  - {p} ({nrows} rows)")
    except Exception as e:
        print(f"  - {p} (size unknown; read error {e})")

