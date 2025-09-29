# loaders/x_loader.py
import pandas as pd
from datetime import timezone
from snscrape.modules.twitter import TwitterSearchScraper  # <-- use this import

def fetch_x(query: str, start_iso: str, end_iso: str, limit: int = 1000) -> pd.DataFrame:
    start = start_iso.split("T")[0]
    end   = end_iso.split("T")[0]
    q = f'({query}) since:{start} until:{end} -filter:retweets'

    rows = []
    for i, t in enumerate(TwitterSearchScraper(q).get_items()):
        if i >= limit: break
        created = t.date.astimezone(timezone.utc)
        rows.append({
            "platform": "x",
            "post_id": t.id,
            "author": t.user.username if t.user else None,
            "created_at": created.isoformat().replace("+00:00", "Z"),
            "content": t.rawContent,
            "like_count": getattr(t, "likeCount", 0),
            "reply_count": getattr(t, "replyCount", 0),
            "share_count": getattr(t, "retweetCount", 0),
            "url": t.url
        })
    return pd.DataFrame(rows)
