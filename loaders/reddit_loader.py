import praw
import pandas as pd
from datetime import datetime, timezone

def fetch_reddit(query: str, start_iso: str, end_iso: str,
                 client_id: str, client_secret: str, user_agent: str,
                 limit: int = 1000) -> pd.DataFrame:
    start_dt = datetime.fromisoformat(start_iso.replace("Z","+00:00")).astimezone(timezone.utc)
    end_dt   = datetime.fromisoformat(end_iso.replace("Z","+00:00")).astimezone(timezone.utc)

    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
    subreddit = reddit.subreddit("all")

    rows = []
    for s in subreddit.search(query, sort="new", limit=limit):
        created = datetime.fromtimestamp(s.created_utc, tz=timezone.utc)
        if start_dt <= created < end_dt:
            title = s.title or ""
            body  = (getattr(s, "selftext", "") or "")
            rows.append({
                "platform": "reddit",
                "post_id": s.id,
                "author": getattr(s.author, "name", None),
                "created_at": created.isoformat().replace("+00:00", "Z"),
                "content": f"{title} {body}".strip(),
                "like_count": getattr(s, "score", 0),
                "reply_count": getattr(s, "num_comments", 0),
                "share_count": None,
                "url": f"https://reddit.com{s.permalink}"
            })
    return pd.DataFrame(rows)
