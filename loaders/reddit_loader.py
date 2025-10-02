import praw
import pandas as pd
from datetime import datetime, timezone

def fetch_reddit(query: str, start_iso: str, end_iso: str,
                             client_id: str, client_secret: str, user_agent: str,
                             limit: int = 5000, exact_phrase: bool = False) -> pd.DataFrame:
    start_dt = datetime.fromisoformat(start_iso.replace("Z","+00:00")).astimezone(timezone.utc)
    end_dt   = datetime.fromisoformat(end_iso.replace("Z","+00:00")).astimezone(timezone.utc)

    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
    sub = reddit.subreddit("all")

    # Use Lucene phrase query if needed
    q = f"\"{query}\"" if exact_phrase else query

    rows, fetched = [], 0
    for s in sub.search(q, sort="new", time_filter="all", syntax="lucene", limit=limit):
        created = datetime.fromtimestamp(s.created_utc, tz=timezone.utc)
        if created < start_dt:
            # results are sorted by new → once we pass the window, we can stop
            break
        if created > end_dt:
            # too new; skip but keep iterating
            continue

        title = s.title or ""
        body  = (getattr(s, "selftext", "") or "")

        # Optional: enforce local text match to be extra strict
        # if exact_phrase and f" {query.lower()} " not in (f" {title} {body} ".lower()):
        #     continue

        rows.append({
            "platform": "reddit",
            "post_id": s.id,
            "author": getattr(s.author, "name", None),
            "created_at": created.isoformat().replace("+00:00", "Z"),
            "content": f"{title} {body}".strip(),
            "like_count": getattr(s, "score", 0),
            "reply_count": getattr(s, "num_comments", 0),
            "share_count": None,
            "url": f"https://reddit.com{s.permalink}",
            "subreddit": str(s.subreddit),
            "query": query
        })
        fetched += 1

    df = pd.DataFrame(rows).drop_duplicates(subset=["post_id"]).reset_index(drop=True)
    return df
