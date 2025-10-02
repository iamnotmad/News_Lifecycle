from googleapiclient.discovery import build
from datetime import datetime, timezone
import pandas as pd

def _parse_iso(s):
    # robust RFC3339 → aware datetime
    return datetime.fromisoformat(s.replace('Z', '+00:00')).astimezone(timezone.utc)

def fetch_youtube_comments(api_key, query, start_iso, end_iso,
                           max_videos=25, max_comments_per_video=500):
    yt = build("youtube", "v3", developerKey=api_key)

    start_dt = _parse_iso(start_iso)
    end_dt   = _parse_iso(end_iso)

    rows = []

    # --- 1) Search videos (paginate) ---
    searched = 0
    next_token = None
    while searched < max_videos:
        page_size = min(50, max_videos - searched)
        resp = yt.search().list(
            q=query, part="id", type="video", maxResults=page_size,
            order="relevance", pageToken=next_token
        ).execute()
        items = resp.get("items", [])
        if not items: break

        for it in items:
            vid = it["id"]["videoId"]

            # --- 2) Fetch top-level comments for each video (paginate) ---
            got = 0
            c_token = None
            while got < max_comments_per_video:
                c_page_size = min(100, max_comments_per_video - got)
                try:
                    c_resp = yt.commentThreads().list(
                        part="snippet",
                        videoId=vid,
                        maxResults=c_page_size,
                        textFormat="plainText",
                        pageToken=c_token,
                        order="time"  # newest first helps with date filtering
                    ).execute()
                except Exception as e:
                    print(f"Skipping video {vid} due to error: {e}")
                    break

                for thread in c_resp.get("items", []):
                    s = thread["snippet"]["topLevelComment"]["snippet"]
                    published = _parse_iso(s["publishedAt"])
                    if published < start_dt or published > end_dt:
                        continue  # filter to your window

                    comment_id = thread["id"]
                    rows.append({
                        "platform": "youtube",
                        "post_id": comment_id,
                        "author": s.get("authorDisplayName"),
                        "author_channel_id": s.get("authorChannelId", {}).get("value"),
                        "created_at": s.get("publishedAt"),
                        "content": s.get("textOriginal"),  # plain text
                        "like_count": s.get("likeCount"),
                        "reply_count": thread["snippet"].get("totalReplyCount"),
                        "share_count": None,  # not available
                        "url": f"https://www.youtube.com/watch?v={vid}&lc={comment_id}",
                        "video_id": vid,
                        "query": query
                    })
                got += len(c_resp.get("items", []))
                c_token = c_resp.get("nextPageToken")
                if not c_token:
                    break
        searched += len(items)
        next_token = resp.get("nextPageToken")
        if not next_token:
            break

    # Deduplicate just in case
    df = pd.DataFrame(rows).drop_duplicates(subset=["post_id"])
    return df.reset_index(drop=True)
