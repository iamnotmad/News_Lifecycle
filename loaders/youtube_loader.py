from googleapiclient.discovery import build
import pandas as pd
from datetime import timezone

def fetch_youtube_comments(api_key, query, start_iso, end_iso, max_videos=10, max_comments=200):
    youtube = build("youtube", "v3", developerKey=api_key)

    # Search for videos
    search_req = youtube.search().list(
        q=query, part="id", type="video", maxResults=max_videos, order="relevance"
    )
    search_resp = search_req.execute()

    rows = []
    for item in search_resp.get("items", []):
        vid = item["id"]["videoId"]

        try:
            # Try fetching comments
            comments_req = youtube.commentThreads().list(
                part="snippet", videoId=vid, maxResults=max_comments, textFormat="plainText"
            )
            comments_resp = comments_req.execute()

            for c in comments_resp.get("items", []):
                snippet = c["snippet"]["topLevelComment"]["snippet"]
                rows.append({
                    "platform": "youtube",
                    "post_id": c["id"],
                    "author": snippet.get("authorDisplayName"),
                    "created_at": snippet.get("publishedAt"),
                    "content": snippet.get("textDisplay"),
                    "like_count": snippet.get("likeCount"),
                    "reply_count": snippet.get("totalReplyCount"),
                    "share_count": None,
                    "url": f"https://www.youtube.com/watch?v={vid}"
                })
        except Exception as e:
            print(f"Skipping video {vid} due to error: {e}")
            continue

    return pd.DataFrame(rows)
