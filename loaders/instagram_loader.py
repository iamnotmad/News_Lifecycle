# instagram_hashtag_fetch_cookies.py
import os
import sys
import time
import json
from datetime import datetime
from typing import Optional, List, Dict

import instaloader
import pandas as pd


def _load_cookies_json_into_instaloader(L: instaloader.Instaloader, cookies_json_path: str) -> None:
    """
    Load cookies exported from the browser (JSON list of cookies) into Instaloader's requests session.
    Works with Cookie-Editor / EditThisCookie exports that include name, value, domain, path, (expiry|expirationDate).
    """
    if not os.path.exists(cookies_json_path):
        raise FileNotFoundError(f"Cookies JSON not found: {cookies_json_path}")

    with open(cookies_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Some extensions export as {"cookies":[...]} — normalize to a list
    cookies = data.get("cookies", data)

    set_count = 0
    for c in cookies:
        name = c.get("name")
        value = c.get("value")
        domain = c.get("domain", ".instagram.com")
        path = c.get("path", "/")

        if not name or value is None:
            continue

        # Only set instagram cookies
        if "instagram.com" not in domain:
            continue

        L.context._session.cookies.set(name, value, domain=domain, path=path)
        set_count += 1

    print(f"[auth] Loaded {set_count} cookies from {cookies_json_path}")

    # Quick sanity check — if cookie set includes 'sessionid', we should be logged in
    who = L.test_login()  # returns username or None
    if who:
        print(f"[auth] Cookie login OK as @{who}")
    else:
        print("[warn] Cookie login did not authenticate (L.test_login() is None).")
        print("       Make sure the JSON contains a valid 'sessionid' cookie from a logged-in Instagram session.")


def _ensure_session(
    L: instaloader.Instaloader,
    login_user: Optional[str],
    sessionfile: Optional[str],
    cookies_json: Optional[str],
    login_pass: Optional[str] = None,
) -> None:
    """
    Ensure we have an authenticated Instaloader session.
    Preference order:
      1) sessionfile
      2) cookies_json (browser-exported)
      3) login_user + login_pass  (then save sessionfile if provided)
    """
    # 1) Session file
    if login_user and sessionfile and os.path.exists(sessionfile):
        print(f"[auth] Loading session for '{login_user}' from: {sessionfile}")
        L.load_session_from_file(login_user, sessionfile)
        who = L.test_login()
        print(f"[auth] Logged in as @{who}" if who else "[warn] Session file loaded but not authenticated.")
        return

    # 2) Cookies JSON
    if cookies_json:
        _load_cookies_json_into_instaloader(L, cookies_json)
        # Optionally persist cookies into an Instaloader session file for future runs
        if login_user and sessionfile:
            try:
                L.save_session_to_file(sessionfile)
                print(f"[auth] Saved cookies to session file: {sessionfile}")
            except Exception as e:
                print(f"[warn] Could not save session file: {e}")
        return

    # 3) Username/Password
    if login_user and login_pass:
        print(f"[auth] Logging in as '{login_user}' …")
        L.login(login_user, login_pass)
        if sessionfile:
            L.save_session_to_file(sessionfile)
            print(f"[auth] Session saved to: {sessionfile}")
        who = L.test_login()
        print(f"[auth] Logged in as @{who}" if who else "[warn] Login success but test_login() returned None.")
        return

    raise RuntimeError(
        "No valid Instagram auth provided.\n"
        "Provide one of:\n"
        "  • login_user + sessionfile (existing)\n"
        "  • cookies_json (browser-exported cookies)\n"
        "  • login_user + login_pass (will save session if sessionfile given)\n"
    )


def fetch_instagram_hashtag(
    hashtag: str,
    start: datetime,
    end: datetime,
    max_posts: int = 300,
    login_user: Optional[str] = None,
    sessionfile: Optional[str] = None,
    cookies_json: Optional[str] = None,
    login_pass: Optional[str] = None,
    write_csv: bool = False,
    csv_path: str = "instagram_hashtag_posts.csv",
) -> pd.DataFrame:
    """
    Fetch posts for a given hashtag between [start, end] (inclusive).
    Auth via sessionfile, cookies_json, or login credentials.

    Returns a DataFrame with:
      ['timestamp','shortcode','permalink','caption','likes','comments','owner_username','is_video']
    """
    L = instaloader.Instaloader(
        download_pictures=False,
        download_videos=False,
        download_video_thumbnails=False,
        save_metadata=False,
        compress_json=False,
        request_timeout=30,
        max_connection_attempts=3,
        quiet=False,
    )

    _ensure_session(
        L,
        login_user=login_user,
        sessionfile=sessionfile,
        cookies_json=cookies_json,
        login_pass=login_pass,
    )

    print(f"[info] Fetching posts for #{hashtag} between {start} and {end} …")
    tag = instaloader.Hashtag.from_name(L.context, hashtag.strip("#"))

    rows: List[Dict] = []
    count_seen = 0
    start_naive = start.replace(tzinfo=None)
    end_naive = end.replace(tzinfo=None)

    for post in tag.get_posts():  # newest → oldest
        time.sleep(0.2)  # be polite; reduce throttling risk
        count_seen += 1

        ts = post.date_utc.replace(tzinfo=None)

        # stop when older than 'start' (feed is reverse-chronological)
        if ts < start_naive:
            break

        if start_naive <= ts <= end_naive:
            try:
                rows.append(
                    {
                        "timestamp": ts,
                        "shortcode": post.shortcode,
                        "permalink": f"https://www.instagram.com/p/{post.shortcode}/",
                        "caption": post.caption or "",
                        "likes": getattr(post, "likes", None),
                        "comments": getattr(post, "comments", None),
                        "owner_username": getattr(post, "owner_username", None),
                        "is_video": getattr(post, "is_video", None),
                    }
                )
            except Exception as e:
                print(f"[warn] Skipped a post due to: {e}")

            if len(rows) >= max_posts:
                print(f"[info] Reached max_posts={max_posts}. Stopping.")
                break

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    print(f"[done] Collected {len(df)} posts (scanned ~{count_seen}).")

    if write_csv:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"[file] Wrote: {csv_path}")

    return df


# ---------- Example usage ----------
if __name__ == "__main__":
    IG_USER = os.getenv("IG_USER")                 # optional, used for sessionfile save
    IG_PASS = os.getenv("IG_PASS")                 # optional
    SESSION_FILE = os.getenv("IG_SESSION")         # e.g. "your_username-session"
    COOKIES_JSON = os.getenv("IG_COOKIES_JSON")    # e.g. "cookies_instagram.json"

    START = datetime(2025, 6, 12)
    END = datetime(2025, 7, 21)

    try:
        df = fetch_instagram_hashtag(
            hashtag="ahmedabadcrash",
            start=START,
            end=END,
            max_posts=200,
            login_user=IG_USER or "woke_detox_positive_thoughts",      # only needed if you also want a saved session file
            sessionfile=SESSION_FILE or "woke_detox_positive_thoughts-session",
            cookies_json=COOKIES_JSON or "D:\Documents\Manu\Course Materials\Thesis\instagram_cookies.json.json",  # <-- point to your exported JSON
            login_pass=IG_PASS,                          # not needed if cookies_json works
            write_csv=True,
            csv_path="data/instagram_ahmedabadcrash.csv",
        )
        print(df.head(10).to_string(index=False))
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
