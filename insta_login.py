# insta_login.py
import os
import getpass
import instaloader

username = input("Instagram username: ").strip()
pwd = getpass.getpass("Instagram password (typing is hidden): ")

L = instaloader.Instaloader(download_pictures=False, download_videos=False,
                            save_metadata=False, compress_json=False)
try:
    L.login(username, pwd)   # may ask for 2FA in exception logs
    session_file = f"session-{username}"
    L.save_session_to_file(session_file)
    print("Saved session to:", session_file)
except Exception as e:
    print("Login failed:", e)
    # If 2FA is required, run CLI: instaloader -l <username> so it can prompt interactively.
