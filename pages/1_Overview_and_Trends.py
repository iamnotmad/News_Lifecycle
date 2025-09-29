# pages/1_Overview_and_Trends.py
import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(layout="wide")
st.title("Overview & Trends")

def _read(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

# ---- Load
combined = _read("data/combined.csv")
daily    = _read("data/daily.csv")

if combined.empty or daily.empty:
    st.warning("Missing `data/combined.csv` or `data/daily.csv`. Run `python run_fetch.py` first.")
    st.stop()

# ---- Parse timestamps
combined["created_at"] = pd.to_datetime(combined["created_at"], utc=True, errors="coerce")
daily["created_at"]    = pd.to_datetime(daily["created_at"], errors="coerce")  # date-only in pipeline

# ---- Filters
colA, colB = st.columns([1,1])
with colA:
    plats = st.multiselect(
        "Filter by platform",
        sorted(combined["platform"].dropna().unique()),
        default=None
    )
with colB:
    dmin = combined["created_at"].min().date()
    dmax = combined["created_at"].max().date()
    drange = st.date_input("Date range", value=(dmin, dmax))

view = combined.copy()
if plats:
    view = view[view["platform"].isin(plats)]
if drange and len(drange) == 2:
    start = pd.Timestamp(drange[0]).tz_localize("UTC")
    end   = pd.Timestamp(drange[1]).tz_localize("UTC") + pd.Timedelta(days=1)
    cts   = view["created_at"]
    view  = view[(cts >= start) & (cts < end)]

if view.empty:
    st.info("No rows after filters. Adjust filters and try again.")
    st.stop()

# ---- Rebuild daily aggregates from FILTERED data
daily_view = (
    view.assign(_dt=view["created_at"])
        .set_index("_dt")
        .groupby(pd.Grouper(freq="D"))
        .agg(
            posts=("post_id","count"),
            likes=("like_count","sum"),
            replies=("reply_count","sum"),
            shares=("share_count","sum"),
            sentiment_compound=("sentiment_compound","mean"),
        )
        .reset_index()
        .rename(columns={"_dt":"date"})
)

# ----------------- Charts + Download buttons -----------------

# 1) Posts per day
st.subheader("Posts per day")
fig_posts = px.line(daily_view, x="date", y="posts")
st.plotly_chart(fig_posts, use_container_width=True)
st.download_button(
    "⬇️ Download 'Posts per day' PNG",
    data=fig_posts.to_image(format="png"),
    file_name="posts_per_day.png",
    mime="image/png"
)

# 2) Engagement over time
c1, c2 = st.columns(2)
with c1:
    st.subheader("Engagement over time")
    fig_eng = px.line(daily_view, x="date", y=["likes","replies","shares"])
    st.plotly_chart(fig_eng, use_container_width=True)
    st.download_button(
        "⬇️ Download 'Engagement over time' PNG",
        data=fig_eng.to_image(format="png"),
        file_name="engagement_over_time.png",
        mime="image/png"
    )

# 3) Sentiment (compound) over time
with c2:
    st.subheader("Average sentiment (compound) over time")
    fig_sent = px.line(daily_view, x="date", y="sentiment_compound")
    st.plotly_chart(fig_sent, use_container_width=True)
    st.download_button(
        "⬇️ Download 'Sentiment over time' PNG",
        data=fig_sent.to_image(format="png"),
        file_name="sentiment_over_time.png",
        mime="image/png"
    )

# 4) Per-platform time series
st.subheader("Per-platform posts")
perplat = (
    view.assign(date=view["created_at"].dt.floor("D"))
        .groupby(["date","platform"])["post_id"]
        .count()
        .reset_index(name="posts")
)
fig_pp = px.line(perplat, x="date", y="posts", color="platform")
st.plotly_chart(fig_pp, use_container_width=True)
st.download_button(
    "⬇️ Download 'Per-platform posts' PNG",
    data=fig_pp.to_image(format="png"),
    file_name="per_platform_posts.png",
    mime="image/png"
)

# ----------------- Data downloads -----------------
st.subheader("Download data")

# Filtered combined
csv_buf = io.StringIO()
view_out = view.copy()
view_out["created_at"] = view_out["created_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
view_out.to_csv(csv_buf, index=False)
st.download_button(
    label="⬇️ Download FILTERED rows (combined)",
    data=csv_buf.getvalue().encode("utf-8"),
    file_name="filtered_combined.csv",
    mime="text/csv",
)

# Daily aggregates (pipeline)
if Path("data/daily.csv").exists():
    st.download_button(
        label="⬇️ Download DAILY aggregates (pipeline)",
        data=Path("data/daily.csv").read_bytes(),
        file_name="daily.csv",
        mime="text/csv",
    )

# Full combined (pipeline)
if Path("data/combined.csv").exists():
    st.download_button(
        label="⬇️ Download COMBINED (pipeline)",
        data=Path("data/combined.csv").read_bytes(),
        file_name="combined.csv",
        mime="text/csv",
    )
