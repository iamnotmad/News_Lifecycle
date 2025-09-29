# pages/2_Emotion_Analytics.py
import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

EMO_KEYS = ["anger","sadness","joy","fear","surprise","disgust"]

st.set_page_config(layout="wide")
st.title("Emotion Analytics")

def _read(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

# ---- Load
df_em = _read("data/combined_with_emotions.csv")
if df_em.empty:
    st.warning("Missing `data/combined_with_emotions.csv`. Run `python run_fetch.py` first.")
    st.stop()

df_em["created_at"] = pd.to_datetime(df_em["created_at"], utc=True, errors="coerce")

# ---- Filters
colA, colB = st.columns([1,1])
with colA:
    platforms = st.multiselect(
        "Filter by platform",
        sorted(df_em["platform"].dropna().unique()),
        default=None
    )
with colB:
    min_d = df_em["created_at"].min().date()
    max_d = df_em["created_at"].max().date()
    date_range = st.date_input("Date range", value=(min_d, max_d))

view = df_em.copy()
if platforms:
    view = view[view["platform"].isin(platforms)]
if date_range and len(date_range) == 2:
    start = pd.Timestamp(date_range[0]).tz_localize("UTC")
    end   = pd.Timestamp(date_range[1]).tz_localize("UTC") + pd.Timedelta(days=1)
    cts   = view["created_at"]
    view  = view[(cts >= start) & (cts < end)]

if view.empty:
    st.info("No rows after filters.")
    st.stop()

# ----------------- Charts + Download buttons -----------------

# 1) Stacked area (daily mean emotions)
st.subheader("Daily Emotion Intensity")
tmp = (
    view.set_index("created_at")
        .groupby(pd.Grouper(freq="D"))[EMO_KEYS]
        .mean()
        .reset_index()
        .rename(columns={"created_at":"date"})
)
fig_area = px.area(tmp, x="date", y=EMO_KEYS)
st.plotly_chart(fig_area, use_container_width=True)
st.download_button(
    "⬇️ Download 'Daily Emotion Intensity' PNG",
    data=fig_area.to_image(format="png"),
    file_name="daily_emotion_intensity.png",
    mime="image/png"
)

# 2) Bars by platform
st.subheader("Average Emotion by Platform")
plat = (view.groupby("platform")[EMO_KEYS].mean().reset_index())
melt = plat.melt(id_vars="platform", value_vars=EMO_KEYS,
                 var_name="emotion", value_name="intensity")
fig_bar = px.bar(melt, x="platform", y="intensity", color="emotion", barmode="group")
st.plotly_chart(fig_bar, use_container_width=True)
st.download_button(
    "⬇️ Download 'Average Emotion by Platform' PNG",
    data=fig_bar.to_image(format="png"),
    file_name="emotion_by_platform.png",
    mime="image/png"
)

# 3) Radar
st.subheader("Emotion Radar")
opt = st.selectbox("Choose platform", ["All"] + sorted(view["platform"].dropna().unique().tolist()))
rad = view if opt == "All" else view[view["platform"] == opt]
means = rad[EMO_KEYS].mean().tolist() if not rad.empty else [0]*len(EMO_KEYS)
cats  = EMO_KEYS + [EMO_KEYS[0]]
vals  = means + [means[0]]
fig_radar = go.Figure(go.Scatterpolar(r=vals, theta=cats, fill='toself', name=opt))
rng = max(0.15, (max(means) if any(means) else 0) + 0.05)
fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, rng])), showlegend=False)
st.plotly_chart(fig_radar, use_container_width=True)
st.download_button(
    "⬇️ Download 'Emotion Radar' PNG",
    data=fig_radar.to_image(format="png"),
    file_name="emotion_radar.png",
    mime="image/png"
)

# 4) Dominant emotion distribution
if "dominant_emotion" in view.columns:
    st.subheader("Dominant Emotion Distribution")
    fig_pie = px.pie(view, names="dominant_emotion")
    st.plotly_chart(fig_pie, use_container_width=True)
    st.download_button(
        "⬇️ Download 'Dominant Emotion Distribution' PNG",
        data=fig_pie.to_image(format="png"),
        file_name="dominant_emotion_pie.png",
        mime="image/png"
    )

# ----------------- Data downloads -----------------
st.subheader("Download data")

# Filtered with emotions
csv_buf = io.StringIO()
view_out = view.copy()
view_out["created_at"] = view_out["created_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
view_out.to_csv(csv_buf, index=False)
st.download_button(
    label="⬇️ Download FILTERED rows (with emotions)",
    data=csv_buf.getvalue().encode("utf-8"),
    file_name="filtered_with_emotions.csv",
    mime="text/csv",
)

# Daily emotions (pipeline)
if Path("data/daily_emotions.csv").exists():
    st.download_button(
        label="⬇️ Download DAILY emotions (pipeline)",
        data=Path("data/daily_emotions.csv").read_bytes(),
        file_name="daily_emotions.csv",
        mime="text/csv",
    )

# Full combined-with-emotions (pipeline)
if Path("data/combined_with_emotions.csv").exists():
    st.download_button(
        label="⬇️ Download COMBINED with emotions (pipeline)",
        data=Path("data/combined_with_emotions.csv").read_bytes(),
        file_name="combined_with_emotions.csv",
        mime="text/csv",
    )
