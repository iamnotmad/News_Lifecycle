# pages/3_Misinformation_Profiler.py
#with guardrails and explanations

import io
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(layout="wide")
st.title("Misinformation Profiler (Heuristic)")

DATA_PATH = Path("data/combined_with_emotions.csv")

if not DATA_PATH.exists():
    st.warning("Missing `data/combined_with_emotions.csv`. Run `python run_fetch.py` first.")
    st.stop()

df = pd.read_csv(DATA_PATH)
if df.empty:
    st.warning("No rows in combined_with_emotions.csv")
    st.stop()

# --- Parse time once
df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")

# --- Ensure columns exist
for col in ["content","platform","url","sentiment_compound","sentiment_pos","sentiment_neu","sentiment_neg",
            "anger","sadness","joy","fear","surprise","disgust"]:
    if col not in df.columns:
        df[col] = 0 if col != "content" else ""

# ---------------- Heuristic scoring with guardrails ----------------

RUMOR_WORDS = [
    r"\brumou?r\b", r"\brumors?\b", r"\brumoured\b", r"\bunverified\b", r"\balleg(ed|edly)\b",
    r"\bclaims?\b", r"\bhoax\b", r"\bfake\b", r"\bfalse\b", r"\bmisleading\b", r"\bdebunk(ed|ing)?\b",
    r"\bconspiracy\b", r"\bviral\b", r"\bclickbait\b", r"\bscam\b", r"\bpropaganda\b"
]
SENSATIONAL = [
    r"\bbreaking\b", r"\bshocking\b", r"\bexplosive\b", r"\bstunning\b", r"\bunbelievable\b",
    r"\byou won'?t believe\b", r"\bproof\b", r"\bexposed\b", r"\bsecret\b", r"\bcover[- ]?up\b",
]
EMOJI_BOOST = r"[🔥🚨🤯😱😡🤬💥❗‼️❕❓]"

DEBUNK_RE = re.compile(r"\b(debunk(ed|ing)?|fact[- ]?check(ed|ing)?|clarif(y|ied|ication))\b", re.IGNORECASE)
RUMOR_RE  = re.compile("|".join(RUMOR_WORDS), re.IGNORECASE)
SENS_RE   = re.compile("|".join(SENSATIONAL), re.IGNORECASE)
EMOJI_RE  = re.compile(EMOJI_BOOST)

URL_RE   = re.compile(r"https?://([^/\s]+)")

# Optional domain lists — fill if you curate
LOW_CRED_DOMAINS = {
    # "example.badsite.com",
}
MAINSTREAM_DOMAINS = {
    # "thehindu.com", "indiatoday.in", "ndtv.com",
}

def caps_ratio(text: str) -> float:
    if not text: return 0.0
    letters = [c for c in text if c.isalpha()]
    if not letters: return 0.0
    upper = sum(1 for c in letters if c.isupper())
    return upper / max(1, len(letters))

def punct_boost(text: str) -> float:
    if not text: return 0.0
    exclam = text.count("!")
    quest  = text.count("?")
    dots   = text.count("...")
    return min(1.0, (math.log1p(exclam) + 0.8*math.log1p(quest) + 0.5*math.log1p(dots)) / 5.0)

def emoji_boost(text: str) -> float:
    if not text: return 0.0
    m = EMOJI_RE.findall(text)
    return min(1.0, len(m) / 3.0)

def rumor_hit(text: str) -> float:
    if not text: return 0.0
    hits = len(RUMOR_RE.findall(text)) + len(SENS_RE.findall(text)) + len(EMOJI_RE.findall(text))
    return min(1.0, hits / 4.0)

def domain_adjust(url: str) -> float:
    if not isinstance(url, str) or not url:
        return 0.0
    m = URL_RE.search(url)
    if not m:
        return 0.0
    host = m.group(1).lower()
    if host in LOW_CRED_DOMAINS or any(host.endswith(d) for d in LOW_CRED_DOMAINS):
        return +0.08  # upweight suspicion
    if host in MAINSTREAM_DOMAINS or any(host.endswith(d) for d in MAINSTREAM_DOMAINS):
        return -0.06  # downweight for mainstream
    return 0.0

def misinfo_score_row(r) -> float:
    text = str(r.get("content",""))

    comp = float(r.get("sentiment_compound", 0.0))
    neu  = float(r.get("sentiment_neu", 0.0))

    anger    = float(r.get("anger", 0.0))
    fear     = float(r.get("fear", 0.0))
    surprise = float(r.get("surprise", 0.0))
    joy      = float(r.get("joy", 0.0))
    disgust  = float(r.get("disgust", 0.0))

    r_hit = rumor_hit(text)
    caps  = caps_ratio(text)
    p_bo  = punct_boost(text)
    e_bo  = emoji_boost(text)

    dom_adj = domain_adjust(str(r.get("url","")))
    debunk  = 1.0 if DEBUNK_RE.search(text) else 0.0

    # style gate — require at least two stylistic signals to fully count
    signals = sum([
        r_hit > 0,
        caps > 0.2,
        p_bo  > 0.2,
        e_bo  > 0.0
    ])
    style_gate = 1.0 if signals >= 2 else 0.5

    # emotion cap (anger/fear/surprise/disgust up, joy down)
    emo_mix = (0.35*anger + 0.25*fear + 0.20*surprise + 0.10*disgust) - 0.10*joy
    emo_mix = max(0.0, min(1.0, emo_mix))

    extreme = max(0.0, min(1.0, abs(comp)))
    low_neu = max(0.0, 0.5 - neu) * 2.0

    score = (
        0.28*r_hit +
        style_gate * (0.12*caps + 0.10*p_bo + 0.07*e_bo) +
        0.18*emo_mix +
        0.12*extreme +
        0.05*low_neu +
        dom_adj - 0.10*debunk
    )
    return float(max(0.0, min(1.0, score)))

with st.spinner("Scoring suspected misinformation…"):
    df["misinfo_score"] = df.apply(misinfo_score_row, axis=1)

# -------------------- Sidebar controls --------------------
st.sidebar.header("Controls")
threshold = st.sidebar.slider("Suspicion threshold", 0.0, 1.0, 0.60, 0.01)
platforms = st.sidebar.multiselect("Platforms", sorted(df["platform"].dropna().unique()), default=None)

dmin = df["created_at"].min().date()
dmax = df["created_at"].max().date()
date_range = st.sidebar.date_input("Date range", value=(dmin, dmax))

view = df.copy()
if platforms:
    view = view[view["platform"].isin(platforms)]
if date_range and len(date_range) == 2:
    start = pd.Timestamp(date_range[0]).tz_localize("UTC")
    end   = pd.Timestamp(date_range[1]).tz_localize("UTC") + pd.Timedelta(days=1)
    cts   = view["created_at"]
    view  = view[(cts >= start) & (cts < end)]

view["suspected_misinfo"] = view["misinfo_score"] >= threshold

# -------------------- KPIs --------------------
c1, c2, c3, c4 = st.columns(4)
total = len(view)
sus   = int(view["suspected_misinfo"].sum())
share = (sus / total * 100) if total else 0.0
avg_sc= view["misinfo_score"].mean() if total else 0.0

c1.metric("Posts (filtered)", f"{total:,}")
c2.metric("Suspected", f"{sus:,}", f"{share:.1f}%")
c3.metric("Avg misinfo score", f"{avg_sc:.2f}")
c4.metric("Threshold", f"{threshold:.2f}")

# -------------------- Charts --------------------
st.subheader("Emergence & Progression Over Time")
ts = (view.assign(date=view["created_at"].dt.floor("D"))
           .groupby(["date","suspected_misinfo"])["post_id"].count()
           .reset_index(name="posts"))
fig1 = px.area(ts, x="date", y="posts", color="suspected_misinfo",
               title="Suspected vs Other over time", groupnorm="fraction")
st.plotly_chart(fig1, use_container_width=True)
st.download_button(
    "⬇️ Download 'Emergence over time' PNG",
    data=fig1.to_image(format="png"),
    file_name="emergence_over_time.png",
    mime="image/png"
)

c5, c6 = st.columns(2)
with c5:
    st.subheader("Platform comparison (share suspected)")
    plat = (view.groupby(["platform","suspected_misinfo"])["post_id"].count()
                .reset_index(name="count"))
    total_plat = plat.groupby("platform")["count"].transform("sum")
    plat["share"] = np.where(total_plat>0, plat["count"]/total_plat, 0)
    fig2 = px.bar(plat, x="platform", y="share", color="suspected_misinfo",
                  barmode="stack", text_auto=".0%")
    fig2.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)
    st.download_button(
        "⬇️ Download 'Platform share suspected' PNG",
        data=fig2.to_image(format="png"),
        file_name="platform_share_suspected.png",
        mime="image/png"
    )

with c6:
    st.subheader("Cumulative suspected (emergence curve)")
    cum = (view[view["suspected_misinfo"]]
               .assign(date=view["created_at"].dt.floor("D"))
               .groupby("date")["post_id"].count()
               .rename("new").reset_index())
    cum["cumulative"] = cum["new"].cumsum()
    fig3 = px.line(cum, x="date", y="cumulative", markers=True)
    st.plotly_chart(fig3, use_container_width=True)
    st.download_button(
        "⬇️ Download 'Cumulative suspected' PNG",
        data=fig3.to_image(format="png"),
        file_name="cumulative_suspected.png",
        mime="image/png"
    )

# -------------------- Top suspected posts & explanations --------------------
st.subheader("Top suspected posts (by score)")
cols = ["created_at","platform","misinfo_score","content","url","like_count","reply_count","share_count"]
topn = (view[view["suspected_misinfo"]]
            .sort_values("misinfo_score", ascending=False)
            [cols].head(25))

if topn.empty:
    st.info("No posts exceed the current threshold.")
else:
    show = topn.copy()
    show["created_at"] = pd.to_datetime(show["created_at"], utc=True, errors="coerce")
    show["content"] = show["content"].fillna("").str.slice(0, 240)
    st.dataframe(show, use_container_width=True, height=480)

    # Optional: Sampling for manual calibration
    if st.button("Sample 20 suspected for manual check"):
        sample = view[view["suspected_misinfo"]].sample(min(20, len(view[view["suspected_misinfo"]]))).loc[:, cols]
        st.dataframe(sample, use_container_width=True)

    # Explanations toggle
    st.checkbox("Show score breakdown for top posts", key="show_explain")
    if st.session_state.show_explain:
        def explain_row(r):
            text = str(r.get("content",""))
            return pd.Series({
                "rumor_hit": rumor_hit(text),
                "caps_ratio": caps_ratio(text),
                "punct_boost": punct_boost(text),
                "emoji_boost": emoji_boost(text),
                "emo_mix": (0.35*float(r.get("anger",0)) + 0.25*float(r.get("fear",0))
                            + 0.20*float(r.get("surprise",0)) + 0.10*float(r.get("disgust",0))
                            - 0.10*float(r.get("joy",0))),
                "extreme": abs(float(r.get("sentiment_compound",0))),
                "low_neu": max(0.0, 0.5 - float(r.get("sentiment_neu",0))) * 2.0,
            })
        # Need underlying emotion/sentiment columns for explain
        needed = ["anger","fear","surprise","disgust","joy","sentiment_compound","sentiment_neu"]
        merged = pd.merge(
            topn,
            view[["created_at","platform","content","url"] + needed],
            on=["created_at","platform","content","url"],
            how="left"
        )
        dbg = merged.apply(explain_row, axis=1)
        st.dataframe(pd.concat([topn.reset_index(drop=True), dbg], axis=1), use_container_width=True, height=520)

# -------------------- Downloads --------------------
st.subheader("Download")

# Filtered + scores
buf = io.StringIO()
out = view.copy()
out["created_at"] = out["created_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
out.to_csv(buf, index=False)
st.download_button("⬇️ Download FILTERED with scores (CSV)",
                   data=buf.getvalue().encode("utf-8"),
                   file_name="filtered_with_misinfo_scores.csv",
                   mime="text/csv")

# Suspected-only
sus_view = view[view["suspected_misinfo"]].copy()
if not sus_view.empty:
    buf2 = io.StringIO()
    s2 = sus_view.copy()
    s2["created_at"] = s2["created_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    s2.to_csv(buf2, index=False)
    st.download_button("⬇️ Download SUSPECTED only (CSV)",
                       data=buf2.getvalue().encode("utf-8"),
                       file_name="suspected_only.csv",
                       mime="text/csv")
