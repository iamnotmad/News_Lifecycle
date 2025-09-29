# app.py
import streamlit as st

st.set_page_config(page_title="News & Misinformation Analytics", layout="wide")
st.title("News & Misinformation Analytics")

st.markdown(
"""
Use the sidebar to open:
- **Overview & Trends** – activity, engagement, sentiment timelines (with filters & downloads)  
- **Emotion Analytics** – anger, sadness, joy, fear, surprise, disgust (with downloads)  
- **Misinformation Profiler (Heuristic)** – suspected misinfo scoring, timelines, platform shares, top posts, explanations (with downloads)
"""
)

#st.info("If you don’t see data, run: `python run_fetch.py` to (re)generate CSVs in `data/`.")
