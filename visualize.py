# visualize.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

EMO_KEYS = ["anger","sadness","joy","fear","surprise","disgust"]
os.makedirs("figs", exist_ok=True)

def _safe_read(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

def plot_emotion_stacked_area(daily_emo_csv="data/daily_emotions.csv", out="figs/emotions_stacked_area.png"):
    de = _safe_read(daily_emo_csv)
    if de.empty:
        print("No daily_emotions.csv found or empty.")
        return
    de["date"] = pd.to_datetime(de["date"])
    de = de.sort_values("date")
    X = de["date"]
    Y = [de[k].fillna(0).values for k in EMO_KEYS]

    plt.figure(figsize=(12,6))
    plt.stackplot(X, Y, labels=EMO_KEYS)
    plt.title("Daily Emotion Intensity (mean per day)")
    plt.xlabel("Date"); plt.ylabel("Intensity")
    plt.legend(loc="upper left", ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved:", out)

def plot_emotion_platform_bars(combined_csv="data/combined_with_emotions.csv", out="figs/emotions_platform_bars.png"):
    df = _safe_read(combined_csv)
    if df.empty:
        print("No combined_with_emotions.csv found or empty.")
        return
    plat = df.groupby("platform")[EMO_KEYS].mean().reindex(sorted(df["platform"].unique()))
    idx = np.arange(len(plat))
    width = 0.12

    plt.figure(figsize=(12,6))
    for i, emo in enumerate(EMO_KEYS):
        plt.bar(idx + i*width, plat[emo].values, width=width, label=emo)
    plt.xticks(idx + width*(len(EMO_KEYS)-1)/2, plat.index, rotation=0)
    plt.title("Average Emotion by Platform")
    plt.ylabel("Mean intensity")
    plt.legend(ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved:", out)

def plot_emotion_radar(combined_csv="data/combined_with_emotions.csv", platform=None, out="figs/emotions_radar.png"):
    df = _safe_read(combined_csv)
    if df.empty:
        print("No combined_with_emotions.csv found or empty.")
        return
    if platform:
        df = df[df["platform"] == platform]
        title = f"Emotion Radar – {platform}"
    else:
        title = "Emotion Radar – All Platforms"

    vals = df[EMO_KEYS].mean().fillna(0).values
    vals = np.append(vals, vals[0])  # close loop
    labels = EMO_KEYS + [EMO_KEYS[0]]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, vals, linewidth=2)
    ax.fill(angles, vals, alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved:", out)

if __name__ == "__main__":
    plot_emotion_stacked_area()
    plot_emotion_platform_bars()
    plot_emotion_radar(platform=None)  # or e.g., platform="reddit"
