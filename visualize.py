# visualize.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

EMO_KEYS = ["anger","sadness","joy","fear","surprise","disgust"]
os.makedirs("figs", exist_ok=True)

def _safe_read(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

def _have_cols(df, cols):
    return [c for c in cols if c in df.columns]

def plot_emotion_stacked_area(daily_emo_csv="data/daily_emotions.csv",
                              out="figs/emotions_stacked_area.png"):
    de = _safe_read(daily_emo_csv)
    if de.empty:
        print("No daily_emotions.csv found or empty.")
        return

    if "date" not in de.columns:
        print("daily_emotions.csv missing 'date' column.")
        return

    # Only keep emotion columns that exist
    cols = _have_cols(de, EMO_KEYS)
    if not cols:
        print("daily_emotions.csv has no emotion columns to plot.")
        return

    de["date"] = pd.to_datetime(de["date"], errors="coerce")
    de = de.dropna(subset=["date"]).sort_values("date")

    X = de["date"]
    Y = [de[c].fillna(0).values for c in cols]

    plt.figure(figsize=(12,6))
    plt.stackplot(X, Y, labels=cols)
    plt.title("Daily Emotion Intensity (mean per day)")
    plt.xlabel("Date")
    plt.ylabel("Intensity")
    plt.legend(loc="upper left", ncol=min(3, len(cols)), frameon=False)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved:", out)

def plot_emotion_platform_bars(combined_csv="data/combined_with_emotions.csv",
                               out="figs/emotions_platform_bars.png"):
    df = _safe_read(combined_csv)
    if df.empty:
        print("No combined_with_emotions.csv found or empty.")
        return

    cols = _have_cols(df, EMO_KEYS)
    if not cols:
        print("combined_with_emotions.csv has no emotion columns to plot.")
        return

    if "platform" not in df.columns:
        print("combined_with_emotions.csv missing 'platform' column.")
        return

    plat = df.groupby("platform")[cols].mean().reindex(sorted(df["platform"].dropna().unique()))
    if plat.empty:
        print("No platform data to plot.")
        return

    idx = np.arange(len(plat))
    width = 0.8 / max(1, len(cols))  # keep bars within the axis

    plt.figure(figsize=(12,6))
    for i, emo in enumerate(cols):
        plt.bar(idx + i*width, plat[emo].fillna(0).values, width=width, label=emo)
    plt.xticks(idx + width*(len(cols)-1)/2, plat.index, rotation=0)
    plt.title("Average Emotion by Platform")
    plt.ylabel("Mean intensity")
    plt.legend(ncol=min(3, len(cols)), frameon=False)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved:", out)

def plot_emotion_radar(combined_csv="data/combined_with_emotions.csv",
                       platform=None,
                       out="figs/emotions_radar.png"):
    df = _safe_read(combined_csv)
    if df.empty:
        print("No combined_with_emotions.csv found or empty.")
        return

    cols = _have_cols(df, EMO_KEYS)
    if not cols:
        print("combined_with_emotions.csv has no emotion columns to plot.")
        return

    data = df if platform is None else df[df.get("platform") == platform]
    title = f"Emotion Radar – {platform}" if platform else "Emotion Radar – All Platforms"
    if data.empty:
        print(f"No rows to plot for platform={platform!r}.")
        return

    vals = data[cols].mean().fillna(0).values
    # close loop
    vals = np.append(vals, vals[0])
    labels = cols + [cols[0]]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, vals, linewidth=2)
    ax.fill(angles, vals, alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    # Optional: bound if your emotions are in [0,1]
    # ax.set_ylim(0, 1)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved:", out)

if __name__ == "__main__":
    plot_emotion_stacked_area()
    plot_emotion_platform_bars()
    plot_emotion_radar(platform=None)  # or e.g., platform="reddit"
