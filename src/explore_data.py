"""Plot the landmark dataset's class distribution and per-class detection failure rate.

Reads:  data/processed/landmarks.csv, data/processed/extraction_stats.csv
Writes: reports/class_distribution.png, reports/failure_rate.png

Usage:
    python src/explore_data.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # write files only; no GUI window needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import NO_HAND_LABEL, PROJECT_ROOT

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"


def plot_class_distribution(landmarks: pd.DataFrame) -> None:
    """Bar chart of how many usable landmark samples each class has."""
    counts = landmarks["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x=counts.index, y=counts.values, color="#4C72B0", ax=ax)
    ax.axhline(counts.mean(), color="black", linestyle="--", linewidth=1, label=f"mean = {counts.mean():.0f}")
    ax.set(title="Landmark samples per class", xlabel="Class", ylabel="Samples")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)


def plot_failure_rate(stats: pd.DataFrame) -> None:
    """Bar chart of the MediaPipe detection failure rate per class.

    For "nothing" the bar shows the false-positive rate (a hand was found where
    none exists), so it is drawn in a different color.
    """
    stats = stats.sort_values("label")
    colors = ["#C44E52" if lbl == NO_HAND_LABEL else "#4C72B0" for lbl in stats["label"]]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(stats["label"], stats["failure_rate"] * 100, color=colors)
    # A 0% bar is invisible, so label the "nothing" value explicitly.
    for x, (lbl, rate) in enumerate(zip(stats["label"], stats["failure_rate"])):
        if lbl == NO_HAND_LABEL:
            ax.annotate(f"{rate:.0%}", (x, rate * 100), xytext=(0, 3), textcoords="offset points",
                        ha="center", color="#C44E52", fontsize=9)
    ax.tick_params(axis="x", labelrotation=45)
    ax.set(
        title="Hand detection failure rate per class\n(red = 'nothing': false-positive rate)",
        xlabel="Class",
        ylabel="Failure rate (%)",
    )
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "failure_rate.png", dpi=150)
    plt.close(fig)


def main() -> None:
    landmarks = pd.read_csv(PROCESSED_DIR / "landmarks.csv")
    stats = pd.read_csv(PROCESSED_DIR / "extraction_stats.csv")
    REPORTS_DIR.mkdir(exist_ok=True)

    plot_class_distribution(landmarks)
    plot_failure_rate(stats)

    print(f"Samples: {len(landmarks)} | Classes: {landmarks['label'].nunique()} | Features: {landmarks.shape[1] - 1}")
    print(stats[["label", "total", "detected", "failure_rate"]].to_string(index=False))
    print(f"Plots saved to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
