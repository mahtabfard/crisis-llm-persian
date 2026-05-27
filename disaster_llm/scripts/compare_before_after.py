#!/usr/bin/env python3
"""
Compare label distributions:
1) Cleaned (before balancing)
2) After oversampling
3) After augmentation

Outputs grouped bar charts (paper-ready).
"""
#G:\Text_Classification\disaster_llm\scripts\compare_before_after.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# -------------------------
# Paths
# -------------------------
ROOT = Path(".")
CLEAN = ROOT / "data/interim/cleaned.csv"
OVER = ROOT / "data/interim/cleaned_oversampled.csv"
AUG = ROOT / "data/interim/cleaned_augmented.csv"
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)

# -------------------------
# Load
# -------------------------
df_clean = pd.read_csv(CLEAN)
df_over = pd.read_csv(OVER)
df_aug = pd.read_csv(AUG)

# -------------------------
# Helpers
# -------------------------
def plot_compare(counter1, counter2, counter3, title, filename):
    labels = sorted(set(counter1) | set(counter2) | set(counter3))
    x = range(len(labels))
    width = 0.25

    y1 = [counter1.get(l, 0) for l in labels]
    y2 = [counter2.get(l, 0) for l in labels]
    y3 = [counter3.get(l, 0) for l in labels]

    plt.figure(figsize=(13,6))
    plt.bar([i - width for i in x], y1, width, label="Before (cleaned)")
    plt.bar(x, y2, width, label="After oversampling")
    plt.bar([i + width for i in x], y3, width, label="After augmentation")

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Number of samples")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / filename)
    plt.close()

# -------------------------
# Counters
# -------------------------
human_clean = Counter(df_clean["human_label"])
human_over  = Counter(df_over["human_label"])
human_aug   = Counter(df_aug["human_label"])

event_clean = Counter(df_clean["event_label"])
event_over  = Counter(df_over["event_label"])
event_aug   = Counter(df_aug["event_label"])

info_clean = Counter(df_clean["info_label"])
info_over  = Counter(df_over["info_label"])
info_aug   = Counter(df_aug["info_label"])

# -------------------------
# Plots
# -------------------------
plot_compare(
    human_clean, human_over, human_aug,
    "Humanitarian Aid Label Distribution (Before vs After)",
    "compare_human_labels.png"
)

plot_compare(
    event_clean, event_over, event_aug,
    "Event Type Distribution (Before vs After)",
    "compare_event_labels.png"
)

plot_compare(
    info_clean, info_over, info_aug,
    "Informativeness Distribution (Before vs After)",
    "compare_info_labels.png"
)

print("Comparison plots saved to reports/")
