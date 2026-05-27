#!/usr/bin/env python3
"""
Run oversampling, lightweight augmentation, plot distributions, and produce a PDF report.
Saves:
 - data/interim/cleaned_oversampled.csv
 - data/interim/cleaned_augmented.csv
 - reports/human_label_dist.png
 - reports/event_label_dist.png
 - reports/info_label_dist.png
 - reports/dataset_report.pdf
"""
#G:\Text_Classification\disaster_llm\scripts\run_augment_and_report.py
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import json
import random
import textwrap
import nltk
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import utils

# -------------------------
# Config / paths
# -------------------------
ROOT = Path(".")
CLEAN_CSV = ROOT / "data/interim/cleaned.csv"
OVERSAMPLED_CSV = ROOT / "data/interim/cleaned_oversampled.csv"
AUGMENTED_CSV = ROOT / "data/interim/cleaned_augmented.csv"
INSTRUCTIONS = ROOT / "data/processed/instructions.jsonl"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_HUMAN = REPORT_DIR / "human_label_dist.png"
FIG_EVENT = REPORT_DIR / "event_label_dist.png"
FIG_INFO = REPORT_DIR / "info_label_dist.png"
PDF_REPORT = REPORT_DIR / "dataset_report.pdf"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# -------------------------
# Utilities
# -------------------------
def safe_read_csv(p: Path):
    return pd.read_csv(p, dtype=str).fillna("")

def ensure_wordnet():
    try:
        nltk.data.find("corpora/wordnet")
    except Exception:
        nltk.download("wordnet")
        nltk.download("omw-1.4")

def synonym_replace_simple(text: str, max_replacements: int = 1) -> str:
    """
    Small, conservative synonym replacer using WordNet.
    If no synonyms found, returns original.
    """
    from nltk.corpus import wordnet
    import re
    toks = re.findall(r"\w+|\W+", text)
    candidate_idxs = [i for i,t in enumerate(toks) if re.match(r"^\w+$", t) and len(t)>2]
    random.shuffle(candidate_idxs)
    replaced = 0
    for i in candidate_idxs:
        w = toks[i]
        synsets = wordnet.synsets(w)
        lemmas = {l.name().replace("_"," ") for s in synsets for l in s.lemmas()}
        lemmas.discard(w)
        if lemmas:
            toks[i] = random.choice(list(lemmas))
            replaced += 1
        if replaced >= max_replacements:
            break
    return "".join(toks)

def plot_and_save_counts(counter: Counter, title: str, out_path: Path, top_n: int = None):
    labels, counts = zip(*counter.most_common()) if top_n is None else zip(*counter.most_common(top_n))
    plt.figure(figsize=(10,6))
    plt.bar(range(len(labels)), counts)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def imread_for_report(path, width):
    img = utils.ImageReader(str(path))
    iw, ih = img.getSize()
    aspect = ih/iw
    return Image(str(path), width=width, height=(width * aspect))

# -------------------------
# 1) Load cleaned CSV and show counts
# -------------------------
print("Loading cleaned CSV:", CLEAN_CSV)
df = safe_read_csv(CLEAN_CSV)
print("Rows:", len(df))

human_counts = Counter(df["human_label"].tolist())
event_counts = Counter(df["event_label"].tolist())
info_counts = Counter(df["info_label"].tolist())

print("\nHumanitarian counts:")
for k,v in human_counts.most_common():
    print(f"  {k}: {v}")
print("\nEvent counts (top 10):")
for k,v in event_counts.most_common(10):
    print(f"  {k}: {v}")
print("\nInformativeness counts:")
for k,v in info_counts.most_common():
    print(f"  {k}: {v}")

# -------------------------
# 2) Oversample to balance humanitarian labels (to max class count)
# -------------------------
max_count = max(human_counts.values())
frames = []
for label, grp in df.groupby("human_label"):
    n = len(grp)
    if n >= max_count:
        frames.append(grp.sample(n=max_count, replace=False, random_state=SEED))
    else:
        # oversample with replacement
        frames.append(grp.sample(n=max_count, replace=True, random_state=SEED))
oversampled = pd.concat(frames).sample(frac=1, random_state=SEED).reset_index(drop=True)
oversampled.to_csv(OVERSAMPLED_CSV, index=False)
print("\nSaved oversampled CSV:", OVERSAMPLED_CSV, "rows:", len(oversampled))

# -------------------------
# 3) Lightweight augmentation for minority humanitarian classes
# -------------------------
# choose minority classes: those with count < median
counts = oversampled["human_label"].value_counts()
median = counts.median()
minority = counts[counts < median].index.tolist()
print("\nMinority classes for augmentation:", minority)

ensure_wordnet()  # download if needed
rows = []
for _, r in oversampled.iterrows():
    rows.append(r)
    if r["human_label"] in minority:
        # produce 1 synthetic example per original (conservative)
        aug_text = synonym_replace_simple(r["text"], max_replacements=1)
        if aug_text != r["text"]:
            new = r.copy()
            new["text"] = aug_text
            rows.append(new)
        else:
            # if no replacement found, duplicate once
            rows.append(r.copy())
aug = pd.DataFrame(rows).reset_index(drop=True)
aug.to_csv(AUGMENTED_CSV, index=False)
print("Saved augmented CSV:", AUGMENTED_CSV, "rows:", len(aug))

# -------------------------
# 4) Plot distributions and save PNGs
# -------------------------
plot_and_save_counts(Counter(aug["human_label"].tolist()), "Humanitarian label distribution (augmented)", FIG_HUMAN)
plot_and_save_counts(Counter(aug["event_label"].tolist()), "Event label distribution (augmented)", FIG_EVENT)
plot_and_save_counts(Counter(aug["info_label"].tolist()), "Informativeness distribution (augmented)", FIG_INFO)
print("Saved plots:", FIG_HUMAN, FIG_EVENT, FIG_INFO)

# -------------------------
# 5) Load instruction JSONL head (first 10) for report
# -------------------------
inst_examples = []
if INSTRUCTIONS.exists():
    with open(INSTRUCTIONS, "r", encoding="utf-8") as fh:
        for i,line in enumerate(fh):
            if i>=10: break
            inst_examples.append(json.loads(line))
else:
    print("Warning: instructions file not found at", INSTRUCTIONS)

# -------------------------
# 6) Create PDF report
# -------------------------
print("Building PDF report:", PDF_REPORT)
styles = getSampleStyleSheet()
doc = SimpleDocTemplate(str(PDF_REPORT), pagesize=A4, title="Dataset Quality Report")
story = []

story.append(Paragraph("Dataset Quality Report", styles["Title"]))
story.append(Spacer(1,12))

# brief stats table
stats_table = [
    ["Metric", "Value"],
    ["Original rows", str(len(df))],
    ["Oversampled rows", str(len(oversampled))],
    ["Augmented rows", str(len(aug))]
]
table = Table(stats_table, hAlign="LEFT")
story.append(table)
story.append(Spacer(1,12))

# add distributions images
story.append(Paragraph("Label Distributions (augmented)", styles["Heading2"]))
story.append(Spacer(1,6))
story.append(imread_for_report(FIG_HUMAN, width=450))
story.append(Spacer(1,6))
story.append(imread_for_report(FIG_EVENT, width=450))
story.append(Spacer(1,6))
story.append(imread_for_report(FIG_INFO, width=450))
story.append(Spacer(1,12))

# add sample instruction examples
story.append(Paragraph("Sample Instructions (first 10)", styles["Heading2"]))
for ex in inst_examples:
    instr = ex.get("instruction", "")
    resp = ex.get("response", {})
    story.append(Paragraph("<b>Instruction:</b>", styles["Normal"]))
    for line in textwrap.wrap(instr, width=140):
        story.append(Paragraph(line, styles["Code"]))
    story.append(Paragraph("<b>Response:</b> " + json.dumps(resp, ensure_ascii=False), styles["Normal"]))
    story.append(Spacer(1,6))

# add first rows of cleaned CSV
story.append(Paragraph("Cleaned CSV (first 10 rows)", styles["Heading2"]))
sample_rows = df.head(10).to_dict(orient="records")
for r in sample_rows:
    row_text = " | ".join([f"{k}: {v}" for k,v in r.items()])
    for line in textwrap.wrap(row_text, width=140):
        story.append(Paragraph(line, styles["Code"]))
    story.append(Spacer(1,6))

doc.build(story)
print("PDF report saved to:", PDF_REPORT)

print("\nDONE.")
