# scripts/augment_train.py

from pathlib import Path
from collections import Counter
import random
import re

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import wordnet


# =========================================================
# Configuration
# =========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

ROOT_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT_DIR / "data" / "interim" / "train.csv"
OUTPUT_PATH = ROOT_DIR / "data" / "interim" / "train_augmented.csv"

TARGET_COLUMN = "human_label"

MINORITY_THRESHOLD_RATIO = 0.50
MAX_REPLACEMENTS = 1

# =========================================================
# NLTK Setup
# =========================================================

def ensure_nltk_resources() -> None:
    """
    Download required NLTK resources if unavailable.
    """
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")
        nltk.download("omw-1.4")


# =========================================================
# Text Augmentation
# =========================================================

def synonym_replacement(
    text: str,
    max_replacements: int = 1
) -> str:
    """
    Replace limited words with WordNet synonyms.
    Conservative augmentation for semantic preservation.
    """

    tokens = re.findall(r"\w+|\W+", text)

    candidate_indices = [
        i for i, token in enumerate(tokens)
        if re.match(r"^\w+$", token) and len(token) > 3
    ]

    random.shuffle(candidate_indices)

    replacements = 0

    for idx in candidate_indices:

        original_word = tokens[idx]

        synsets = wordnet.synsets(original_word)

        synonyms = {
            lemma.name().replace("_", " ")
            for synset in synsets
            for lemma in synset.lemmas()
        }

        synonyms.discard(original_word)

        if not synonyms:
            continue

        replacement = random.choice(list(synonyms))

        if replacement.lower() == original_word.lower():
            continue

        tokens[idx] = replacement
        replacements += 1

        if replacements >= max_replacements:
            break

    return "".join(tokens)


# =========================================================
# Dataset Utilities
# =========================================================

def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load training dataset.
    """

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Training dataset is empty.")

    return df


def identify_minority_classes(
    df: pd.DataFrame,
    target_column: str
) -> list:
    """
    Identify minority classes based on ratio threshold.
    """

    counts = df[target_column].value_counts()

    max_count = counts.max()

    minority_classes = counts[
        counts < (max_count * MINORITY_THRESHOLD_RATIO)
    ].index.tolist()

    return minority_classes


# =========================================================
# Augmentation Pipeline
# =========================================================

def augment_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply augmentation only to minority classes.
    """

    minority_classes = identify_minority_classes(
        df,
        TARGET_COLUMN
    )

    print("\nMinority Classes:")
    for cls in minority_classes:
        print(f" - {cls}")

    augmented_rows = []

    for _, row in df.iterrows():

        augmented_rows.append(row.to_dict())

        if row[TARGET_COLUMN] not in minority_classes:
            continue

        augmented_text = synonym_replacement(
            row["text"],
            max_replacements=MAX_REPLACEMENTS
        )

        if augmented_text.strip() == row["text"].strip():
            continue

        new_row = row.copy()
        new_row["text"] = augmented_text

        augmented_rows.append(new_row.to_dict())

    augmented_df = pd.DataFrame(augmented_rows)

    return augmented_df.sample(
        frac=1.0,
        random_state=SEED
    ).reset_index(drop=True)


# =========================================================
# Reporting
# =========================================================

def print_statistics(
    original_df: pd.DataFrame,
    augmented_df: pd.DataFrame
) -> None:

    print("\nDataset Statistics")
    print("=" * 50)

    print(f"Original samples : {len(original_df):,}")
    print(f"Augmented samples: {len(augmented_df):,}")

    growth = (
        (len(augmented_df) - len(original_df))
        / len(original_df)
    ) * 100

    print(f"Growth           : {growth:.2f}%")

    print("=" * 50)

    print("\nHumanitarian Label Distribution:")
    print(
        augmented_df[TARGET_COLUMN]
        .value_counts(normalize=True)
        .round(4)
    )


# =========================================================
# Main
# =========================================================

def main() -> None:

    print("\nLoading training dataset...")
    train_df = load_dataset(TRAIN_PATH)

    print(f"Loaded {len(train_df):,} training samples.")

    ensure_nltk_resources()

    print("\nApplying augmentation pipeline...")
    augmented_df = augment_dataset(train_df)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    augmented_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print_statistics(
        train_df,
        augmented_df
    )

    print(f"\nSaved augmented dataset:")
    print(f" - {OUTPUT_PATH}")

    print("\nAugmentation completed successfully.")


if __name__ == "__main__":
    main()