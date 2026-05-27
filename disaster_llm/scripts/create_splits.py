# scripts/create_splits.py

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

# =========================================================
# Paths
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT_DIR / "data" / "interim" / "cleaned.csv"

TRAIN_PATH = ROOT_DIR / "data" / "interim" / "train.csv"
VAL_PATH = ROOT_DIR / "data" / "interim" / "val.csv"
TEST_PATH = ROOT_DIR / "data" / "interim" / "test.csv"

# =========================================================
# Split Configuration
# =========================================================

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# =========================================================
# Utilities
# =========================================================

def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load cleaned dataset.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    return df


def create_stratification_labels(df: pd.DataFrame) -> pd.Series:
    """
    Create composite labels for stratified splitting.
    """
    return (
        df["event_label"].astype(str)
        + "__"
        + df["info_label"].astype(str)
    )


def save_split(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save split dataframe.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def print_split_statistics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> None:
    """
    Print split statistics.
    """
    total = len(train_df) + len(val_df) + len(test_df)

    print("\nDataset Split Summary")
    print("=" * 40)

    print(f"Train: {len(train_df):,} samples ({len(train_df)/total:.1%})")
    print(f"Validation: {len(val_df):,} samples ({len(val_df)/total:.1%})")
    print(f"Test: {len(test_df):,} samples ({len(test_df)/total:.1%})")

    print("=" * 40)


# =========================================================
# Main
# =========================================================

def main() -> None:

    print("\nLoading cleaned dataset...")
    df = load_dataset(INPUT_PATH)

    print(f"Loaded {len(df):,} samples.")

    # Create stratification labels
    stratify_labels = create_stratification_labels(df)

    # -----------------------------------------------------
    # Train / Temp Split
    # -----------------------------------------------------

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - TRAIN_SIZE),
        random_state=SEED,
        shuffle=True,
        stratify=stratify_labels
    )

    # -----------------------------------------------------
    # Validation / Test Split
    # -----------------------------------------------------

    temp_stratify = create_stratification_labels(temp_df)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        shuffle=True,
        stratify=temp_stratify
    )

    # -----------------------------------------------------
    # Save Splits
    # -----------------------------------------------------

    save_split(train_df, TRAIN_PATH)
    save_split(val_df, VAL_PATH)
    save_split(test_df, TEST_PATH)

    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------

    print_split_statistics(train_df, val_df, test_df)

    print("\nSaved files:")
    print(f" - {TRAIN_PATH}")
    print(f" - {VAL_PATH}")
    print(f" - {TEST_PATH}")

    print("\nDataset splitting completed successfully.")
    print("\nTrain Event Distribution:")
    print(train_df["event_label"].value_counts(normalize=True).round(3))

    print("\nValidation Event Distribution:")
    print(val_df["event_label"].value_counts(normalize=True).round(3))

    print("\nTest Event Distribution:")
    print(test_df["event_label"].value_counts(normalize=True).round(3))

if __name__ == "__main__":
    main()