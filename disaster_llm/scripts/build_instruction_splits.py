# scripts/build_instruction_splits.py

from pathlib import Path
import json
import pandas as pd


# =========================================================
# Paths
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

TRAIN_INPUT = ROOT_DIR / "data" / "interim" / "train_augmented.csv"
VAL_INPUT = ROOT_DIR / "data" / "interim" / "val.csv"
TEST_INPUT = ROOT_DIR / "data" / "interim" / "test.csv"

OUTPUT_DIR = ROOT_DIR / "data" / "processed"

TRAIN_OUTPUT = OUTPUT_DIR / "train.jsonl"
VAL_OUTPUT = OUTPUT_DIR / "val.jsonl"
TEST_OUTPUT = OUTPUT_DIR / "test.jsonl"


# =========================================================
# Prompt Template
# =========================================================

SYSTEM_INSTRUCTION = (
    "Given the following tweet, classify:\n"
    "1. The disaster event type.\n"
    "2. Whether the tweet is informative.\n"
    "3. The humanitarian aid category."
)


# =========================================================
# Utilities
# =========================================================

def load_dataframe(path: Path) -> pd.DataFrame:
    """
    Load CSV dataset safely.
    """

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Dataset is empty: {path}")

    return df


def build_instruction(row: pd.Series) -> dict:
    """
    Convert one row into instruction-tuning format.
    """

    instruction = (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Tweet: {row['text']}"
    )

    response = {
        "event": row["event_label"],
        "informative": row["info_label"],
        "humanitarian": row["human_label"]
    }

    return {
        "instruction": instruction,
        "response": response
    }


def export_jsonl(df: pd.DataFrame, output_path: Path) -> None:
    """
    Export dataframe into JSONL instruction dataset.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w", encoding="utf-8") as f:

        for _, row in df.iterrows():

            sample = build_instruction(row)

            f.write(
                json.dumps(
                    sample,
                    ensure_ascii=False
                ) + "\n"
            )


def print_statistics(
    name: str,
    df: pd.DataFrame
) -> None:
    """
    Print dataset summary.
    """

    print(f"\n{name} Statistics")
    print("=" * 50)

    print(f"Samples: {len(df):,}")

    print("\nEvent Distribution:")
    print(
        df["event_label"]
        .value_counts(normalize=True)
        .round(4)
    )

    print("=" * 50)


# =========================================================
# Main
# =========================================================

def main() -> None:

    print("\nLoading datasets...")

    train_df = load_dataframe(TRAIN_INPUT)
    val_df = load_dataframe(VAL_INPUT)
    test_df = load_dataframe(TEST_INPUT)

    print("\nExporting instruction datasets...")

    export_jsonl(train_df, TRAIN_OUTPUT)
    export_jsonl(val_df, VAL_OUTPUT)
    export_jsonl(test_df, TEST_OUTPUT)

    print_statistics("Train", train_df)
    print_statistics("Validation", val_df)
    print_statistics("Test", test_df)

    print("\nSaved instruction datasets:")
    print(f" - {TRAIN_OUTPUT}")
    print(f" - {VAL_OUTPUT}")
    print(f" - {TEST_OUTPUT}")

    print("\nInstruction dataset generation completed successfully.")


if __name__ == "__main__":
    main()