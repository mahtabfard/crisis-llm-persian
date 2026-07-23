# scripts/build_instruction_splits_v2.py
"""
Builds instruction-tuning JSONL files with FOUR prompt variants per sample,
matching the CrisisSense-LLM paper's Appendix templates 1-4:

  Template 1: classify event type only
  Template 2: classify "useful for humanitarian aid" (informativeness) only
  Template 3: classify humanitarian aid type only
  Template 4: combined multi-label (event + useful + humanitarian)

Each input row -> 4 output records. Final dataset size = 4x input rows.

Usage:
  python build_instruction_splits_v2.py
"""

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

TRAIN_OUTPUT = OUTPUT_DIR / "train_v2.jsonl"
VAL_OUTPUT = OUTPUT_DIR / "val_v2.jsonl"
TEST_OUTPUT = OUTPUT_DIR / "test_v2.jsonl"


# =========================================================
# Label vocabularies (from the paper's Appendix prompts)
# =========================================================

EVENT_TYPES = [
    "HURRICANE", "FLOOD", "EARTHQUAKE", "DISASTER", "EVENTS", "EXPLOSION",
    "BOMBING", "FIRE", "LANDSLIDE", "CRASH", "DISEASE", "SHOOTING",
    "COLLAPSE", "HAZARD", "VOLCANO",
]

HUMANITARIAN_TYPES = [
    "NOT HUMANITARIAN", "OTHER RELEVANT INFORMATION", "DONATION AND VOLUNTEERING",
    "REQUESTS OR NEEDS", "SYMPATHY AND SUPPORT", "INFRASTRUCTURE AND UTILITY DAMAGE",
    "AFFECTED INDIVIDUAL", "CAUTION AND ADVICE", "INJURED OR DEAD PEOPLE",
    "DISEASE RELATED", "RESPONSE EFFORTS", "PERSONAL UPDATE",
    "MISSING AND FOUND PEOPLE", "DISPLACED AND EVACUATION",
    "PHYSICAL LANDSLIDE", "TERRORISM RELATED",
]

EVENT_LIST_STR = "\n".join(EVENT_TYPES)
HUMANITARIAN_LIST_STR = "\n".join(HUMANITARIAN_TYPES)


# =========================================================
# Mapping helpers: your normalized labels -> paper's label vocab
# =========================================================

# Your dataset_builder.py produces these event_label values:
#   earthquake, flood, hurricane, tornado, wildfire, explosion,
#   landslide, shooting, crash, other
EVENT_LABEL_MAP = {
    "earthquake": "EARTHQUAKE",
    "flood": "FLOOD",
    "hurricane": "HURRICANE",
    "tornado": "DISASTER",       # paper has no TORNADO category -> bucket as DISASTER
    "wildfire": "FIRE",
    "explosion": "EXPLOSION",
    "landslide": "LANDSLIDE",
    "shooting": "SHOOTING",
    "crash": "CRASH",
    "other": "DISASTER",
}

# Your dataset_builder.py produces these human_label values:
#   not_humanitarian, infrastructure_damage, donation_volunteering,
#   sympathy_support, affected_individuals, missing_or_found,
#   caution_advice, requests_or_needs, other
HUMANITARIAN_LABEL_MAP = {
    "not_humanitarian": "NOT HUMANITARIAN",
    "infrastructure_damage": "INFRASTRUCTURE AND UTILITY DAMAGE",
    "donation_volunteering": "DONATION AND VOLUNTEERING",
    "sympathy_support": "SYMPATHY AND SUPPORT",
    "affected_individuals": "AFFECTED INDIVIDUAL",
    "missing_or_found": "MISSING AND FOUND PEOPLE",
    "caution_advice": "CAUTION AND ADVICE",
    "requests_or_needs": "REQUESTS OR NEEDS",
    "other": "OTHER RELEVANT INFORMATION",
}

# info_label is "yes"/"no" -> paper's "useful" is True/False
INFO_LABEL_MAP = {
    "yes": "True",
    "no": "False",
}


def map_event(label: str) -> str:
    return EVENT_LABEL_MAP.get(str(label).lower().strip(), "DISASTER")


def map_humanitarian(label: str) -> str:
    return HUMANITARIAN_LABEL_MAP.get(str(label).lower().strip(), "OTHER RELEVANT INFORMATION")


def map_useful(label: str) -> str:
    return INFO_LABEL_MAP.get(str(label).lower().strip(), "False")


# =========================================================
# Prompt template builders (Appendix templates 1-4)
# =========================================================

def template_1_event(text: str, event_value: str) -> dict:
    instruction = (
        "### Instruction:\n"
        "For the following text, below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n"
        "INSTRUCTION: This is a classification task.\n"
        f"Classify the event type the text described into one of the following 14 categories:\n"
        f"{EVENT_LIST_STR}\n"
        "Format the output as JSON with the following keys: event type\n\n"
        f"text: {text}\n"
        "### Response:"
    )
    response = {"event type": event_value}
    return {"instruction": instruction, "response": response, "template": 1}


def template_2_useful(text: str, useful_value: str) -> dict:
    instruction = (
        "### Instruction:\n"
        "For the following text, below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n"
        "INSTRUCTION: This is a classification task.\n"
        "Is the given text useful for humanitarian aid\n"
        "Answer True if yes, False if not or unknown.\n"
        "Format the output as JSON with the following key: useful\n\n"
        f"text: {text}\n"
        "### Response:"
    )
    response = {"useful": useful_value}
    return {"instruction": instruction, "response": response, "template": 2}


def template_3_humanitarian(text: str, human_value: str) -> dict:
    instruction = (
        "### Instruction:\n"
        "For the following text, below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n"
        "INSTRUCTION: This is a classification task.\n"
        f"Classify the humanitarian aid type the text described into one of the following 16 types:\n"
        f"{HUMANITARIAN_LIST_STR}\n"
        "Format the output as JSON with the following keys: humanitarian aid type\n\n"
        f"text: {text}\n"
        "### Response:"
    )
    response = {"humanitarian aid type": human_value}
    return {"instruction": instruction, "response": response, "template": 3}


def template_4_combined(text: str, event_value: str, useful_value: str, human_value: str) -> dict:
    instruction = (
        "### Instruction:\n"
        "For the following text, below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n"
        "This is a multi-label classification task.\n"
        f"FIRST, classify the event type the text described into one of the following 14 categories:\n"
        f"{EVENT_LIST_STR}\n"
        "SECOND, is the given text useful for humanitarian aid\n"
        "Answer True if yes, False if not or unknown.\n"
        f"THIRD, classify the humanitarian aid type the text described into one of the following 16 types:\n"
        f"{HUMANITARIAN_LIST_STR}\n"
        "If you don't know the answer, please just say that you don't know the answer. "
        "Don't make up an answer.\n"
        "Format the output as a JSON with the following keys, the JSON MUST contain these three keys:\n"
        "Event type:\nUseful:\nHumanitarian aid type:\n\n"
        f"text: {text}\n"
        "Response:"
    )
    response = {
        "Event type": event_value,
        "Useful": useful_value,
        "Humanitarian aid type": human_value,
    }
    return {"instruction": instruction, "response": response, "template": 4}


# =========================================================
# Core conversion
# =========================================================

def row_to_records(row: pd.Series) -> list:
    text = str(row["text"])
    event_value = map_event(row["event_label"])
    useful_value = map_useful(row["info_label"])
    human_value = map_humanitarian(row["human_label"])

    return [
        template_1_event(text, event_value),
        template_2_useful(text, useful_value),
        template_3_humanitarian(text, human_value),
        template_4_combined(text, event_value, useful_value, human_value),
    ]


def export_jsonl(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            for record in row_to_records(row):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1

    print(f"  Wrote {n_written:,} records -> {output_path}")


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Dataset is empty: {path}")
    return df


# =========================================================
# Main
# =========================================================

def main() -> None:
    print("\nLoading datasets...")
    train_df = load_dataframe(TRAIN_INPUT)
    val_df = load_dataframe(VAL_INPUT)
    test_df = load_dataframe(TEST_INPUT)

    print(f"  Train: {len(train_df):,} rows")
    print(f"  Val:   {len(val_df):,} rows")
    print(f"  Test:  {len(test_df):,} rows")

    print("\nExporting instruction datasets (4 variants per row)...")
    export_jsonl(train_df, TRAIN_OUTPUT)
    export_jsonl(val_df, VAL_OUTPUT)
    export_jsonl(test_df, TEST_OUTPUT)

    print("\nDone. Note: TEST set also has 4 variants per row, but for")
    print("inference/evaluation you'll mainly use template 4 (combined)")
    print("records from test_v2.jsonl, since that's what the paper")
    print("evaluates overall accuracy on.")


if __name__ == "__main__":
    main()
