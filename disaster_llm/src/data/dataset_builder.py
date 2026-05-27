#G:\Text_Classification\disaster_llm\src\data\dataset_builder.py
from pathlib import Path
import pandas as pd
from typing import List
from src.utils.logger import get_logger

logger = get_logger(__name__)

class CrisisBenchBuilder:
    """
    Combines all TSV files from all_data_en into a single cleaned dataset.
    Normalizes labels and prepares for instruction tuning.
    """

    HUMANITARIAN_LABELS = {
        "not_humanitarian": "not_humanitarian",
        "infrastructure_and_utilities_damage": "infrastructure_damage",
        "donation_and_volunteering": "donation_volunteering",
        "sympathy_and_support": "sympathy_support",
        "affected_individuals": "affected_individuals",
        "missing_trapped_found_people": "missing_or_found",
        "caution_and_advice": "caution_advice",
        "requests_or_needs": "requests_or_needs",
    }

    EVENT_MAP = {
        "earthquake": ["earthquake"],
        "flood": ["flood"],
        "hurricane": ["hurricane", "cyclone"],
        "tornado": ["tornado"],
        "wildfire": ["fire"],
        "explosion": ["explosion"],
        "landslide": ["landslide", "mudslide"],
        "shooting": ["shooting", "gun"],
        "crash": ["crash", "accident"],
    }

    def __init__(self, data_dir: str, output_path: str):
        self.data_dir = Path(data_dir)
        self.output_path = Path(output_path)

    def load_all(self) -> pd.DataFrame:
        tsv_files = list(self.data_dir.glob("*.tsv"))
        logger.info(f"Found {len(tsv_files)} TSV files in {self.data_dir}")

        frames = []
        for file in tsv_files:
            logger.info(f"Loading {file.name}")
            df = pd.read_csv(
            file,
            sep="\t",
            dtype=str,
            on_bad_lines="skip",   # <-- FIX
            engine="python"        # <-- FIX
            ).fillna("")

            frames.append(df)

        merged = pd.concat(frames, ignore_index=True)
        logger.info(f"Merged dataset size: {len(merged):,}")
        return merged

    def normalize_event(self, event: str) -> str:
        e = event.lower()
        for label, keywords in self.EVENT_MAP.items():
            if any(k in e for k in keywords):
                return label
        return "other"

    def normalize_humanitarian(self, label: str) -> str:
        return self.HUMANITARIAN_LABELS.get(label.lower(), "other")

    def build(self) -> pd.DataFrame:
        df = self.load_all()

        logger.info("Normalizing event types...")
        df["event_label"] = df["event"].apply(self.normalize_event)

        logger.info("Normalizing humanitarian labels...")
        df["human_label"] = df["class_label"].apply(self.normalize_humanitarian)

        logger.info("Deriving informativeness label...")
        df["info_label"] = df["human_label"].apply(
            lambda x: "no" if x == "not_humanitarian" else "yes"
        )

        df = df[df["text"].str.len() > 0]
        df = df.drop_duplicates(subset=["id", "text"])

        logger.info(f"Final cleaned size: {len(df):,}")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, index=False)
        logger.info(f"Saved cleaned dataset → {self.output_path}")

        return df
