#G:\Text_Classification\disaster_llm\data\instruction_builder.py
import json
from pathlib import Path
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

class InstructionBuilder:
    """
    Converts cleaned CrisisBench rows into instruction-tuning samples.
    Format matches paper: instruction -> response
    """

    def __init__(self, input_csv: str, output_jsonl: str):
        self.input_csv = Path(input_csv)
        self.output_jsonl = Path(output_jsonl)

    def build(self):
        df = pd.read_csv(self.input_csv)
        records = []

        for _, row in df.iterrows():
            instruction = (
                "Given the following tweet, classify:\n"
                "1. The disaster event type.\n"
                "2. Whether the tweet is informative.\n"
                "3. The humanitarian aid category.\n\n"
                f"Tweet: {row['text']}"
            )

            response = {
                "event": row["event_label"],
                "informative": row["info_label"],
                "humanitarian": row["human_label"]
            }

            records.append({
                "instruction": instruction,
                "response": response
            })

        self.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_jsonl, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        logger.info(f"Instruction dataset saved → {self.output_jsonl}")
