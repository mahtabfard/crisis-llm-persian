#G:\Text_Classification\disaster_llm\src\data\encode_utils.py
import json
from typing import Dict, List
from dataclasses import dataclass

IGNORE_IDX = -100


class JsonHandler:
    """Utility class for loading JSONL or JSON files."""
    @staticmethod
    def load_json(path: str) -> List[Dict]:
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f]
        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            raise ValueError("Unsupported file format.")


@dataclass
class DialogueEncoder:
    """
    Encodes instruction-response examples for supervised fine-tuning.
    """
    tokenizer: any
    system_prompt: str = "You are a classifier for crisis-related tweets."

    def encode_dialog(self, dialog: Dict, max_length: int) -> Dict:
        """
        Encode a single example consisting of:
        - instruction
        - response (JSON-style dict)
        """

        instruction = dialog["instruction"]
        response = json.dumps(dialog["response"], ensure_ascii=False)

        text = (
            f"<s>[SYSTEM]\n{self.system_prompt}</s>\n"
            f"[USER]\n{instruction}\n"
            f"[ASSISTANT]\n{response}</s>"
        )

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors=None
        )

        # labels = input_ids except user/system parts masked with IGNORE_IDX
        labels = encoded["input_ids"].copy()
        user_end = text.index("[ASSISTANT]")  # mask everything before assistant
        cutoff = len(self.tokenizer(text[:user_end])["input_ids"])

        for i in range(cutoff):
            labels[i] = IGNORE_IDX

        return {
            "input_ids": encoded["input_ids"],
            "labels": labels
        }
