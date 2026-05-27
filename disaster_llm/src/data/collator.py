#G:\Text_Classification\disaster_llm\src\data\collator.py
import torch
from typing import Dict, List
from .encode_utils import IGNORE_IDX


class DataCollatorForSFT:
    """
    Pads input_ids and labels for SFT training.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_id
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_IDX
        )

        attention_mask = input_ids.ne(self.pad_id)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }
