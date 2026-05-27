#G:\Text_Classification\disaster_llm\src\data\sft_dataset.py
import torch
from torch.utils.data import Dataset
from typing import Dict, List
from .encode_utils import JsonHandler, DialogueEncoder, IGNORE_IDX


class SFTDataset(Dataset):
    """
    Supervised fine-tuning dataset for instruction→response pairs.
    """
    def __init__(self, jsonl_path: str, encoder: DialogueEncoder, max_seq_length: int):
        self.samples = JsonHandler.load_json(jsonl_path)
        self.encoder = encoder
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dialog = self.samples[idx]
        encoded = self.encoder.encode_dialog(dialog, self.max_seq_length)
        encoded["input_ids"] = torch.tensor(encoded["input_ids"], dtype=torch.long)
        encoded["labels"] = torch.tensor(encoded["labels"], dtype=torch.long)
        return encoded
