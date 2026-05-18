from pathlib import Path

dataset_code = """
import json
import torch
from torch.utils.data import Dataset

class CoralDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_input_length, max_target_length):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source_text = item['input']
        target_text = item['target']
        view_type = item['view_type']

        inputs = self.tokenizer(
            source_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_input_length,
            return_tensors="pt",
        )
        targets = self.tokenizer(
            target_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": targets["input_ids"].squeeze(0),
            "view_type": torch.tensor(0 if view_type == "global" else 1, dtype=torch.long)
        }
"""

dataset_path = Path("/mnt/data/dataset.py")
dataset_path.write_text(dataset_code.strip())

dataset_path.name