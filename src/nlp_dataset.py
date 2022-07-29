import torch
from torch.utils.data import Dataset


class NLPDataset(Dataset):
    # map from keys to data samples
    def __init__(self, encoded_dicts: list[dict], targets: list[int]):
        # encoded_dicts is a list that contains encoded_dict for input text.
        self.encoded_dicts = encoded_dicts
        self.targets = targets

    def __len__(self):
        return len(self.encoded_dicts)

    def __getitem__(self, idx: int):
        encoded_dict = self.encoded_dicts[idx]
        target = self.targets[idx]
        input_ids = torch.tensor(encoded_dict["input_ids"])
        attention_mask = torch.tensor(encoded_dict["attention_mask"])
        token_type_ids = torch.tensor(encoded_dict["token_type_ids"])
        target = torch.tensor(target).float()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "target": target,
        }
