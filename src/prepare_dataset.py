import pickle
import yaml
from sklearn.model_selection import train_test_split
from bert_dataloader import BERTDataloader
from torch.utils.data import DataLoader

"""
Prepare the dataset for traning BERT model
"""


class Preprocessor:
    def __init__(
        self,
        config_path: str,
    ):
        self.cfg_path = config_path
        self.cfg = self._load_config()

    def _load_config(self):
        with open(self.cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg

    def _load_dataset(self):
        with open(self.cfg["DATA_PARAM"]["path"], "rb") as f:
            dataset = pickle.load(f)
        return dataset

    def _split_dataset(self, dataset: list[list[str, int]]):
        train_param = self.cfg["TRAIN_PARAM"]
        train_size, valid_size, test_size = (
            train_param["train_size"],
            train_param["valid_size"],
            train_param["test_size"],
        )
        assert train_size + valid_size + test_size == 1, "train, valid, testの割合が不適切です"
        assert len(dataset[0]) == 2, "[['文1', 'ラベル'], ['文2', 'ラベル'], ...]というデータを用意してください"
        idx_lst = list(range(len(dataset)))
        train_idxes, test_idxes, _, _ = train_test_split(
            idx_lst, idx_lst, test_size=test_size, random_state=train_param["random_seed"]
        )
        train_idxes, valid_idxes, _, _ = train_test_split(
            train_idxes, train_idxes, test_size=valid_size, random_state=train_param["random_seed"]
        )
        sents = {
            "train": [dataset[train_idx][0] for train_idx in train_idxes],
            "valid": [dataset[valid_idx][0] for valid_idx in valid_idxes],
            "test": [dataset[test_idx][0] for test_idx in test_idxes],
        }
        targets = {
            "train": [dataset[train_idx][1] for train_idx in train_idxes],
            "valid": [dataset[valid_idx][1] for valid_idx in valid_idxes],
            "test": [dataset[test_idx][1] for test_idx in test_idxes],
        }
        return sents, targets

    def _create_dataloader(self, sents: list[str], targets: list[int]):
        assert len(sents) == len(targets), "文とラベルの数が合わず、データセットに不備があります"
        bert_dataloader = BERTDataloader(self.cfg)
        return bert_dataloader(sents, targets)

    def __call__(self):
        print("Start loading dataset...")
        dataset = self._load_dataset()
        print("End loading dataset...")
        print("Start splitting dataset...")
        sents, targets = self._split_dataset(dataset)
        print("End splitting dataset...")
        print("Start creating dataloader...")
        train_dataloader = self._create_dataloader(sents["train"], targets["train"])
        valid_dataloader = self._create_dataloader(sents["valid"], targets["valid"])
        test_dataloader = self._create_dataloader(sents["test"], targets["test"])
        print("End creating dataloader...")
        return train_dataloader, valid_dataloader, test_dataloader


# test
preprocessor = Preprocessor("../config/config.yaml")
train_dataloader, valid_dataloader, test_dataloader = preprocessor()