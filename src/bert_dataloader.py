from transformers import BertJapaneseTokenizer
from nlp_dataset import NLPDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


class BERTDataloader:
    def __init__(self, cfg):
        self.model_type = cfg["MODEL_PARAM"]["type"]
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(self.model_type)
        self.add_special_tokens = cfg["TOKENIZER_PARAM"]["add_special_tokens"]
        self.max_length = cfg["TOKENIZER_PARAM"]["max_length"]
        self.pad_to_max_length = cfg["TOKENIZER_PARAM"]["pad_to_max_length"]
        self.batch_size = cfg["TRAIN_PARAM"]["batch_size"]

    def _encode_sents(self, sents: list[str]) -> list[dict]:
        encoded_dicts = []
        for sent in tqdm(sents):
            encoded_dict = self.tokenizer(
                sent,
                add_special_tokens=self.add_special_tokens,
                max_length=self.max_length,
                pad_to_max_length=self.pad_to_max_length,
            )
            encoded_dicts.append(encoded_dict)
        return encoded_dicts

    def __call__(self, sents: list[str], targets: list[int]):
        encoded_dicts = self._encode_sents(sents)
        dataset = NLPDataset(encoded_dicts, targets)
        return DataLoader(dataset, batch_size=self.batch_size)
