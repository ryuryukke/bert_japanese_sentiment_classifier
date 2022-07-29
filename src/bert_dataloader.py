from transformers import BertJapaneseTokenizer
from nlp_dataset import NLPDataset
from torch.utils.data import DataLoader


class BERTDataloader:
    def __init__(self, model_type, add_special_tokens=True, max_length=64, pad_to_max_length=True):
        self.model_type = model_type
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_type)
        self.add_special_tokens = add_special_tokens
        self.max_length = max_length
        self.pad_to_max_length = pad_to_max_length

    def _encode_sents(self, sents: list[str]) -> list[dict]:
        encoded_dicts = []
        for sent in sents:
            encoded_dict = self.tokenizer(
                sent,
                add_special_tokens=self.add_special_tokens,
                max_length=self.max_length,
                pad_to_max_length=self.pad_to_max_length,
            )
            encoded_dicts.append(encoded_dict)
        return encoded_dicts

    def __call__(self, sents: list[str], targets: list[int], batch_size=64):
        encoded_dicts = self._encode_sents(sents)
        dataset = NLPDataset(encoded_dicts, targets)
        return DataLoader(dataset, batch_size=batch_size)
