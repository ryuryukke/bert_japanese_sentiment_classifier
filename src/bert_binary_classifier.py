import torch.nn as nn
from transformers import BertModel, BertConfig


class BertBinaryClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bert_model = BertModel.from_pretrained(cfg["MODEL_PARAM"]["type"])
        bert_conf = BertConfig(cfg["MODEL_PARAM"]["type"])
        self.fc1 = nn.Linear(bert_conf.hidden_size, bert_conf.hidden_size)
        self.fc2 = nn.Linear(bert_conf.hidden_size, cfg["MODEL_PATH"]["n_label"])
        self.relu = nn.ReLU()

    def forward(self, ids, mask):
        _, h = self.bert_model(ids, attention_mask=mask)
        h = self.relu(h)
        h = self.fc1(h)
        h = self.relu(h)
        logit = self.fc2(h)
        return logit
