import math
import torch
import torch.nn as nn
from torch.optim import Adam
import yaml
from tqdm import tqdm
from bert_binary_classifier import BertBinaryClassifier
import datetime
import subprocess
import numpy as np


class Trainer:
    def __init__(self, config_path: str):
        self.cfg_path = config_path
        self.cfg = self._load_config()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model = BertBinaryClassifier(self.cfg)
        self.model.to(self.device)
        self.epoch = self.cfg["TRAIN_PARAM"]["n_epoch"]
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.cfg["TRAIN_PARAM"]["learning_rate"])

    def _load_config(self):
        with open(self.cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg

    def _train_loop(self, train_dataloader):
        losses = []
        self.model.train()
        self.optimizer.zero_grad()
        for _, batch_dict in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            input_ids = batch_dict["input_ids"].to(self.device)
            attention_mask = batch_dict["attention_mask"].to(self.device)
            targets = batch_dict["target"].to(self.device)

            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, targets)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            losses.append(loss.item())
        return {"loss": np.array(losses).mean()}

    def _test_loop(self, test_dataloader):
        losses, predicts, targets = [], [], []
        self.model.eval()

        for _, batch_dict in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            input_ids = batch_dict["input_ids"].to(self.device)
            attention_mask = batch_dict["attention_mask"].to(self.device)
            targets = batch_dict["target"].to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)

            loss = self.criterion(logits, targets)
            losses.append(loss.item())
            targets += targets.cpu().tolist()
            predicts += logits.sigmoid().cpu().tolist()

        accuracy = self._cal_accuracy(targets, predicts)

        return {"loss": np.array(losses).mean(), "accuracy": accuracy}

    def _cal_accuracy(self, targets, predicts):
        assert len(targets) == len(predicts), "教師ラベルと予測ラベルの数が不一致です。"
        match_cnt = 0
        for target, predict in zip(targets, predicts):
            preds = np.where(np.array(predict) < 0.5, 0.0, 1.0)
            if target == preds.tolist():
                match_cnt += 1
        return match_cnt / len(targets)

    def _save_model(self, epoch):
        dt_now = datetime.datetime.now()
        dir_name = f"{dt_now.year}_{dt_now.month}_{dt_now.day}_{dt_now.hour}_{dt_now.minute}"
        dir_path = f"../model/{dir_name}"
        subprocess.call(["mkdir", "-p", dir_path])
        torch.save(self.model.state_dict(), f"{dir_path}/epoch_{epoch + 1}.pth")
        return

    def __call__(self, train_dataloader, valid_dataloader, test_dataloader):
        self.model.train()
        self.optimizer.zero_grad()
        for epoch in range(self.epoch):
            train_res = self._train_loop(train_dataloader)
            print(f"epoch {epoch + 1}: train_loss = {train_res['loss']}")
            self._save_model(epoch)
            valid_res = self._test_loop(valid_dataloader)
            print(f"epoch {epoch + 1}: valid_loss = {valid_res['loss']}, accuracy = {valid_res['accuracy']}")
            test_res = self._test_loop(test_dataloader)
            print(f"epoch {epoch + 1}: valid_loss = {test_res['loss']}, accuracy = {test_res['accuracy']}")
        return
