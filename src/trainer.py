import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

import yaml
from tqdm import tqdm
import datetime
import subprocess

from bert_binary_classifier import BertBinaryClassifier


class Trainer:
    def __init__(self, config_path: str):
        self.cfg_path = config_path
        self.cfg = self._load_config()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = BertBinaryClassifier(self.cfg)
        self.model.to(self.device)
        self.epoch = self.cfg["TRAIN_PARAM"]["n_epoch"]
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.cfg["TRAIN_PARAM"]["learning_rate"])

    def _load_config(self):
        with open(self.cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg

    def _load_pretrained_model(self, pretrained_model_path):
        model = BertBinaryClassifier(self.cfg)
        model.load_state_dict(torch.load(pretrained_model_path))
        model.eval()
        return model

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

    def _test_loop(self, test_dataloader, load_pretrained_model=False, pretrained_model_path=None):
        losses, outputs, all_targets = [], [], []
        self.model.eval()

        for _, batch_dict in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            input_ids = batch_dict["input_ids"].to(self.device)
            attention_mask = batch_dict["attention_mask"].to(self.device)
            targets = batch_dict["target"].to(self.device)

            with torch.no_grad():
                if load_pretrained_model:
                    pretrained_model = self._load_pretrained_model(pretrained_model_path)
                    logits = pretrained_model(input_ids, attention_mask)
                else:
                    logits = self.model(input_ids, attention_mask)

            loss = self.criterion(logits, targets)
            losses.append(loss.item())

            all_targets += targets.cpu().tolist()
            outputs += nn.Softmax(dim=1)(logits).cpu().tolist()

        accuracy = self._cal_accuracy(all_targets, outputs)

        return {"loss": np.array(losses).mean(), "accuracy": accuracy}

    def _cal_accuracy(self, targets, outputs):
        assert len(targets) == len(outputs), "教師ラベルと予測ラベルの数が不一致です。"
        match_cnt = 0
        for target, output in zip(targets, outputs):
            pred = np.where(np.array(output) < 0.5, 0.0, 1.0)
            if target == pred.tolist():
                match_cnt += 1
        return match_cnt / len(targets)

    def _save_model(self, epoch):
        dt_now = datetime.datetime.now()
        dir_name = f"{dt_now.year}_{dt_now.month}_{dt_now.day}"
        dir_path = f"../model/{dir_name}"
        subprocess.call(["mkdir", "-p", "../model"])
        subprocess.call(["mkdir", "-p", dir_path])
        torch.save(self.model.state_dict(), f"{dir_path}/{dt_now.hour}_{dt_now.minute}_epoch_{epoch + 1}.pth")
        return

    def __call__(self, train_dataloader, valid_dataloader, test_dataloader):
        self.model.train()
        self.optimizer.zero_grad()
        for epoch in range(self.epoch):
            train_res = self._train_loop(train_dataloader)
            print(f"epoch {epoch + 1}: train_loss = {train_res['loss']}")
            self._save_model(epoch)
            valid_res = self._test_loop(valid_dataloader)
            print(f"epoch {epoch + 1}: valid_loss = {valid_res['loss']}, valid_accuracy = {valid_res['accuracy']}")
            test_res = self._test_loop(test_dataloader)
            print(f"epoch {epoch + 1}: test_loss = {test_res['loss']}, test_accuracy = {test_res['accuracy']}")
        return
