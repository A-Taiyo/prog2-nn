import torch
from torch import nn


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits


def test_accuracy(model, dataloader, device='cpu'):
    # 全てのミニバッチに対して推論して、正解率を計算する
    n_corrects = 0  # 正解した個数

    model = model.to(device)
    model.eval()
    for image_batch, label_batch in dataloader:
        image_batch = image_batch.to(device)
        label_batch = label_batch.to(device)
        # モデルに入れて結果(logits)を出す
        with torch.no_grad():
            logits_batch = model(image_batch)

        predict_batch = logits_batch.argmax(dim=1)
        n_corrects += (label_batch == predict_batch).sum().item()

    # 精度(正解率)を計算する
    accuracy = n_corrects / len(dataloader.dataset)

    return accuracy


def train(model, dataloader, loss_fn, optimizer, device='cpu'):
    """1 epoch の学習に使う"""
    model = model.to(device)
    model.train()
    for image_batch, label_batch in dataloader:
        image_batch = image_batch.to(device)
        label_batch = label_batch.to(device)
        # モデルにバッチを入れて計算
        logits_batch = model(image_batch)

        # 損失(誤差)を計算する
        loss = loss_fn(logits_batch, label_batch)

        # 最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 最後のバッチのロス
    return loss.item()


def test(model, dataloader, loss_fn, device='cpu'):
    loss_total = 0.0

    model = model.to(device)
    model.eval()
    for image_batch, label_batch in dataloader:
        image_batch = image_batch.to(device)
        label_batch = label_batch.to(device)
        with torch.no_grad():
            logits_batch = model(image_batch)

        loss = loss_fn(logits_batch, label_batch)
        loss_total += loss.item()

    
    return loss_total / len(dataloader)