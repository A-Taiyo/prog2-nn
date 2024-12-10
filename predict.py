import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

import models


# モデルをインスタンス化する
model = models.MyModel()
print(model)

# データセットのロード
ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True)
    ])
)

# image は PIL ではなく Tensor に変換済み
image, target = ds_train[0]
# (1, H, W) から (1, 1, H, W) に次元を上げる
image = image.unsqueeze(dim=0)

# モデルに入れて結果(logits)を出す
model.eval()
with torch.no_grad():
    logits = model(image)

print(logits)

# ロジットをグラフにする
plt.bar(range(len(logits[0])), logits[0])
plt.show()

# クラス確率をグラフにする
probs = logits.softmax(dim=1)
plt.bar(range(len(probs[0])), probs[0])
plt.ylim(0, 1)
plt.show()

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image[0, 0], cmap='gray_r')
plt.title(f'class: {target} ({datasets.FashionMNIST.classes[target]})')

plt.subplot(1, 2, 2)
plt.bar(range(len(probs[0])), probs[0])
plt.ylim(0, 1)

plt.show()