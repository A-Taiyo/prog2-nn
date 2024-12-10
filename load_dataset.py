import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms.v2 as transforms
import torch

ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
)

print(f'numbers of datasets:{len(ds_train)}')

image, target = ds_train[0]
print(type(image), target)

plt.imshow(image)
plt.title(target)
plt.show()

# PIL 画像を 'torch.Tensor' に変換する
image = transforms.functional.to_image(image)
image = transforms.functional.to_dtype(image, dtype=torch.float32, scale=True)
print(image.shape, image.dtype)
print(image.min(), image.max())