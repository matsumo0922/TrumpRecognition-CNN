from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import Compose
from tqdm import tqdm

IMAGE_SIZE = (420, 420)


# %%
def get_transform() -> Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
        transforms.Resize(IMAGE_SIZE)
    ])


# %%
def get_datasets(transform) -> Tuple[CocoDetection, CocoDetection]:
    train_dir = "./dataset/train"
    valid_dir = "./dataset/valid"

    train_datasets = datasets.CocoDetection(root=train_dir, annFile=f"{train_dir}/annotations_train.json", transform=transform)
    valid_datasets = datasets.CocoDetection(root=valid_dir, annFile=f"{valid_dir}/annotations_valid.json", transform=transform)

    return train_datasets, valid_datasets


# %%
def get_dataloader(train_datasets, valid_datasets, batch_size) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_datasets, batch_size=batch_size, shuffle=True)

    print("train batch size: ", len(train_loader))
    print("test batch size: ", len(valid_loader))

    return train_loader, valid_loader


# %%
class CNN(nn.Module):
    def __init__(self, n_hidden, n_output):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2))
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2))
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2))
        )

        self.l1 = nn.Linear(61 * 41 * 41, n_hidden)
        self.l2 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = torch.flatten(x, 1)
        x = self.l1(x)
        x = self.l2(x)

        return x

    def check_cnn_size(self):
        test_tensor = torch.FloatTensor(5, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])
        cnn = self.layer3(self.layer2(self.layer1(test_tensor)))
        return cnn.shape


# %%
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_datasets, valid_datasets = get_datasets(get_transform())
    train_loader, valid_loader = get_dataloader(train_datasets, valid_datasets, 5)

    cnn = CNN(512, 52)


# %%
if __name__ == "__main__":
    main()
