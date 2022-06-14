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

from CocoDataset import CocoDataset

IMAGE_SIZE = (420, 420)
BATCH_SIZE = 5
LEARNING_RATE = 0.99
EPOCHS = 300


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

    train_datasets = CocoDataset(root=train_dir, annFile=f"{train_dir}/annotations_train.json", transform=transform)
    valid_datasets = CocoDataset(root=valid_dir, annFile=f"{valid_dir}/annotations_valid.json", transform=transform)

    return train_datasets, valid_datasets


# %%
def get_dataloader(train_datasets, valid_datasets, batch_size) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_datasets, batch_size=batch_size, shuffle=True)

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

        self.l1 = nn.Linear(107584, n_hidden)
        self.l2 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = torch.flatten(x, 1)
        x = self.l1(x)
        x = self.l2(x)

        return x

    def check_cnn_size(self, x: torch.FloatTensor = torch.FloatTensor(5, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])):
        cnn = self.layer3(self.layer2(self.layer1(x)))
        return torch.flatten(cnn).shape


# %%
def get_labels(targets: list, dataset: CocoDetection):
    cats = dataset.coco.cats


# %%
def train_model(
        cnn: CNN,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        criterion: nn.CrossEntropyLoss,
        optimizer: optim.Adam,
        device: torch.device
):
    history = np.zeros((0, 5))
    cnn.to(device)

    for epoch in range(EPOCHS):
        train_acc, train_loss = 0.0, 0.0
        valid_acc, valid_loss = 0.0, 0.0
        num_trained, num_tested = 0.0, 0.0

        cnn.train()

        for inputs, labels in tqdm(train_loader):
            num_trained += len(labels)

            labels = labels.type(torch.LongTensor)

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = cnn(inputs).to(device)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            predicted = torch.max(outputs, 1)[1]

            train_acc += (predicted == labels).sum().item()
            train_loss += loss.item()

        cnn.eval()

        for inputs, labels in valid_loader:
            num_tested += len(labels)

            labels = labels.type(torch.LongTensor)

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = cnn(inputs).to(device)
            loss = criterion(outputs, labels)

            predicted = torch.max(outputs, 1)[1]

            valid_acc += (predicted == labels).sum().item()
            valid_loss += loss.item()

        train_acc /= num_trained
        train_loss *= (BATCH_SIZE / num_trained)

        valid_acc /= num_tested
        valid_loss *= (BATCH_SIZE / num_tested)

        print(f"Epoch [{epoch + 1}/{EPOCHS}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, valid loss: {valid_loss:.5f}, valid acc: {valid_acc:.5f}")

        items = np.array([epoch, train_loss, train_acc, valid_loss, valid_acc])
        history = np.vstack((history, items))

    return history


# %%
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_datasets, valid_datasets = get_datasets(get_transform())
    train_loader, valid_loader = get_dataloader(train_datasets, valid_datasets, BATCH_SIZE)

    print(f"train data size: {len(train_datasets)}, test data size: {len(valid_datasets)}")

    cnn = CNN(512, 52)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    train_model(cnn, train_loader, valid_loader, criterion, optimizer, device)


# %%
if __name__ == "__main__":
    main()
