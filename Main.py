from typing import Tuple

import numpy
import numpy as np
import matplotlib.pyplot as plt

import os
import time

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.nn.functional as func
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchinfo import summary
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection, ImageFolder
from torchvision.transforms import Compose
from PIL import Image
from tqdm import tqdm

from CocoDataset import CocoDataset
from FReLU import FReLU

CATS = ['10C', '10D', '10H', '10S', '11C', '11D', '11H', '11S', '12C', '12D', '12H', '12S', '13C', '13D', '13H', '13S', '1C', '1D', '1H', '1S', '2C',
        '2D', '2H', '2S', '3C', '3D', '3H', '3S', '4C', '4D', '4H', '4S', '5C', '5D', '5H', '5S', '6C', '6D', '6H', '6S', '7C', '7D', '7H', '7S',
        '8C', '8D', '8H', '8S', '9C', '9D', '9H', '9S']

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 100
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.005
EPOCHS = 125

N_INPUT = 15488
N_OUTPUT = 52


# %%
def get_transform(is_train) -> Compose:
    if is_train:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
            transforms.RandomErasing(0.5, scale=(0.02, 0.3), ratio=(0.3, 0.3)),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.2),
            transforms.RandomGrayscale(0.5)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])


# %%
def get_datasets() -> Tuple[ImageFolder, ImageFolder]:
    train_dir = "./dataset/train"
    valid_dir = "./dataset/valid"

    train_datasets = ImageFolder(root=train_dir, transform=get_transform(True))
    valid_datasets = ImageFolder(root=valid_dir, transform=get_transform(False))

    return train_datasets, valid_datasets


# %%
def get_dataloader(train_datasets, valid_datasets, batch_size) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(train_datasets, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True)
    valid_loader = DataLoader(valid_datasets, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True)

    return train_loader, valid_loader


# %%
def get_cats(coco_datasets: CocoDataset):
    return list(map(lambda x: x[1]["name"], coco_datasets.coco.cats.items()))


# %%
class CNN(nn.Module):
    def __init__(self, n_input, n_output, n_hidden1, n_hidden2):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            FReLU(32),
            nn.MaxPool2d((2, 2))
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            FReLU(64),
            nn.MaxPool2d((2, 2))
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5),
            FReLU(128),
            nn.MaxPool2d((2, 2))
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),
            FReLU(128),
            nn.MaxPool2d((2, 2))
        )

        self.dropout1 = nn.Dropout2d(p=0.3)
        self.dropout2 = nn.Dropout(p=0.5)

        self.l1 = nn.Linear(n_input, n_hidden1)
        self.l2 = nn.Linear(n_hidden1, n_hidden2)
        self.l3 = nn.Linear(n_hidden2, n_output)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.l1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)

        return x

    def check_cnn_size(self, x: torch.FloatTensor):
        cnn = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return torch.flatten(cnn).shape


# %%
def train_model(
        cnn: CNN,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        criterion: nn.CrossEntropyLoss,
        optimizer: optim.Optimizer,
        device: torch.device
):
    history = np.zeros((0, 5))
    scaler = amp.GradScaler()
    cnn.to(device)

    for epoch in range(EPOCHS):
        train_acc, train_loss = 0.0, 0.0
        valid_acc, valid_loss = 0.0, 0.0
        num_trained, num_tested = 0.0, 0.0

        cnn.train()

        for inputs, labels in tqdm(train_loader):
            num_trained += len(labels)

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with amp.autocast():
                outputs = cnn(inputs).to(device)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            predicted = torch.max(outputs, 1)[1]

            train_acc += (predicted == labels).sum().item()
            train_loss += loss.item()

        cnn.eval()

        for inputs, labels in valid_loader:
            num_tested += len(labels)

            inputs = inputs.to(device)
            labels = labels.to(device)

            with amp.autocast():
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

        if (epoch + 1) % 10 == 0:
            show_loss_carve(history)
            show_accuracy_graph(history)

    return history


# %%
def show_loss_carve(history: numpy.ndarray):
    plt.rcParams["figure.figsize"] = (8, 6)
    plt.plot(history[:, 0], history[:, 1], "b", label="train")
    plt.plot(history[:, 0], history[:, 3], "k", label="valid")
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.title("loss carve")
    plt.legend()

    plt.savefig(f"./result/loss-{int(time.time())}.jpg")
    plt.show()


# %%
def show_accuracy_graph(history: numpy.ndarray):
    plt.rcParams["figure.figsize"] = (8, 6)
    plt.plot(history[:, 0], history[:, 2], "b", label="train")
    plt.plot(history[:, 0], history[:, 4], "k", label="valid")
    plt.xlabel("iter")
    plt.ylabel("acc")
    plt.title("accuracy")
    plt.legend()

    plt.savefig(f"./result/accuracy-{int(time.time())}.jpg")
    plt.show()


# %%
def show_result(cnn: CNN, valid_loader: DataLoader, device):
    for images, labels in valid_loader:
        break

    cnn.to(device)
    cnn.eval()

    test_labels = labels.to(device)
    test_images = images.to(device)

    outputs = cnn(test_images).to(device)
    predicts = torch.max(outputs, 1)[1]

    plt.figure(figsize=(21, 15))

    for index in range(50):
        ax = plt.subplot(5, 10, index + 1)

        answer_label = CATS[test_labels[index].item()]
        predicted_label = CATS[predicts[index].item()]

        color = "k" if answer_label == predicted_label else "b"
        ax.set_title(f"{answer_label}:{predicted_label}", c=color, fontsize=20)

        image_np = images[index].numpy().copy()
        image = np.transpose(image_np, (1, 2, 0))
        image = (image + 1) / 2

        plt.imshow(image)
        ax.set_axis_off()

    plt.savefig(f"./result/result-{int(time.time())}.jpg")
    plt.show()


# %%
def get_predict(path: str, cnn: CNN, device):
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
    ])

    image = Image.open(path)
    image = image.convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    cnn.to(device)
    cnn.eval()

    output = cnn(image)
    output = output.tolist()[0]
    output = list(enumerate(output))
    output = sorted(output, key=lambda x: x[1], reverse=True)

    return output


# %%
def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_datasets, valid_datasets = get_datasets()
    train_loader, valid_loader = get_dataloader(train_datasets, valid_datasets, BATCH_SIZE)

    cnn = CNN(N_INPUT, N_OUTPUT, 1024, 512)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    history = train_model(cnn, train_loader, valid_loader, criterion, optimizer, device)

    save_path = f"./weights/weight-{int(time.time())}.pth"
    torch.save(cnn.state_dict(), save_path)

    show_loss_carve(history)
    show_accuracy_graph(history)
    show_result(cnn, valid_loader, device)


# %%
def predict():
    # print("Enter the path of the PTH file > ", end="")
    # pth_path = input().strip()
    pth_path = "E:\IntelliJ\Projects\KCS\TrumpRecognition-CNN\weights\weight-1655617999.pth"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn = CNN(N_INPUT, N_OUTPUT, 1024, 512)
    cnn.load_state_dict(torch.load(pth_path))

    while True:
        print("Enter the path of the image to predict > ", end="")
        image_path = input().strip()

        if image_path.lower() == "exit":
            print("Ended the process.")
            break

        result = get_predict(image_path, cnn, device)
        print(f"Result > {list(map(lambda x: (CATS[x[0]], x[1]), result[:5]))}")


# %%
def info():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn = CNN(N_INPUT, N_OUTPUT, 1024, 512).to(device)

    print(cnn.check_cnn_size(torch.FloatTensor(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]).to(device)))
    print(summary(model=cnn, input_size=(5, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])))


# %%
def main():
    print("Choose mode predict mode [P], train mode [T] or info mode [I]: > ", end="")
    mode = input().strip()

    if mode.upper() == "P":
        predict()
    elif mode.upper() == "T":
        train()
    elif mode.upper() == "I":
        info()
    else:
        print("Invalid input. ")


# %%
if __name__ == "__main__":
    main()
