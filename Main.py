import datetime
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

from Model import ResNet
from Model import DiffBlock
from PrintLog import PrintLog
from FReLU import FReLU

CATS = ['10C', '10D', '10H', '10S', '11C', '11D', '11H', '11S', '12C', '12D', '12H', '12S', '13C', '13D', '13H', '13S', '1C', '1D', '1H', '1S', '2C',
        '2D', '2H', '2S', '3C', '3D', '3H', '3S', '4C', '4D', '4H', '4S', '5C', '5D', '5H', '5S', '6C', '6D', '6H', '6S', '7C', '7D', '7H', '7S',
        '8C', '8D', '8H', '8S', '9C', '9D', '9H', '9S']

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 50
LEARNING_RATE = 0.01
MOMENTUM = 0.9
EPOCHS = 75
N_OUTPUT = 52


def get_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")

    n_device = torch.cuda.device_count()

    if n_device == 1:
        return torch.device("cuda:0")

    devices = list(map(lambda i: f"cuda:{i[0]}", enumerate([None] * n_device)))
    device_names = list(map(lambda x: torch.cuda.get_device_name(x), devices))

    print(f"Find any GPU -> {device_names}")
    print("Please enter the index of the GPU to use (1~) > ", end="")
    index = int(input().strip())

    return torch.device(devices[index - 1])


def get_transform(is_train) -> Compose:
    if is_train:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
            transforms.RandomErasing(0.5, scale=(0.02, 0.3), ratio=(0.3, 0.3)),
            transforms.RandomHorizontalFlip(0.4),
            transforms.RandomVerticalFlip(0.4),
            transforms.RandomApply([transforms.GaussianBlur(3)], 0.2),
            # transforms.RandomApply([transforms.Grayscale()], 0.2),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.6, contrast=0.6)], 0.2),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])


def get_datasets():
    train_dir = "./dataset/train"
    valid_dir = "./dataset/valid"

    train_datasets = ImageFolder(root=train_dir, transform=get_transform(True))
    valid_datasets = ImageFolder(root=valid_dir, transform=get_transform(False))

    # train_datasets = datasets.CIFAR10(train_dir, train=True, download=True, transform=get_transform(True))
    # valid_datasets = datasets.CIFAR10(valid_dir, train=False, download=True, transform=get_transform(False))

    return train_datasets, valid_datasets


def get_dataloader(train_datasets, valid_datasets, batch_size) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(train_datasets, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True)
    valid_loader = DataLoader(valid_datasets, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True)

    return train_loader, valid_loader


def train_model(
        net: ResNet,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        criterion: nn.CrossEntropyLoss,
        optimizer: optim.Optimizer,
        device: torch.device,
        outputs_path: str,
        logger: PrintLog
):
    start_time = time.perf_counter()
    history = np.zeros((0, 5))
    scaler = amp.GradScaler()
    net.to(device)

    for epoch in range(EPOCHS):

        train_acc, train_loss = 0.0, 0.0
        valid_acc, valid_loss = 0.0, 0.0
        num_trained, num_tested = 0.0, 0.0

        net.train()

        for inputs, labels in tqdm(train_loader):
            num_trained += len(labels)

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with amp.autocast():
                outputs = net(inputs).to(device)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            predicted = torch.max(outputs, 1)[1]

            train_loss += loss.item()
            train_acc += (predicted == labels).sum().item()

        net.eval()

        for valid_inputs, valid_labels in valid_loader:
            num_tested += len(valid_labels)

            valid_inputs = valid_inputs.to(device)
            valid_labels = valid_labels.to(device)

            with amp.autocast():
                valid_outputs = net(valid_inputs).to(device)
                loss = criterion(valid_outputs, valid_labels)

            valid_predicted = torch.max(valid_outputs, 1)[1]

            valid_loss += loss.item()
            valid_acc += (valid_predicted == valid_labels).sum().item()

        train_acc /= num_trained
        valid_acc /= num_tested

        train_loss *= (BATCH_SIZE / num_trained)
        valid_loss *= (BATCH_SIZE / num_tested)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        eta = get_time_from_sec((elapsed_time / (epoch + 1)) * (EPOCHS - (epoch + 1)))

        message = f"Epoch [{epoch + 1}/{EPOCHS}] "
        message += f"loss: {train_loss:.5f}, acc: {train_acc:.5f}, valid loss: {valid_loss:.5f}, valid acc: {valid_acc:.5f} "
        message += f"[ETA: {str(eta[0]).zfill(2)}:{str(eta[1]).zfill(2)}:{str(eta[2]).zfill(2)}]"

        logger.println(message)

        items = np.array([epoch, train_loss, train_acc, valid_loss, valid_acc])
        history = np.vstack((history, items))

        if (epoch + 1) % 5 == 0 and (epoch + 1) != EPOCHS:
            save_weight(outputs_path, epoch, net)
            show_loss_carve(outputs_path, epoch, history)
            show_accuracy_graph(outputs_path, epoch, history)

    return history


def show_loss_carve(parent_path, epoch, history: numpy.ndarray):
    plt.rcParams["figure.figsize"] = (8, 6)
    plt.plot(history[:, 0], history[:, 1], "b", label="train")
    plt.plot(history[:, 0], history[:, 3], "k", label="valid")
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.title("loss carve")
    plt.legend()

    plt.savefig(f"{parent_path}/loss-epoch-{epoch}.jpg")
    plt.show()


def show_accuracy_graph(parent_path, epoch, history: numpy.ndarray):
    plt.rcParams["figure.figsize"] = (8, 6)
    plt.plot(history[:, 0], history[:, 2], "b", label="train")
    plt.plot(history[:, 0], history[:, 4], "k", label="valid")
    plt.xlabel("iter")
    plt.ylabel("acc")
    plt.title("accuracy")
    plt.legend()

    plt.savefig(f"{parent_path}/accuracy-epoch-{epoch}.jpg")
    plt.show()


def show_result(net: ResNet, valid_loader: DataLoader, device, parent_path):
    for images, labels in valid_loader:
        break

    net.to(device)
    net.eval()

    test_labels = labels.to(device)
    test_images = images.to(device)

    outputs = net(test_images).to(device)
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

    plt.savefig(f"{parent_path}/result.jpg")
    plt.show()


def save_weight(parent_path, epoch, model):
    save_dir = f"{parent_path}/weights"
    save_path = f"{save_dir}/weight-epoch-{epoch}.pth"

    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)


def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def get_time_from_sec(sec) -> Tuple[int, int, int]:
    timedelta = datetime.timedelta(seconds=sec)
    m, s = divmod(timedelta.seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s


def get_predict(path: str, net: ResNet, device):
    transform = get_transform(False)

    image = Image.open(path)
    image = image.convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    net.to(device)
    net.eval()

    output = net(image)
    output = output.tolist()[0]
    output = list(enumerate(output))
    output = sorted(output, key=lambda x: x[1], reverse=True)

    return output


def train():
    time_id = get_time()
    result_path = f"./results/{time_id}"
    os.makedirs(result_path, exist_ok=True)

    log = PrintLog(result_path + "/log.txt")
    device = get_device()

    train_datasets, valid_datasets = get_datasets()
    train_loader, valid_loader = get_dataloader(train_datasets, valid_datasets, BATCH_SIZE)

    log.println(f"train data: {len(train_datasets)}, valid data: {len(valid_datasets)}")

    net = ResNet(DiffBlock, N_OUTPUT)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    history = train_model(net, train_loader, valid_loader, criterion, optimizer, device, result_path, log)

    save_weight(result_path, "finish", net)
    show_loss_carve(result_path, "finish", history)
    show_accuracy_graph(result_path, "finish", history)
    show_result(net, valid_loader, device, result_path)


def predict():
    # print("Enter the path of the PTH file > ", end="")
    # pth_path = input().strip()
    pth_path = "E:\IntelliJ\Projects\KCS\TrumpRecognition-CNN\weights\weight-2022-06-22-12-42-33.pth"

    device = get_device()
    net = ResNet(DiffBlock, N_OUTPUT)
    net.load_state_dict(torch.load(pth_path))

    while True:
        print("Enter the path of the image to predict > ", end="")
        image_path = input().strip()

        if image_path.lower() == "exit":
            print("Ended the process.")
            break

        result = get_predict(image_path, net, device)
        rate_sum = sum(list(map(lambda x: x[1], result)))
        take_list = list(map(lambda x: (CATS[x[0]], x[1]), result[:7]))

        for trump, rate in take_list:
            print(f"{trump}, {(rate / rate_sum) * 100}%, {rate}")


def info():
    device = get_device()
    net = ResNet(DiffBlock, N_OUTPUT).to(device)
    tensor = torch.FloatTensor(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]).to(device)

    print(net(tensor).to(device).shape)
    print(summary(model=net, input_size=(BATCH_SIZE, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])))


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


if __name__ == "__main__":
    main()
