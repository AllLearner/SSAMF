from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from MobileNetv3 import mobilenetv3_large
from dataloader import MyDataset
from torch.utils import data
from torch import optim
import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler
import torch
import os

def train(net, device,train_path,test_path):
    train_dataset = MyDataset(train_path)
    dataloaders_train = DataLoader(train_dataset,
                                   batch_size=32,
                                   shuffle=True,
                                   num_workers=2
                                   )
    test_dataset = MyDataset(test_path)
    dataloaders_test = DataLoader(test_dataset,
                                   batch_size=16,
                                   shuffle=True,
                                   num_workers=2
                                   )
    #weight_CE = torch.FloatTensor([1, 7.32, 17.75])
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=1e-5, momentum=0.9)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    train_correct = 0
    train_total = 0
    running_loss = 0

    net.train()
    train_bar = tqdm(dataloaders_train, file=sys.stdout)
    for step, (x, y) in enumerate(train_bar):
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)

        optimizer.zero_grad()

        y_pred = net(x)
        y = y.long()
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            train_correct += (y_pred == y).sum().item()
            train_total += y.size(0)
            running_loss += loss.item()

    exp_lr_scheduler.step()
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = train_correct / train_total

    test_correct = 0
    test_running_loss = 0
    test_total = 0
    net.eval()
    with torch.no_grad():
        test_bar = tqdm(dataloaders_test, file=sys.stdout)
        for step, (x, y) in enumerate(test_bar):
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            y_pred = net(x)
            y = y.long()
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_running_loss += loss.item()
            test_total += y.size(0)

    test_epoch_acc = test_correct / test_total
    test_epoch_loss = test_running_loss / len(test_dataset)
    return epoch_loss, epoch_acc, test_epoch_acc, test_epoch_loss

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

if __name__ == "__main__":
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print("using {} device.".format(device))
    train_path = './train/'
    test_path = './val/'

    model = mobilenetv3_large()
    model.load_state_dict(torch.load('mobilenetv3-large-1cd25616.pth', map_location='cpu'))
    for param in model.parameters():
        param.requires_grad = True
    model.classifier = nn.Sequential(
            nn.Linear(960, 1280),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(1280, 2),
        )
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    model.to(device)

    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []
    best_loss = float('inf')
    best_acc = 0
    Epochs = 100
    for epoch in range(Epochs):
        epoch_loss, epoch_acc, test_epoch_acc, test_epoch_loss = train(model, device, train_path, test_path)

        train_acc.append(epoch_acc)
        train_loss.append(epoch_loss)
        test_loss.append(test_epoch_loss)
        test_acc.append(test_epoch_acc)
        print("epoch:", epoch,
              "train_loss:", round(epoch_loss, 5),
              "train_acc:", round(epoch_acc, 5),
              "test_loss:", round(test_epoch_loss, 5),
              "test_acc:", round(test_epoch_acc, 5),
              )
        if test_epoch_acc >= best_acc:
            best_acc = test_epoch_acc
            torch.save(model.state_dict(), 'MobileNetV3_best_model.pth')

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)  
    ax2 = fig.add_subplot(2, 1, 2)  

    ax1.plot(range(1, Epochs + 1), train_loss, label="train_loss")
    ax1.plot(range(1, Epochs + 1), test_loss, label="test_loss")
    ax1.legend()

    ax2.plot(range(1, Epochs + 1), train_acc, label="train_acc")
    ax2.plot(range(1, Epochs + 1), test_acc, label="test_acc")
    ax2.legend()
    plt.show()