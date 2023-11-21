from matplotlib import pyplot as plt
from tifffile import tifffile
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from PIL import Image
import numpy as np
from MobileNetv3 import mobilenetv3_large
from dataloader_classify import MyDataset
from torch.utils import data
from torch import optim
import torch.nn as nn
from torch.optim import lr_scheduler
import torch
import cv2
import os

def train(net, device, test_path):
    test_dataset = MyDataset(test_path)
    dataloaders_test = DataLoader(test_dataset,
                                   batch_size=16,
                                   shuffle=True,
                                   num_workers=0
                                   )
    net.load_state_dict(torch.load('MobileNetV3_best_model.pth', map_location=device))
    net.eval()
    test_bar = tqdm(dataloaders_test, file=sys.stdout)
    for step, (x, y, z) in enumerate(test_bar):
        x = x.to(device=device, dtype=torch.float32)
        y_pred = net(x)
        y_pred = torch.argmax(y_pred, dim=1)

        y_pred = y_pred.cuda().data.cpu().numpy()
        y = y.cuda().data.cpu().numpy()

        for i in range(16):
            if y_pred[i] == 0:
                name = str(z[i])
                path = os.path.join('.\\class\\mobilenetv3\\0', name)
                im = y[i, :, :, :]
                I = np.uint8(im)
                cv2.imwrite(path, I)
            elif y_pred[i] == 1:
                name = str(z[i])
                path = os.path.join('.\\class\\mobilenetv3\\1', name)
                im = y[i, :, :, :]
                I = np.uint8(im)
                cv2.imwrite(path, I)
            else:
                name = str(z[i])
                path = os.path.join('.\\class\\mobilenetv3\\2', name)
                im = y[i, :, :, :]
                I = np.uint8(im)
                cv2.imwrite(path, I)

def rescale(im):
    im_min = im.min()
    im_max = im.max()
    return (im - im_min) / (im_max - im_min)

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
    test_path = './test/'

    model = mobilenetv3_large()
    model.classifier = nn.Sequential(
            nn.Linear(960, 1280),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(1280, 3),
        )
    model.to(device)
    train(model, device, test_path)