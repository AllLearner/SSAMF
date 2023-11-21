import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
import os
from simplecv import mean_std_normalize
from PIL import Image
from tqdm import tqdm
import sys
import os
import cv2


class MyDataset(Dataset):
    def __init__(self, path, transform=None):
        self.transform = transforms.Compose([
            transforms.ToTensor()  
        ])
        self.image_path = path
        self.image_names = os.listdir(self.image_path)

    def __len__(self):
        return len(self.image_names)

    def Normalize(self, image):
        im_cmean = image.reshape((-1, image.shape[-1])).mean(axis=0)
        im_cstd = image.reshape((-1, image.shape[-1])).std(axis=0)
        image = mean_std_normalize(image, im_cmean, im_cstd)
        return image

    def __getitem__(self, item):
        image_name = self.image_names[item]
        path = os.path.join(self.image_path, image_name)
        image = cv2.imread(path)
        #image = self.Normalize(image)
        image1 = image.reshape(image.shape[2], image.shape[0], image.shape[1])
        #image1 = image.tranpose(2, 0, 1)
        return image1, image, image_name

if __name__ == "__main__":
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    print("using {} device.".format(device))
    path = 'C:\\Users\\user7\\project\\bingchonghai\\test1'
    image_dataset = MyDataset(path)
    dataloaders_test = DataLoader(image_dataset,
                                   batch_size=1,
                                   shuffle=False
                                   )
    print(len(dataloaders_test))
    print(len(image_dataset))
    test_bar = tqdm(dataloaders_test, file=sys.stdout)
    for step, (image, im, name) in enumerate(test_bar):
        print(image.shape)
        print(im.shape)
        print(name)