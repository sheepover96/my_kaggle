import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import pandas as pd
from sklearn.model_selection import train_test_split
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

LABEL_IDX = 1
IMG_IDX = 2

class ToTensor:

    def __call__(self, image):
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)

class Normalize:

    def __call__(self, image):
        shape = image.shape
        image = (image - np.mean(image))/np.std(image)*16+64
        return image


class MyDataset(Dataset):

    def __init__(self, csv_file_path, root_dir, transform=None):
        self.image_dataframe = pd.read_csv(csv_file_path)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_dataframe)

    def __getitem__(self, idx):

        label = self.image_dataframe.iat[idx, LABEL_IDX]
        img_name = os.path.join(self.root_dir, 'classification-of-handwritten-letters',
                'letters2', self.image_dataframe.iat[idx, IMG_IDX])
        image = io.imread(img_name)

        if self.transform:
            image = self.transform(image)

        return image, label


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(4, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 100)
        self.fc2 = nn.Linear(100, 34)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


argvs = sys.argv
if len(argvs) != 2:
    print('no input file')
    sys.exit()

input_file_path = argvs[1]
imgDataset = MyDataset(input_file_path, ROOT_DIR, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ]))

train_data, test_data = train_test_split(imgDataset, test_size=0.2)
print(type( train_data ))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True)

model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    correct = 0
    total = 0
    for batch_idx, (image, label) in enumerate(train_loader):
        image, label = Variable(image), Variable(label)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(image), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.data[0]))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    conf_matrix = [ [ 0 for j in range(34) ] for i in range(34)]
    for (image, label) in test_loader:
        image, label = Variable(image.float(), volatile=True), Variable(label)
        output = model(image)
        test_loss += criterion(output, label).data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        flabel = label.data.view_as(pred)
        correct += pred.eq(flabel).long().cpu().sum()
        for i in range(34):
            for j in range(34):
                conf_matrix[i][j] += pred[flabel==i].eq(j).long().cpu().sum().item()

    print(conf_matrix)
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, 1000 + 1):
    train(epoch)
    test()
