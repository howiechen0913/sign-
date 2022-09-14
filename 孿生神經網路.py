# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 17:37:00 2022

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 13:05:21 2022

@author: user
"""

import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import PIL.ImageOps

import torchvision
import torchvision.datasets as datasets
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
import pandas as pd
import time
from efficientnet_pytorch import EfficientNet


device = torch.device("cuda:0")
use_gpu = torch.cuda.is_available()

def img_loader(img_path):
    img = Image.open(img_path[0])
    return img.convert("RGB")

def make_dataset(image_path,name_list,label_list):
    samples = []

    for img_name, label in zip(name_list, label_list):
        

        img_name = img_name + '.png'
        target = label

        img_path = os.path.join(image_path,img_name)
        samples.append([img_path, target])


    return samples

class CaptchaData(Dataset):
    def __init__(self, image_path,name_list,label_list,
                 transform=None, target_transform=None):
        super(Dataset, self).__init__()
        self.image_path = image_path
        self.name_list = name_list
        self.label_list = label_list

        self.transform = transform
        self.target_transform = target_transform
        self.samples = make_dataset(self.image_path,self.name_list,self.label_list
                                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img0_tuple = random.choice(self.samples)
        should_get_same_class = random.randint(0,1)

        if should_get_same_class:
            while True:
                img1_tuple = random.choice(self.samples)

                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                img1_tuple = random.choice(self.samples)

                if img0_tuple[1] != img1_tuple[1]:
                    break
        img_0 = img_loader(img0_tuple)
        img_1 = img_loader(img1_tuple)
        if self.transform is not None:
            img_0 = self.transform(img_0)
            img_1 = self.transform(img_1)
            
        if self.target_transform is not None:
            img0_target = self.target_transform(img0_tuple[1])
            img1_target = self.target_transform(img1_tuple[1])

        return img_0, img_1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])]))


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        # Setting up the Sequential of CNN Layers
        self.features = nn.Sequential(
            
            nn.Conv2d(3, 96, kernel_size=11,stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

            nn.Conv2d(256,384 , kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,256 , kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

        )
        

        self.classifier = nn.Sequential(
            nn.Linear(256*29*13, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128,2))
        
  
  
    def forward_once(self, x):
        # Forward pass 
        output = self.features(x)

        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        return output

    def forward(self, input1, input2):

        output1 = self.forward_once(input1)

        output2 = self.forward_once(input2)
        return output1, output2

class SiameseNetwork2(nn.Module):

    def __init__(self):
        super(SiameseNetwork2, self).__init__()

        
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11,stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3,stride=1),
            nn.ReLU(inplace=True)
        )

        
        self.classifier = nn.Sequential(
            nn.Linear(384*11*3, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256,2)
        )
        
    def forward_once(self, x):

        output = self.features(x)

        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        return output

    def forward(self, input1, input2):

        output1 = self.forward_once(input1)

        output2 = self.forward_once(input2)

        return output1, output2

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      
      euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)

      loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


      return loss_contrastive


train_image_path = 'D:/新光實習/孿生神經網路數據/整理後的訓練數據/手寫簽名_train_img'
train_csv_path = 'D:/新光實習/孿生神經網路數據/整理後的訓練數據/手寫簽名_train_label/train_label.csv'
df_csv = pd.read_csv(train_csv_path)
train_label_list = df_csv['label'].tolist()
train_name_list = df_csv['image'].tolist()


train_transforms = transforms.Compose([
    transforms.Resize((256,128)),

    transforms.ToTensor(),

                                ])

val_image_path = 'D:/新光實習/孿生神經網路數據/整理後的訓練數據/手寫簽名_val_img'
val_csv_path = 'D:/新光實習/孿生神經網路數據/整理後的訓練數據/手寫簽名_val_label/val_label.csv'
df_csv = pd.read_csv(val_csv_path)
val_label_list = df_csv['label'].tolist()
val_name_list = df_csv['image'].tolist()

val_transforms = transforms.Compose([
    transforms.Resize((256,128)),

    transforms.ToTensor(),

                                ])


dataset_sizes = 20

train_dataset = CaptchaData(train_image_path,train_name_list,train_label_list,transform = train_transforms)
train_data_loader = DataLoader(train_dataset, batch_size=dataset_sizes, num_workers=0,shuffle=True)
train_dataset_sizes = len(train_dataset.samples)

val_dataset = CaptchaData(val_image_path,val_name_list,val_label_list,transform = val_transforms)
val_data_loader = DataLoader(val_dataset, batch_size=dataset_sizes, num_workers=0,shuffle=True)
val_dataset_sizes = len(val_dataset.samples)

n = 0


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_loss = 100
    n = 0
    train_counter = []
    val_counter = []
    train_loss_history = [] 
    val_loss_history = [] 
    train_iteration_number= 0
    val_iteration_number= 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0
            if phase == 'train':
                n = 0
                for i, (data0, data1, label) in enumerate(train_data_loader):

                    input0, input1, label = data0, data1, label

                    if use_gpu == True :
                        input0, input1, label = Variable(input0.cuda(device)), Variable(input1.cuda(device)), Variable(label.cuda(device))

                    optimizer.zero_grad()
                    output1 = model(input0)
                    output2 = model(input1)

                    loss = criterion(output1, output2, label)

                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                    n+=1

                epoch_loss = running_loss / n

                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            if phase == 'val':
                n = 0
                for i, (data0, data1, label) in enumerate(val_data_loader):
                    input0, input1, label = data0, data1, label
                    if use_gpu == True :
                        input0, input1, label = Variable(input0.cuda(device)), Variable(input1.cuda(device)), Variable(label.cuda(device))
                    output1 = model(input0)
                    output2 = model(input1)
                    loss = criterion(output1, output2, label)
                    running_loss += loss.item()

                    n+=1
                epoch_loss = running_loss / n

                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = model.state_dict()
                    
        time_elapsed = time.time() - since
        print('Training complete in {:0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val loss: {:4f}'.format(best_loss))
        model.load_state_dict(best_model_wts)
        torch.save(model, 'D:/新光實習/孿生神經網路數據/model/test_model.pkl')
        torch.save(model.state_dict(), 'D:/新光實習/孿生神經網路數據/model/test_model-p.pkl')

if __name__ == '__main__':
    model =SiameseNetwork2()
    if use_gpu == True:
        model = model.cuda(device)
    criterion = ContrastiveLoss().cuda(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer,[18,30,40,55], gamma=0.8)
    train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=70)