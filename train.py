#This is the main function:
#Written by Savion Ragster
#April - May 2022
#this program loads in images and can train a neural network

##To run this program:
#python train.py --train True --arch vgg13  --epochs 2 --device gpu   

import py_compile
from functions import prepro, imshow, predict
from training import *
from training2 import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from get_input_args import get_input_args
#import helper

#HYPERPARAMS TO CHANGE HERE ))))))I***************
in_args = get_input_args()
data_dir = in_args.data_dir
save_dir = in_args.save_dir
epochs = in_args.epochs
device = in_args.device
arch = in_args.arch
train = in_args.train
lr = in_args.learning_rate
hidden_l = in_args.hidden_units

#3 different model architectures to choose from:
#From classifier.py:
vgg16 = models.vgg16(pretrained=True)
vgg13 = models.vgg16(pretrained=True)
vgg19 = models.vgg16(pretrained=True)
models = {'vgg': vgg16, 'vgg13': vgg13, 'vgg19': vgg19}

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#Data Transforms:
test_transform =  transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                                            std = [0.229, 0.224, 0.225])])

train_transform = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                                            std = [0.229, 0.224, 0.225])])
valid_transform =  transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                                             std = [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(train_dir, transform = train_transform)
test_data =  datasets.ImageFolder(test_dir, transform = test_transform)
valid_data = datasets.ImageFolder(valid_dir, transform = valid_transform)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle = True)
valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size = 64, shuffle = True)

#To setup program to use correct device:
if device == 'gpu':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')
else:
    device = torch.device("cpu")
    print(f'device: {device}')

 
model = models[arch]#sets model arch from str input.

for param in model.parameters():
    param.requires_grad = False #this speeds up training and makes features not get updaed #feeze feature params (turn off gradients):
        
model.classifier = nn.Sequential(#attach classifier to model
    nn.Linear(25088, 500),#pass in a fully connected layer - first tuple is inputs, second in neurons in hidden layer
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(500, 102), #There are 102 outputs (different flower options)
    nn.LogSoftmax(dim = 1)
    )#give this a list of ops and it passes a tensor thru it sequentially

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr)#only ßtrain classifier parameters - not features (frozen)
model.to(device);#Move model to device


#To train or not:
if(train):
    training2(model, optimizer, criterion, train_dataloader, test_dataloader, valid_dataloader, device, epochs, 100)
    #trainingf(model, optimizer, criterion, train_dataloader, test_dataloader, valid_dataloader, device, epochs, 100)#old training
#Setup save checkpoint:
checkpoint = {'input_size': 25088,
              'output_size':102,
              'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'class_to_idx': train_data.class_to_idx}
if(train):
    torch.save(checkpoint, save_dir)

def load(filepath):
    checkpoint = torch.load(filepath, map_location = ('cuda' if(device =='cuda') else 'cpu'))#From Udacity Knowlege: Survesh https://knowledge.udacity.com/questions/42098
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

new_model = load(save_dir)