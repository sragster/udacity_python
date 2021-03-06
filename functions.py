#all the functions for the Image classifier Part 2
#Written by Savion Ragster #April - May 2022

import torch
import torch.nn as nn
import numpy as np
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from get_input_args import get_input_args
from PIL import Image

def prepro(image_path):
    pic = Image.open(image_path)
    transform = transforms.Compose([transforms.Resize(256), 
                                    transforms.CenterCrop(224), 
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    pic2 = transform(pic)#do the transform
    np_image = np.array(pic2)#convert to np array
    return pic2


def imshow(image, ax=None, title=None):
   # """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax

def predict(image_path, model, cat_to_name, device, top_k):   
    classes = []
    labels = []
    top_labels = []
    top_probs = []
    
    single_img = prepro(image_path).unsqueeze_(0)#adds dim for it to work (batsch size)
    model.to(device)#move model to device
    single_img  = single_img.to(device)#move image to device
    logps = model(single_img)#This works pass img thru model
    ps = torch.exp(logps)
    top_ps, top_class = ps.topk(top_k, dim = 1)
    
    #Make sure to run this functiononly once (flip the dic only one time):
    model.class_to_idx = dict(map(reversed, model.class_to_idx.items()))#this reverses the dictionary
    top_class = np.array(top_class)
    top_ps = top_ps.cpu().detach().numpy()#move top probablilites to cpu after prediction then convert to numpy for math below:
  
    #From top_class tensor- convert these ints(classes) to image labels:
    for item in range(top_k):
        top_probs.append(top_ps[0][item])
    for item in range (top_k):
        classes.append(top_class[0][item])
    for item in classes:
        labels.append(model.class_to_idx[item])
    for item in labels:
        top_labels.append(cat_to_name[item])
    return top_labels, top_probs