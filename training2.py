#version 3.1 
#Training2:
#Written by Savion Ragster
#May 2022
#Used for training and displaying stats. 

#Issues:
# Accuracy is super low - not sure why - code is copied from the Jupyter notebook where
# it always easily gets to >70% accuracy with 3 epochs
import torch
import torch.nn as nn
import numpy as np
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


def training2(model, optimizer, criterion, train_dataloader, test_dataloader, valid_dataloader, device, epochs, print_every = 100):
    steps = 0 
    train_loss = 0
 
    for epoch in range(epochs):
        #running_loss = 0
        for images, labels in train_dataloader:
            steps += 1 
            images, labels = images.to(device), labels.to(device)#move stuff
            optimizer.zero_grad()
            logps = model(images)#these are the logits - the raw output of the nural network - not probabilities. 
            loss = criterion(logps, labels)#calculate the loss from the logits and labels. 
            loss.backward()#backward pass to calculate gradient tensor
            optimizer.step()#this updates the gradients 
            train_loss += loss.item()
           

            if steps % print_every == 0:#enter evaluation loop
                model.eval()#turn off dropout  switches into evaluation
                valid_loss = 0 
                valid_accuracy = 0
                with torch.no_grad():
                    for images, labels in valid_dataloader:
                        images, labels = images.to(device), labels.to(device)
                        logps = model.forward(images)   #
                        batch_loss = criterion(logps, labels)#
                        valid_loss += batch_loss.item()#

                        #calculate accuracy:
                        ps = torch.exp(logps)#to get actual probabilities 
                        top_ps, top_class = ps.topk(1, dim = 1)#gives us first largest probability - along the col
                        equality = top_class == labels.view(*top_class.shape)
                        valid_accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                        print(valid_accuracy)

             
                print(f"Epoch {epoch+1}/{epochs}.."
                      f"Train loss: {train_loss/print_every: .3f}.."
                      f"Valid loss {valid_loss/len(valid_dataloader):.3f}.."
                      f"Valid accuracy: {valid_accuracy/len(valid_dataloader):.3f}"
                      )
                train_loss = 0 #reset running loss {original location}
                model.train()#put back into training mode
    return model
#version 3.1