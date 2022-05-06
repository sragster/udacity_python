#version 3 
#Training:

import torch
import torch.nn as nn
import numpy as np
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
#from train.py import device#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trainingf(model, optimizer, criterion, train_dataloader, test_dataloader, valid_dataloader, device, epochs, print_every = 100):
    steps = 0 
    train_loss = 0
    train_losses, test_losses = [], []


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
                test_loss = 0 
                test_accuracy = 0
                with torch.no_grad():
                    for images, labels in test_dataloader:
                        images, labels = images.to(device), labels.to(device)
                        logps = model.forward(images)   #
                        batch_loss = criterion(logps, labels)#changed
                        test_loss += batch_loss.item()#changed
                    
                        #calculate accuracy:
                        ps = torch.exp(logps)#to get actual probabilities 
                        top_ps, top_class = ps.topk(1, dim = 1)#gives us first largest probability - along the col
                        equality = top_class == labels.view(*top_class.shape)
                        test_accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                
                test_losses.append(test_loss/len(test_dataloader))
                train_losses.append(train_loss/len(train_dataloader))
            
                print(f"Epoch {epoch+1}/{epochs}.."
                    f"Train loss: {train_loss/print_every: .3f}.."
                    f"Test loss {test_loss/len(test_dataloader):.3f}.."
                    f"Test accuracy: {test_accuracy/len(test_dataloader):.3f}"
                    )
                train_loss = 0 #reset running loss {original location}
                model.train()#put back into training mode
    return model        