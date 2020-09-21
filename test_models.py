#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 15:28:21 2020

@author: ombretta
"""

import torchvision
import torchvision.transforms as transforms
import torch 

import bagnets.pytorchnet

import tqdm

def validation(valid_loader, model, device, criterion):
                
    model.eval()
    running_valid_loss, total_valid, correct_valid = 0, 0, 0
    
    with torch.no_grad():
        for batch_X, batch_y in tqdm(valid_loader):
            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
        
            outputs, out_logits = model(batch_X) 
            labels = batch_y.long().to(outputs.device)
            loss = criterion(out_logits, labels) 
            _, predictions = torch.max(outputs.detach(), 1)
           
            running_valid_loss += loss.item() 
            total_valid += batch_y.shape[0]
            correct_valid += predictions.eq(labels).sum().item()  
            
    return running_valid_loss/len(valid_loader), correct_valid/total_valid 



device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
imagenet_val_dir = "/tudelft.net/staff-bulk/ewi/insy/CV-DataSets/imagenet/raw-data/val/"

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
imagenet_data = torchvision.datasets.ImageFolder(
        imagenet_val_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

imagenet_data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=1,
                                          shuffle=False)
## BAGNET TESTING ON IMAGENET ##
# available are bagnet9, bagnet17 and bagnet33

models = [bagnets.pytorchnet.bagnet9, 
          bagnets.pytorchnet.bagnet17, 
          bagnets.pytorchnet.bagnet33]

criterion = torch.nn.CrossEntropyLoss()

for model in models:
    print(model)
    pytorch_model = model(pretrained=True) 
    test_loss, test_acc = validation(imagenet_data_loader, model, device, criterion)
        
    print("Test loss", test_loss)
    print("Test accuracy", test_acc)