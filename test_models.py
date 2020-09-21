#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 15:28:21 2020

@author: ombretta
Using code from https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""

import torchvision
import torchvision.transforms as transforms
import torch 

import bagnets.pytorchnet

import tqdm
import time

# def validation(valid_loader, model, device, criterion):
                
#     model.eval()
#     running_valid_loss, total_valid, correct_valid = 0, 0, 0
    
#     with torch.no_grad():
#         for batch_X, batch_y in tqdm(valid_loader):
            
#             batch_X = batch_X.to(device)
#             batch_y = batch_y.to(device)
        
#             outputs, out_logits = model(batch_X) 
#             labels = batch_y.long().to(outputs.device)
#             loss = criterion(out_logits, labels) 
#             _, predictions = torch.max(outputs.detach(), 1)
           
#             running_valid_loss += loss.item() 
#             total_valid += batch_y.shape[0]
#             correct_valid += predictions.eq(labels).sum().item()  
            
#     return running_valid_loss/len(valid_loader), correct_valid/total_valid 


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    
    
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
    
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
    
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
    
def validate(val_loader, model, device, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images.to(device)
            target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 8 == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg




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
                                          batch_size=8,
                                          shuffle=False,
                                          num_workers=2)

## BAGNET TESTING ON IMAGENET ##
# available are bagnet9, bagnet17 and bagnet33

models = [bagnets.pytorchnet.bagnet9, 
          bagnets.pytorchnet.bagnet17, 
          bagnets.pytorchnet.bagnet33]

criterion = torch.nn.CrossEntropyLoss()

for model in models:
    print(model)
    pytorch_model = model(pretrained=True) 

    test_loss, test_acc = validate(imagenet_data_loader, pytorch_model, device, criterion)
        
    print("Test loss", test_loss)
    print("Test accuracy", test_acc)
