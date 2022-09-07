#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
import os
from PIL import Image
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
from sklearn import preprocessing

def test_image(net_g, datatest, args):
    net_g.eval()
   
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=16)
    
    length_dataset= len(data_loader.dataset)
    for idx, (data, label) in enumerate(data_loader):

        data, label = data.to(args.device), label.to(args.device)
        log_probs = net_g(data)
        
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, label, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(label.data.view_as(y_pred)).long().cpu().sum()

    _, predicted = torch.max(log_probs, 1)
    
    # print label and predict label
    print('label:', label)
    print('predict:', predicted)
    
    test_loss /= length_dataset
    test_accuracy = 100.00 * correct / length_dataset
    
    return test_accuracy, test_loss

