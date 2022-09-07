#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch


from options import args_parser
from update import LocalUpdate
from models import CMnist_Net
from FedAvg import FedAvg
from test import test_image
from getData import GetDataset


if __name__ == '__main__':
    # parse args
    args = args_parser()

    # select device
    if torch.cuda.is_available() and args.gpu != -1:
      args.device = torch.device('cuda:{}'.format(args.gpu))
      print('gpu')
    else:
      args.device = torch.device('cpu')
      print('cpu')
    
    # load dataset and split dataset to clients
    if args.dataset == 'Cmnist':
       dataset_train , dataset_test , group_clients = GetDataset(args)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.dataset == 'Cmnist':
        global_net = CMnist_Net(args=args).to(args.device)
    else:
        exit('Error: unrecognized model')

    # copy weights
    weight_global = global_net.state_dict()

    # training
    print(global_net)
    global_net.train()
    loss_train = []   

    for iter in range(args.global_epochs):
        loss_locals = []
        weight_locals = []
        m = max(int(args.fraction * args.num_clients), 1)
        idxs_clients = np.random.choice(range(args.num_clients), m, replace=False)
        print("The selected clients set:")
        for idx in idxs_clients:
            print(idx,end=",")

            local = LocalUpdate(args=args, dataset=dataset_train, idxs=group_clients[idx])
            weight, loss = local.train(net=copy.deepcopy(global_net).to(args.device))
            weight_locals.append(copy.deepcopy(weight))
            loss_locals.append(copy.deepcopy(loss))
        print()
        # update global weights
        weight_global = FedAvg(weight_locals)

        # copy weight to global_net
        global_net.load_state_dict(weight_global)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter+1, loss_avg))
        print()
        loss_train.append(loss_avg)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_C{}_iid{}.png'.format(args.dataset,args.global_epochs, args.fraction, args.iid))

    # testing
    global_net.eval()
    train_accuracy, train_loss = test_image(global_net, dataset_train, args)
    test_accuracy, test_loss = test_image(global_net, dataset_test, args)
    print("Training accuracy: {:.2f}".format(train_accuracy))
    print("Testing accuracy: {:.2f}".format(test_accuracy))

