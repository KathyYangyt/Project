#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(weight_list):
    weight_avg = copy.deepcopy(weight_list[0])
    for w in weight_avg.keys():
        for i in range(1,len(weight_list)):
            weight_avg[w] += weight_list[i][w]
        weight_avg[w] = torch.div(weight_avg[w], len(weight_list))
    return weight_avg