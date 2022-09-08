import copy
import torch


def FedAvg(weight_list):
    weight_avg = copy.deepcopy(weight_list[0])
    for k in weight_avg.keys():
        for i in range(1,len(weight_list)):
            weight_avg[k] += weight_list[i][k]
        weight_avg[k] = weight_avg[k]/len(weight_list)
    return weight_avg