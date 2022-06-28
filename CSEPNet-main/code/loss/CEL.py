# -*- coding: utf-8 -*-
# @Time    : 2020/3/28
# @Author  : Lart Pang
# @FileName: CEL.py
# @GitHub  : https://github.com/lartpang

from torch import nn
import torch



class CEL(nn.Module):
    def __init__(self):
        super(CEL, self).__init__()
        print("You are using `CEL`!")
        self.eps = 1e-6

    def forward(self, pred, target):
        pred = pred.sigmoid()
        intersection = pred * target
        numerator = (pred - intersection).sum() + (target - intersection).sum()
        denominator = pred.sum() + target.sum()
        return numerator / (denominator + self.eps)


def _iou(pred, target, size_average):

    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
        Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand1
        IoU1 = Iand1/Ior1

        #IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)

    return IoU/b

class IOU(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):

        return _iou(pred, target, self.size_average)