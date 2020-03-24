#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 10:29:04 2020

@author: ombretta
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    
    def __init__(self, weight=None, 
                 gamma=2., reduction='none'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input_tensor, target_tensor):
        # log_prob = F.log_softmax(input_tensor, dim=-1) # Alredy in the model
        log_prob = input_tensor
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob, 
            target_tensor, 
            weight=self.weight,
            reduction = self.reduction
        )