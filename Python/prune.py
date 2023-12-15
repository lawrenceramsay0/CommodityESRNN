# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 22:48:41 2023

@author: lawre
"""
from torch.nn.utils import prune


def prune_model_l1_unstructured(model, layer_type, proportion):
    for module in model.modules():
        if isinstance(module, layer_type):
            prune.l1_unstructured(module, 'weight', proportion)
            prune.remove(module, 'weight')
    return model