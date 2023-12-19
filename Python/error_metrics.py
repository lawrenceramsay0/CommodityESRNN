# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:19:14 2023

@author: lawre
"""
#https://www.statology.org/smape-python/
import numpy as np
def smape(a, f):
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)

