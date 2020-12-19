# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 12:54:04 2017

@author: Amr
"""

#       Interquartile range
from base_preprocessing import base_preprocessing
class mav_preprocessing(base_preprocessing):
    def apply(self,list_signal):
        out = iqr(list_signal)
        print('mav_preprocessing  :   '+str(out))
        return out

""""
from scipy.stats import iqr
import numpy as np
x = np.array([[10, 7, 4], [3, 2, 1]])
"""