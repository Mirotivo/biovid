# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 12:54:04 2017

@author: Amr
"""
#   Mean of the absolute values of the second differences
from base_preprocessing import base_preprocessing
class mavsd_preprocessing(base_preprocessing):
    def apply(self,list_signal):
        input = list_signal[2:] - list_signal[:-2]
        out = sum(abs(i) for i in input)/len(input)
        print('mavsd_preprocessing  :   '+str(out))
        return out