# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 12:54:04 2017

@author: Amr
"""
# Mean of the absolute values of the first differences of the normalized signal
from base_preprocessing import base_preprocessing
class mavfdn_preprocessing(base_preprocessing):
    def apply(self,list_signal):
        input = list_signal[1:] - list_signal[:-1]
        input_normalized=input/max(abs(input))
        out = sum(abs(i) for i in input_normalized)/len(input_normalized)
        print('mavfdn_preprocessing  :   '+str(out))
        return out
