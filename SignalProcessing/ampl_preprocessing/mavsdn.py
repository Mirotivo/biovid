# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 12:54:04 2017

@author: Amr
"""
# Mean of the absolute values of the second differences of the normalized signal
from base_preprocessing import base_preprocessing
class mavsdn_preprocessing(base_preprocessing):
    def apply(self,list_signal):
        input = list_signal[2:] - list_signal[:-2]
        input_normalized=input/max(abs(input))
        out = sum(abs(i) for i in input_normalized)/len(input_normalized)
        print('mavsdn_preprocessing  :   '+str(out))
        return out