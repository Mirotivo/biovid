# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 12:54:04 2017

@author: Amr
"""
import numpy as np
from base_preprocessing import base_preprocessing
class p2p_preprocessing(base_preprocessing):
    def apply(self,list_signal):
        
        
        #p2p = max(signal) - min(signal)
        print('p2p_preprocessing  :   '+str(len(list_signal)))
        value = float(list_signal[np.argmax(list_signal,axis=0)]) - float(list_signal[np.argmin(list_signal,axis=0)])
        print(value)
        return value