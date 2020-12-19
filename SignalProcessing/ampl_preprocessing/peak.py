# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 12:54:04 2017

@author: Amr
"""
import numpy as np

from base_preprocessing import base_preprocessing
class peak_preprocessing(base_preprocessing):
    def apply(self,list_signal):
        print('peak_preprocessing  :   '+str(len(list_signal)))        
        value= list_signal[np.argmax(list_signal,axis=0)]
        print(value)
        return value