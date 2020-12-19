# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 12:54:04 2017

@author: Amr
"""
import numpy as np

from base_preprocessing import base_preprocessing
class rms_preprocessing(base_preprocessing):
    #rms = rms(signal)
    def apply(self,list_signal):
        print('rms_preprocessing  :   '+str(len(list_signal)))
        rms = np.sqrt(np.mean(list_signal**2))
        print(rms)
        return rms