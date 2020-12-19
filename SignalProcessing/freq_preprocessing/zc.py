# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 12:54:04 2017

@author: Amr
"""

#       fMode
from base_preprocessing import base_preprocessing
class fmode_preprocessing(base_preprocessing):
    def apply(self,list_signal):
        import numpy as np
        ACF = list_signal[:-1]*list_signal[1:]
        ZC = ACF[ np.where( ACF <0 ) ]
        out = len(ZC)
        print('fmode_preprocessing  :   '+str(out))
        return out