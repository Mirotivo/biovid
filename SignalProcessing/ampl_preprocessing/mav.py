# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 12:54:04 2017

@author: Amr
"""

#       Mean Absolute Value
from base_preprocessing import base_preprocessing
class mav_preprocessing(base_preprocessing):
    def apply(self,list_signal):
        out = sum(abs(i) for i in list_signal)/len(list_signal)
        print('mav_preprocessing  :   '+str(out))
        return out
