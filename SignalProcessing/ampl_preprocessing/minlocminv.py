# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 12:54:04 2017

@author: Amr
"""
from base_preprocessing import base_preprocessing
from numpy import mean
from peakdetect import peakdet

class minlocminv_preprocessing(base_preprocessing):
    def apply(self,list_signal):
        print('minlocminv_preprocessing  :   '+str(len(list_signal)))
        maxtab, mintab = peakdet(list_signal,0.001)
        value= mean(mintab[:,1])
        print(value)
        return value