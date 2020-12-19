# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 12:54:04 2017

@author: Amr

"""

#       Spectral Entropy
from base_preprocessing import base_preprocessing
from scipy.fftpack import fft
import math
import numpy
class mav_preprocessing(base_preprocessing):
    def apply(self,list_signal):
        list_signal = numpy.array(list_signal).tolist()
        PSD=abs(fft(list_signal))**2
        """%Normalization"""
        PSD_Norm = (PSD/max(abs(PSD))).tolist()
        """%Entropy Calculation"""
        PSDEntropy = 0
        for x in list(set(PSD_Norm)):
            p_x = float(PSD_Norm.count(x))/len(PSD_Norm)
            if p_x > 0:
                PSDEntropy += - p_x*math.log(p_x, 2)
        return PSDEntropy

    
'''
list_signal = [1,1,1,1,1,1,1,1,1,1]
print (calc(list_signal))
'''