# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 12:54:04 2017

@author: Amr
"""

#       fmedian
from base_preprocessing import base_preprocessing
class fmode_preprocessing(base_preprocessing):
    def apply(self,list_signal):
        from scipy.fftpack import fft
        import numpy as np
        # Number of sample points
        N = 2816
        # sample spacing
        T = 1.0 / 1953
        # y = time domain signal
        y = list_signal
        yf = fft(y)
        Yf = 2.0/N * np.abs(yf[0:N//2])
        out = np.median(Yf)
        print('fmedian_preprocessing  :   '+str(out))
        return out