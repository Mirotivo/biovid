# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 12:54:04 2017

@author: Amr

"""

#       Approximate Entropy
from base_preprocessing import base_preprocessing
import numpy as np
class mav_preprocessing(base_preprocessing):
    def apply(self,list_signal):
        def ApEn(d, m, r):
            def _maxdist(x_i, x_j):
                return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
            def _phi(m):
                x = [[d[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
                C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
                return (N - m + 1.0)**(-1) * sum(np.log(C))
            N = len(d)
            return abs(_phi(m + 1) - _phi(m))               
        return ApEn(list_signal, 2, 3)
