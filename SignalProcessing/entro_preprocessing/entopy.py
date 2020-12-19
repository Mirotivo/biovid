# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 12:54:04 2017

@author: Amr
"""

#       Shannon Entropy
from base_preprocessing import base_preprocessing
class mav_preprocessing(base_preprocessing):
    def apply(self,list_signal):
        import numpy
        import math
        list_signal = numpy.array(list_signal).tolist()
        if not list_signal:
            return 0
        entropy = 0
        for x in list(set(list_signal)):
            p_x = float(list_signal.count(x))/len(list_signal)
            if p_x > 0:
                entropy += - p_x*math.log(p_x, 2)
        return entropy
    
"""
list_signal = [1,1,1,1,1,0,1,1,1,1]
print (apply(list_signal))
"""