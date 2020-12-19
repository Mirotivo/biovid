# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 12:28:20 2017

@author: Amir
"""

import os
import scipy


class base_preprocessing(object):
    
    def  __init__(self):
        print('ini  '+self.__class__.__name__)
        
    def apply(self,list_signal):
        print(len(list_signal))
