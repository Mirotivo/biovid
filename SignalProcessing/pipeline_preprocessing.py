# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 05:20:34 2017

@author: Amir
"""
from base_preprocessing import base_preprocessing
from ampl_preprocessing.rms import rms_preprocessing 
from ampl_preprocessing.p2p import p2p_preprocessing 
from ampl_preprocessing.peak import peak_preprocessing
#from ampl_preprocessing.mlocmaxv import mlocmaxv_preprocessing
#from ampl_preprocessing.minlocminv import minlocminv_preprocessing
from ampl_preprocessing.mav import mav_preprocessing
from ampl_preprocessing.mavfd import mavfd_preprocessing
from ampl_preprocessing.mavfdn import mavfdn_preprocessing
from ampl_preprocessing.mavsd import mavsd_preprocessing
from ampl_preprocessing.mavsdn import mavsdn_preprocessing

class pipeline_preprocessing(object):
    def __init__(self):
        pass

        
    def apply(self,list_signal):
        print('Applying pipeline.......')
        kl=self.FindAllSubclasses(base_preprocessing)
        result=[]
        for item in kl:
            obj_preprocessing= globals()[item[1]]
            instance=obj_preprocessing()
            result.append((item[1],instance.apply(list_signal)))
        return result

    def FindAllSubclasses(self,classType):
        import sys, inspect
        subclasses = []
        callers_module = sys._getframe(1).f_globals['__name__']
        classes = inspect.getmembers(sys.modules[callers_module], inspect.isclass)
        for name, obj in classes:
            if (obj is not classType) and (classType in inspect.getmro(obj)):
                subclasses.append((obj, name))
        return subclasses