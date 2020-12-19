# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 17:59:12 2017

@author: Amir
"""

import os
import scipy
import sys
from pipeline_preprocessing import pipeline_preprocessing
import pandas as pd
import numpy as np
from helpers import from_csv_to_nparray
from csv_helpers import csv_helpers

class convert_Data_To_Feature(object):
    def convert(self,directory,newcsvpath):
        files=get_filepaths(directory)
        csv=from_csv_to_nparray()
        pipeline_pre =pipeline_preprocessing()
        result=[]
        All=[]
        for file in files:
            vs=csv.get_nparray(file[0])
            gsr=vs[:,1]
            ecg=vs[:,2]
            emg_trapezius=vs[:,3]
            emg_corrugator=vs[:,4]
            emg_zygomaticus=vs[:,5]
            result_gsr=pipeline_pre.apply(gsr) 
            result_emg_trapezius=pipeline_pre.apply(emg_trapezius) 
            result_emg_corrugator=pipeline_pre.apply(emg_corrugator) 
            result_emg_zygomaticus=pipeline_pre.apply(emg_zygomaticus)
            result=getvector(result_gsr)+getvector(result_emg_trapezius)+getvector(result_emg_corrugator)+getvector(result_emg_zygomaticus)
            result.append(getResult(file[1]))
            All.append(result)
        csv =csv_helpers()
        csv.from_list_to_csv(All,newcsvpath)
        
def get_filepaths(self,directory,extension=''):
   file_paths =[]
   for (dirpath, dirnames, files) in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(dirpath, filename)
            if filename.endswith(extension):
                file_paths.append((filepath,filename))  # Add it to the list.
            

   return file_paths;
   
def getResult(item):
        print(item)
        result=0
        if "BL1" in item:
            result=0
        elif "PA1" in item:
            result=1
        elif "PA2" in item:
            result=2
        elif "PA3" in item:
            result=3
        elif "PA4" in item:
            result=4 
        return result


def getvector(result_tuble):
    result =[]
    for item in result_tuble:
        result.append(item[1])
    return result






        