# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 14:01:20 2017

@author: Amir
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_filepaths(directory,extension=''):
   file_paths =[]
   for (dirpath, dirnames, files) in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(dirpath, filename)
            if filename.endswith(extension):
                file_paths.append((filepath,filename))  # Add it to the list.
            

   return file_paths;


def save_plot(values,path):
     print(path)
     for val in values:
         
         plt.scatter(val[0],val[2])
     plt.savefig(path)   
     plt.show()

files=get_filepaths('D:/ASU/Thieses/Pain/BioVid/PartB/filterd_071309_w_21','.csv')
plt.ioff()
for item in files:
    sampleDf=pd.read_csv(item[0]) 
    fil=sampleDf.iloc[:,0]
    values =[]
    currentRow=[]
    print(item[1])
    for  row in  fil:
        currentRow=str(row).split("\t")
        values.append(currentRow) 
    plt.clf()
    save_plot(values,'D:/ASU/Thieses/Pain/BioVid/PartB/filterd_071309_w_21.ecg/'+item[1]+'.png')
   
print('************************Finish*************************')            