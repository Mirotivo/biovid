# -*- coding: utf-8 -*-
"""
Created on Tue May  2 00:17:11 2017

@author: Amir
"""
import sys
import os 
import pandas as pd
from os import walk
import numpy as np


def get_filepaths(directory,extension=''):
   file_paths =[]
   for (dirpath, dirnames, files) in walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(dirpath, filename)
            if filename.endswith(extension):
                file_paths.append((filepath,filename))  # Add it to the list.
            

   return file_paths;
   
   
   
def main(): 
    
    values =[]
    for dir_item in os.listdir('C:/Users/Amir/OneDrive/Thesis/Dataset'):
        personpath='C:/Users/Amir/OneDrive/Thesis/Dataset'+'/'+dir_item
       
        filepaths=get_filepaths(personpath,".csv")
        index=0
        for item in filepaths:
              index+=1
              df=pd.read_csv('D:/071309_w_21-BL1-081.csv')
              df=df.drop('FrameIndex',1)
              result=0
              if "BL1" in item[1]:
                   result=0
              elif "PA1" in item[1]:
                   result=1
              elif "PA2" in item[1]:
                   result=2
              elif "PA3" in item[1]:
                   result=3
              elif "PA4" in item[1]:
                   result=4 
              
              feature =[]
              for column in df:
                  feature.append(np.max(df[column]))
                  feature.append(np.mean(df[column]))
                  print('max for '+column+' = ' +str(np.max(df[column])))
                  print('mean for '+column+' = '+str(np.mean(df[column])))           
              print("indexxxxxxxxxx-----------------"+str(index))
              print(item[0])
              feature.append(result)
              values.append(feature)  
    path='D:/full.csv'
    from_list_to_csv(values,path)   
    
    return
    
    
def from_list_to_csv(_list,filepath):
    import csv
    with open(filepath,'w',newline='') as resultFile:
        wr = csv.writer(resultFile)
        wr.writerows(_list)    

        
        
main() 
