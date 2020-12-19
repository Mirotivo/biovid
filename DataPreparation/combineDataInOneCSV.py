import sys 
import os
import pandas as pd


# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 18:44:39 2017

@author: Amir
"""

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
    
    for dir_item in os.listdir('D:/ASU/Thieses/Pain/BioVid/PartA/biosignals_filtered'):
        personpath='D:/ASU/Thieses/Pain/BioVid/PartA/biosignals_filtered'+'/'+dir_item
        values =[]
        filepaths=get_filepaths(personpath,".csv")
        sampleDf=None
        index=0
        for item in filepaths:
              index+=1
              print("indexxxxxxxxxx-----------------"+str(index))
              print(item[0])
              if sampleDf is None :
                  sampleDf=pd.read_csv(item[0]) 
              else:
                  print('Appending-----------------------------')
                  test=pd.read_csv(item[0])
                  sampleDf=sampleDf.append( test)      
        All=sampleDf.iloc[:,:].values   
        path=personpath+'/full.csv'
        from_list_to_csv(All,path)   
    
    return
    
    
def from_list_to_csv(_list,filepath):
    import csv
    with open(filepath,'w',newline='') as resultFile:
        wr = csv.writer(resultFile)
        wr.writerows(_list)    

        
        
main() 