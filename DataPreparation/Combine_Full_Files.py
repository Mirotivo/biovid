import sys 
import os
import pandas as pd


# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 18:44:39 2017

@author: Amir
"""


def main(): 
    index=0
    sampleDf=None
    for dir_item in os.listdir('D:/ASU/Thieses/Pain/BioVid/PartA/biosignals_filtered'):
        personpath='D:/ASU/Thieses/Pain/BioVid/PartA/biosignals_filtered'+'/'+dir_item
        values =[]
        path=personpath+'/full.csv'
       
        index+=1
        print("indexxxxxxxxxx-----------------"+str(index))
        print(path)
        if sampleDf is None :
            sampleDf=pd.read_csv(path) 
        else:
            print('Appending-----------------------------')
            test=pd.read_csv(path)
            sampleDf=sampleDf.append( test)      
    All=sampleDf.iloc[:,:].values   
    fullpath='D:/ASU/Thieses/Pain/BioVid/PartA/full.csv'
    from_list_to_csv(All,fullpath)   
    
    return
    
    
def from_list_to_csv(_list,filepath):
    import csv
    with open(filepath,'w',newline='') as resultFile:
        wr = csv.writer(resultFile)
        wr.writerows(_list)    

        
        
main() 