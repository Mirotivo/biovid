import sys
import os 
import pandas as pd
from os import walk
import numpy as np


#lma tb2a 3aiz trg3 el files kolha 
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
    filepaths=get_filepaths('D:/ASU/Thieses/Pain/BioVid/PartA/biosignals_filtered',".csv")
    for item in filepaths:
        print(item)
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
        modify_file(item[0],result)
    return




def from_list_to_csv(_list,filepath):
    import csv
    with open(filepath,'w',newline='') as resultFile:
        wr = csv.writer(resultFile)
        wr.writerows(_list)

def modify_file(filepath,result):        
    sampleDf=pd.read_csv(filepath)
    fil=sampleDf.iloc[:,0]
    values =[]
    currentRow=[]
    values.append(['time','gsr','ecg','emg_trapezius','emg_corrugator','emg_zygomaticus','Result'])
    for  row in  fil:
        currentRow=str(row).split("\t")
        currentRow.append(result)
        values.append(currentRow) 
    from_list_to_csv(values,filepath)    
    



    
    
main()
