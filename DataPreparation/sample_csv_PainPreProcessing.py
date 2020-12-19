import sys
import os 
import pandas as pd
from os import walk
import numpy as np



def from_list_to_csv(_list,filepath):
    import csv
    with open(filepath,'w',newline='') as resultFile:
        wr = csv.writer(resultFile)
        wr.writerows(_list)

sampleDf=pd.read_csv(os.path.join('D:\ASU\Thieses\Pain\BioVid\PartA\starting_point','after_samples.csv'))
fil=sampleDf.iloc[:,0]
values =[]
values.append(['subject_id','subject_name','class_id','class_name','sample_id','sample_name'])
for  row in  fil:
    print(index)
    values.append(str(row).split("\t"))

    
from_list_to_csv(values,os.path.join('D:\ASU\Thieses\Pain\BioVid\PartA\starting_point','after_samples.csv'))    




