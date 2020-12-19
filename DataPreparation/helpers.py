import pandas as pd
import numpy as np


class from_csv_to_nparray(object):
    def __init__(self):
        print('from_csv_to_nparray')
    
    def get_nparray(self,filepath):
        dataframe=pd.read_csv(filepath)
        lst=dataframe.iloc[:,0]
        All=[]
        for item in lst:
            current=str(item).split("\t")
            All.append([current[0],float(current[1]),float(current[2]),float(current[3]),float(current[4]),float(current[5])])
        vs=np.array(All,dtype=float)
        return vs
        

