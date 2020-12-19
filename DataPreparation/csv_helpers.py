# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 17:44:16 2017

@author: Amir
"""
import csv
        
class csv_helpers(object):
    
    def from_list_to_csv(self,_list,filepath):
        with open(filepath,'w',newline='') as resultFile:
            wr = csv.writer(resultFile)
            wr.writerows(_list)    
