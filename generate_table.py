import pandas as pd
import numpy as np
import os
import sys
import glob


class gen_table:
    def __init__(self, datacsv, total_versions=4):
        self.datacsv = pd.read_csv(datacsv)
        self.datacsvnp = self.datacsv.to_numpy()
        self.columns = list(self.datacsv.columns)
        self.versions = total_versions
        self.dataframe = []

    def genbasictable(self):
        heading = self.columns
        heading[0] = " "
        rows = ['%', 'Mean Threshold', 'Mean Threshold', 'Intersection Threshold', 'Mean Threshold Clipped', 'Intersection Threshold Clipped', 'Mean Threshold Normalised', 'Intersection Threshold Normalised', 'Accuracy Mean Threshold Version 1', 'Accuracy Intersection Threshold Version 1',
                'Accuracy Mean Threshold Version 2', 'Accuracy Intersection Threshold Version 2', 'Accuracy Mean Threshold Version 3', 'Accuracy Intersection Threshold Version 3', 'Accuracy Mean Threshold Version 4', 'Accuracy Intersection Threshold Version 4']
        for i in rows:
            row_data = [i]
            for i in range(len(self.columns)-1): row_data.append(0.00)
            self.dataframe.append(row_data)
        
    def fill_nonacc_columns(self):
        for i in range(1,len(self.columns)): self.dataframe[0][i]=self.datacsvnp[:,i].mean()
    
    # def fill_acc_columns(self):
# gen = gen_table('./data.csv')
# gen.genbasictable()
# gen.fill_nonacc_columns()