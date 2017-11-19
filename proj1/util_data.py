#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



class DataSet:
    train_data = pd.read_csv('data/train.csv',na_values="-1")
    test_data = pd.read_csv('data/test.csv',na_values="-1")
    
    def __init__(self):
        return
    
    def get_training_set(self,n = None):
        if(n):
            return(self.train_data.sample(n))
        else:
            return(self.train_data)
    
    def get_testing_set(self, n = None):
        if(n):
            return(self.test_data.sample(n))
        else:
            return(self.test_data)
    
    def get_scaled_training_set(self, n_zeros, n_ones = None):
        zeros = self.train_data[self.train_data['target']==0]
        ones = self.train_data[self.train_data['target']==1]
        
        if(n_ones):
            return(pd.concat([zeros.sample(n_zeros),ones.sample(n_ones)],axis=0))
        
        else:
            return(pd.concat([zeros.sample(n_zeros),ones],axis=0))
    
    def output(self,Y):
        res = []
        for i,p in zip(self.test_data.id, Y):
            res.append([i,max(0,p)])
        pd.DataFrame(res,columns=["id","target"]).to_csv("prediction.csv",index = False)


    
    
