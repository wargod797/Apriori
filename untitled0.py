# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:30:11 2019

@author: sridhar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transaction = []
for i in range(0,7501):
    transaction.append([str(dataset.values[i,j]) for j in range(0,20)])
    
#Training the Apori Dataset
from apyori import apriori
rules = apriori(transaction, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = pd.DataFrame(list(rules))