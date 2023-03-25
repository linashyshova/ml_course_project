#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 01:37:05 2023

@author: mpetrov
"""

import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression

data = pd.read_csv('../data/train.csv')

train_set = data.sample(frac = 0.9)
train_set_features = train_set.loc[:, train_set.columns != 'label']

validation_set = data.drop(train_set.index)
validation_set_features = validation_set.loc[:, validation_set.columns != 'label']

# logreg = LogisticRegression(max_iter=1000000000, multi_class='multinomial', solver='newton-cg').fit(train_set_features, train_set['label'])

# with open('model_serialized', 'wb') as ouf:
#     pickle.dump(logreg, ouf)

logreg = 0
with open('model_serialized.file', 'rb') as inpf:
    logreg = pickle.load(inpf)

from pprint import pprint
pprint(vars(logreg))
#logreg.predict(validation_set_features)