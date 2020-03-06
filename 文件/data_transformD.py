# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from  seriesToSupervisedLearning import seriesToSupervisedLearning
import types

'''不换行显示'''
pd.set_option('expand_frame_repr',False)

dateset = pd.read_csv('data/expData.csv',header = 0, index_col = 0)
dateset_columns = dateset.columns
values = dateset.values
#print(dateset)

scaler = MinMaxScaler(feature_range=(0,1))
#print(scaler)
scaled = scaler.fit_transform(values)
#print(values)
#print(type(values))
reframed = seriesToSupervisedLearning(scaled, dateset_columns, 5, 1)
#print(reframed)
reframed.drop(reframed.columns[[20, 21, 22]], axis = 1, inplace = True)
#print(reframed)

values = reframed.values
#print(values)
n_train = 8645
train = values[:8645, :]
test = values[8645:, :]
#print(train)
train_X, train_Y = train[:, :-1], train[:, -1]
test_X, test_Y = test[:, :-1], test[:, -1]

#print(train_X.shape[0])
#print(train_X.shape[1])
#print(train_X)

#转化为LSTM的数据格式：
train_X = train_X.reshape((train_X.shape[0], 5, 4))
test_X = test_X.reshape((test_X.shape[0], 5, 4))
#print(train_X)
#print(test_Y)
