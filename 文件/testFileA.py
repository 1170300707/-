from keras import Sequential
from keras.layers import LSTM, Dense
from keras.models import load_model
from data_transformA import scaler, train_X, train_Y, test_X, test_Y
import matplotlib.pyplot as plt
from numpy import concatenate
from math import sqrt
import pandas as pd
from sklearn.metrics import mean_squared_error
import types


model = load_model('model/expModelA.h5')

yHat = model.predict(test_X)
#print(yHat)
test_X = test_X.reshape(test_X.shape[0],20)
#print(test_X)
inv_yHat = concatenate((yHat, test_X[:, 17:]), axis=1)
#print(inv_yHat)
#print(type(inv_yHat))
#print(scaler)
inv_yHat = scaler.inverse_transform(inv_yHat)
inv_yHat = inv_yHat[:, 0]

test_Y = test_Y.reshape((len(test_Y), 1))
inv_Y = concatenate((test_Y, test_X[:, 17:]), axis=1)
inv_Y = scaler.inverse_transform(inv_Y)
inv_Y = inv_Y[:, 0]

data1 = pd.DataFrame(inv_yHat)
data2 = pd.DataFrame(inv_Y)
data1.to_csv('data/testResultA.csv')
data2.to_csv('data/testSetA.csv')

plt.plot(inv_yHat,color='r')
plt.plot(inv_Y)
plt.show()

rmse = sqrt(mean_squared_error(inv_yHat,inv_Y))
print('Test RMSE: %.3f' % rmse)