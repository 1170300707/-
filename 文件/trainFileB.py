from keras import Sequential
from keras.layers import LSTM, Dense,Activation,Dropout
from data_transformB import scaler, train_X, train_Y, test_X, test_Y
import matplotlib.pyplot as plt
from numpy import concatenate
from math import sqrt
from sklearn.metrics import mean_squared_error

model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
History = model.fit(train_X,train_Y,epochs=100,batch_size=64,validation_data=(test_X,test_Y))
model.save("model/expModelB.h5")

plt.plot(History.history['loss'],label = 'train')
plt.plot(History.history['val_loss'], label = 'test')
plt.legend()
plt.show()