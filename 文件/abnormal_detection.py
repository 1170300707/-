import pandas as pd
from matplotlib import pyplot
import numpy as np
import types

testResultA = pd.read_csv('data/testResultA.csv', header=0, index_col=0)
testSetA = pd.read_csv('data/testSetA.csv', header=0, index_col=0)
testResultB = pd.read_csv('data/testResultB.csv', header=0, index_col=0)
testSetB = pd.read_csv('data/testSetB.csv', header=0, index_col=0)
testResultD = pd.read_csv('data/testResultD.csv', header=0, index_col=0)
testSetD = pd.read_csv('data/testSetD.csv', header=0, index_col=0)

dataRA = testResultA.values
dataSA = testSetA.values
dataRB = testResultB.values
dataSB = testSetB.values
dataRD = testResultD.values
dataSD = testSetD.values

eA = list()
i = 0
while(1):
    eA.append(abs(dataRA[i][0] - dataSA[i][0]))
    i += 1
    if(i >= dataRA.shape[0]):
        break
eB = list()
i = 0
while(1):
    eB.append(abs(dataRB[i][0] - dataSB[i][0]))
    i += 1
    if(i >= dataRB.shape[0]):
        break
eD = list()
i = 0
while(1):
    eD.append(abs(dataRD[i][0] - dataSD[i][0]))
    i += 1
    if(i >= dataRD.shape[0]):
        break


A_mean = np.mean(eA)
A_std = np.std(eA, ddof=1)
B_mean = np.mean(eB)
B_std = np.std(eB, ddof=1)
D_mean = np.mean(eD)
D_std = np.std(eD, ddof=1)

#print(len(eA))
#print(len(eB))
#print(len(eD))
#print(eA)
#print(eB)
#print(eD)
#print(A_mean)
#print(A_std)
#print(B_mean)
#print(B_std)
#print(D_mean)
#print(D_std)

WrongA = np.zeros(len(eA))
WrongB = np.zeros(len(eB))
WrongD = np.zeros(len(eD))

i = 0
num1 = 0
num2 = 0
num3 = 0
for num in eA:
    if(abs(num - A_mean) >= 3*A_std):
        WrongA[i] = 1
        #print("A错误： " + str(i))
        num1 += 1
    i += 1

i = 0
for num in eB:
    if(abs(num - B_mean) >= 3*B_std):
        WrongB[i] = 1
        #print("B错误： " + str(i))
        num2 += 1
    i += 1

i = 0
for num in eD:
    if(abs(num - D_mean) >= 3*D_std):
        WrongD[i] = 1
        #print("E错误： " + str(i))
        num3 += 1
    i += 1

print(num1)
print(num2)
print(num3)
#print(WrongA)

summary = np.zeros(len(eA))
i = 0
while(1):
    if(i >= len(eA)):
        break
    if WrongA[i] == 1 :
        if WrongB[i] == 1 :
            if WrongD[i] == 1 :
                summary[i] = 1;
    i += 1

'''num = 0
for nu in summary:
    if nu == 1:
        num+=1
print(num)'''


pyplot.figure()

pyplot.subplot(3,1,1)
pyplot.plot(dataRA,color='r')
pyplot.plot(dataSA)
i = 0
for num in summary:
    if num == 1:
        pyplot.axvline(x = i, ls = "-", c = "green")
    i += 1
pyplot.title('A')

pyplot.subplot(3,1,2)
pyplot.plot(dataRB,color='r')
pyplot.plot(dataSB)
i = 0
for num in summary:
    if num == 1:
        pyplot.axvline(x = i, ls = "-", c = "green")
    i += 1
pyplot.title('B')

pyplot.subplot(3,1,3)
pyplot.plot(dataRD,color='r')
pyplot.plot(dataSD)
i = 0
for num in summary:
    if num == 1:
        pyplot.axvline(x = i, ls = "-", c = "green")
    i += 1
pyplot.title('D')

pyplot.show()