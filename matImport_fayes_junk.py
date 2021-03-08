#!/usr/bin/env python
# coding: utf-8

# In[5]:


from scipy.io import loadmat
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
import numpy as np
import sys
import h5py


# In[21]:


avgMat = loadmat('avgMat.mat')
avgMatData = [[element for element in upperElement] for upperElement in avgMat['avgMat']]
#Data is stored as a list of lists of lists (35x5x817)
# print(len(avgMatData))
# print(len(avgMatData[1]))
# print(len(avgMatData[1][1]))

# def plot_raw_data(df, sample):
#     out = []
#     for i in range(5):
#         out.append([])
#     for i in range(35):
#         for j in range(5):
#             out[j].append(df[j][i][sample])
#     plt.matshow(out)
#     # Plot
#     plt.colorbar()
#     plt.title('heat' + str(sample))
#     plt.savefig('step' + str(sample) + '.png')
#     plt.close()
# df = pd.DataFrame(avgMatData)
# print(df)
# plot_raw_data(df, 25)
# plot_raw_data(df, 125)
# plot_raw_data(df, 225)
# plot_raw_data(df, 325)
# plot_raw_data(df, 425)
# plot_raw_data(df, 525)
# plot_raw_data(df, 625)
# plot_raw_data(df, 725)
# plot_raw_data(df, 800)
# sys.exit()

def loss_lin_reg(X, y, b):
    sum_val = 0
    for i in range(len(y)):
        sum_val += (y[i] - b[0] - b[1] * X[i])**2
    return 0.5 * sum_val
def grad_lin_reg(X, y, b):
    grad = np.zeros(len(b))
    for deg in range(len(b)):
        for i in range(len(y)):
            grad[deg] += (y[i] - b[0] - b[1] * X[i])* (X[i] ** deg)
    return -grad
learning_rate = 0.003
X = np.array(avgMatData).ravel()
X = X[np.logical_not(np.isnan(X))]
total_sample_number = len(avgMatData[0][0])
b = np.ones(total_sample_number)
train_size = total_sample_number
y = np.empty((167,train_size))
for i in range(167):
    y[i] = np.ones(train_size) * i
y = y.ravel()

loss1 = loss_lin_reg(X, y, b)
for i in range(40):
    b += learning_rate * grad_lin_reg(X, y, b)
loss2 = loss_lin_reg(X, y, b)

# y_test = np.empty((167, total_sample_number))
# for i in range(167):
#     y_test[i] = np.ones(total_sample_number) * i
# y_test = y_test.ravel()
# loss_test = loss_lin_reg(X, y_test, b)

plt.scatter(X, y)
plt.plot(np.arange(170), b[0] + b[1] * np.arange(170), 'r')
plt.xlim(-0.002,0.00025)
plt.ylim(-5,170)
plt.title('after')
plt.show()
sys.exit()

sys.exit()


# In[54]:

evCA = np.array(loadmat('evCA.mat')['evCA']).ravel()
evCA = evCA[np.logical_not(np.isnan(evCA))]
y_train = np.empty((167, 300))
loss1 = loss_lin_reg(evCA, y_train, b)
for i in range(40):
    b += learning_rate * grad_lin_reg(evCA, y_train, b)
loss2 = loss_lin_reg(evCA, y_train, b)

y_test = np.empty((167, total_sample_number))
for i in range(167):
    y_test[i] = np.ones(total_sample_number) * i
y_test = y_test.ravel()
loss_test = loss_lin_reg(evCA, y_test, b)


sys.exit()

# Pretty sure evMat is what Michael used to make evCA
evMat = loadmat('evMat.mat')
evMatData = [[element for element in upperElement] for upperElement in evMat['evMat']]

V = evMatData[0][0][0]
t = evMatData[0][0][1]
evLog = evMatData[0][0][2]
#t is a numpy array of 817 values (corresponding to those in V)
#V is a numpy array of numpy arrays of numpy arrays of numpy arrays representing a 4D vector (5 x 8 x 817 x 500)
#evLog is... uhhh... not too sure tbh, its a mess. If need be I'll parse it but I don't think it really matters for us.


# In[105]:


ex1 = loadmat('ex1.mat')
ex1Data = [[element for element in upperElement] for upperElement in ex1['ex1']]
#First 5 data files raw data stored as a list of:
#'filename': String vector
#'rawData': list of lists containing V, t, and log - see ev for info on how to parse this, is essentially the same.
#'processedData': Empty vector
#'referenceTrace': List of lists containing V, t, and log - parsing is essentially the same as ev
#'eventTimeSample': List of one list containing the times(unconverted)
#'eventTime': List of one list containing the times(converted)
#'samplingFrequency' list of one value with the sampling frequency in Hz

ex2 = loadmat('ex2.mat')
ex2Data = [[element for element in upperElement] for upperElement in ex2['ex2']]
#Last 5 data files raw data stored in the same way as ex1


# In[106]:


print(ex2Data[0][1])


# In[103]:


def parseEx(mat,fileNum,info):
    if info == 'filename':
        return mat[0][fileNum][0]
    elif info == 'rawData':
        V = mat[0][fileNum][1][0][0][0]
        t = mat[0][fileNum][1][0][0][1]
        log = mat[0][fileNum][1][0][0][2]
        return V,t,log
    elif info == 'processedData':
        return mat[0][fileNum][2]
    elif info == 'referenceTrace':
        V = mat[0][fileNum][1][0][0][0]
        t = mat[0][fileNum][1][0][0][1]
        log = mat[0][fileNum][1][0][0][2]
        return V,t,log
    elif info == 'eventTimeSample':
        return mat[0][fileNum][4]
    elif info == 'eventTime':
        return mat[0][fileNum][5]
    elif info == 'samplingFrequency':
        return mat[0][fileNum][6]
    
#Function for parsing Ex file. May be useful when we need to start referencing these variables several times.


# In[107]:


#Example of using this function:
#  I'll use the parseEx function to get the V,t,log data for the reference trace for the second file in the ex2Data
#  (7th file total)
(V,t,log) = parseEx(ex2Data,2,'referenceTrace')
print(t)

