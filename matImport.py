#!/usr/bin/env python
# coding: utf-8

# In[5]:


from scipy.io import loadmat
import pandas as pd
import numpy as np
import h5py


# In[21]:


avgMat = loadmat('avgMat.mat')
avgMatData = [[element for element in upperElement] for upperElement in avgMat['avgMat']]
#Data is stored as a list of lists of lists (35x5x817)
print(len(avgMatData))
print(len(avgMatData[1]))
print(len(avgMatData[1][1]))


# In[54]:


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

