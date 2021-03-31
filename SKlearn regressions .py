#!/usr/bin/env python
# coding: utf-8

# In[33]:


from scipy.io import loadmat
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
import numpy as np
import sys
import h5py
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error


# In[8]:


##DATA


# In[9]:


data = loadmat("C:\\Users\\Anika\\Downloads\\evCa.mat")
time_data = loadmat("C:\\Users\\Anika\\Downloads\\evMat.mat")


# In[49]:


timeKey=[key for key in time_data.keys()]
full_time=time_data[timeKey[3]]
dataKey=[key for key in data.keys()]
full_data=data[dataKey[3]]


# In[50]:


plt.figure
testx = full_data[1][1]
testy=full_time[0][0][0][1][0]

testx_np = np.array([np.array(xi) for xi in testx])
times=full_time[0][0][2][2][1][0][0][0]

plt.plot(testx,testy)
plt.show()


# In[51]:


##Linear
regressor = LinearRegression()
X_train=np.array([[xi] for xi in testx[0]])
y_train=np.array([[xi] for xi in testy[0]])
regressor.fit(X_train, y_train)

print(regressor.intercept_,regressor.coef_)
X_test=np.array([[xi] for xi in testx[1]])
y_test=np.array([[xi] for xi in testy[1]])
regressor.fit(X_train, y_train)
prediction= regressor.predict(X_test)
print(regressor.intercept_,regressor.coef_)

print( "means squared error:", mean_squared_error(y_test, prediction))


# In[52]:


##Ridge
regressor = linear_model.Ridge(alpha=.5)
X_train=np.array([[xi] for xi in testx[0]])
y_train=np.array([[xi] for xi in testy[0]])
regressor.fit(X_train, y_train)

print(regressor.intercept_,regressor.coef_)
X_test=np.array([[xi] for xi in testx[1]])
y_test=np.array([[xi] for xi in testy[1]])
regressor.fit(X_train, y_train)
prediction= regressor.predict(X_test)
print(regressor.intercept_,regressor.coef_)

print( "means squared error:", mean_squared_error(y_test, prediction))


# In[53]:


# ##Logistic
# regressor = LogisticRegression()
# X_train=np.array([[xi] for xi in testx[0]])
# y_train=np.array([[xi] for xi in testy[0]])
# regressor.fit(X_train, y_train)

# print(regressor.intercept_,regressor.coef_)


# In[54]:


##Ridge
regressor = linear_model.BayesianRidge()
X_train=np.array([[xi] for xi in testx[0]])
y_train=np.array([[xi] for xi in testy[0]])
X_test=np.array([[xi] for xi in testx[1]])
y_test=np.array([[xi] for xi in testy[1]])
regressor.fit(X_train, y_train)
prediction= regressor.predict(X_test)
print(regressor.intercept_,regressor.coef_)

print( "means squared error:", mean_squared_error(y_test, prediction))


# In[40]:


##A few other plots


# In[62]:


plt.figure
testx = full_data[1][1][0]
testy=full_time[0][0][0][1][0][0]

testx_np = np.array([np.array(xi) for xi in testx])
times=full_time[0][0][2][2][1][0][0][0]

plt.plot(testx,testy, color="#f03c54")
plt.show()


# In[61]:


##value vs time, seems to have no correlation 
plt.figure
testx = full_data[1][1][0]
testy=time_data[timeKey[3]][0][0][2][2][1][0][0][0]

testx_np = np.array([np.array(xi) for xi in testx])
times=full_time[0][0][2][2][1][0][0][0]

plt.plot(testy,testx, color="#941b2b")
plt.show()


# In[ ]:




