import numpy as np
from scipy.io import loadmat
import sys
from mpl_toolkits import mplot3d
from random import uniform
import math
import pickle
# import CNN.py

# Recommended number hidden layers: 817 * 2 // 3 + 1 = 546 OR < 817 // 2 = 408

evCA = np.array(loadmat('evCA.mat')['evCA']).ravel()
def create_y(training_size, num_outputs):
    if num_outputs == 1:
        y_1 = np.empty((167 * training_size, 1))
        for i in range(167):
            y_1[i] = np.ones((training_size, 1)) * i
        y_1 = y_1.ravel()
    elif num_outputs == 2:
        y_2 = np.empty((167 * training_size, num_outputs))
        y_temp = np.zeros((167, num_outputs))
        c = 0
        for i in range(0,2):
            for j in range(1,4):
                y_temp[c] = np.array([i,j])
                c += 1
        for i in range(2,33):
            for j in range(0,5):
                y_temp[c] = np.array([i,j])
                c += 1
        for i in range(33,35):
            for j in range(1,4):
                y_temp[c] = np.array([i,j])
                c += 1
        for i in range(167 * training_size):
            y_2[i] = y_temp[i % 167]

y_2 = y_2.ravel()

# Train with 5 hidden layers, full dataset (100 training points), different activation functions
# training_size = 100
# X_train = np.empty((167 * training_size, 817))
# c = 0
# for i in range(len(evCA)):
#     if len(evCA[i]) > 1:
#         temp = evCA[i][:,0:training_size].transpose()
#         for j in range(training_size):
#             X_train[c] =temp[j]
#             c += 1
# reg_sig = Vanilla_Network(817, 'sigmoid', num_hidden_layers = 5)
# reg_sig.train(X_train, y_1)
# pickle.dump(reg_sig, open('sig_817_5_1.sav', 'wb'))
# reg_lin = Vanilla_Network(817, 'linear', num_hidden_layers = 5)
# reg_lin.train(X_train, y_1)
# pickle.dump(reg_lin, open('lin_817_5_1.sav', 'wb'))
# reg_relu = Vanilla_Network(817, 'ReLU', num_hidden_layers = 5)
# reg_relu.train(X_train, y_1)
# pickle.dump(reg_relu, open('relu_817_5_1.sav', 'wb'))

# Train with 5 hidden layers, full dataset (50 training points), different activation functions, different gammas
# training_size = 50
# for i in [0.003, 0.005, 0.01, 0.05, 0.1, 0.5, 0.9]:
    # reg_sig = Vanilla_Network(817, 'sigmoid', gamma = c, num_hidden_layers = 5)
    # pickle.dump(reg_sig, open('sig_817_5_1_g' + str(c) + '.sav', 'wb'))
    # reg_lin = Vanilla_Network(817, 'linear', gamma = c, num_hidden_layers = 5)
    # pickle.dump(reg_lin, open('lin_817_5_1_g' + str(c) + '.sav', 'wb'))
    # reg_relu = Vanilla_Network(817, 'ReLU', gamma = c, num_hidden_layers = 5)
    # pickle.dump(reg_relu, open('relu_817_5_1_g' + str(c) + '.sav', 'wb'))

# Select a subset of the 817 time points


# Select a subset of the electrodes (maybe everything down the middle, every 5 electrodes: [2,0],[2,4],[2,9],...)


# Number outputs is 2


# Fewer neurons in hidden layers


# Alter bias??
