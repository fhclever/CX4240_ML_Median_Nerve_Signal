import numpy as np
import pandas as pd
from scipy.io import loadmat
import sys
from mpl_toolkits import mplot3d
from random import uniform
import math
import pickle
import NN_softmax as NN

# Recommended number hidden layer neurons: 817 * 2 // 3 + 1 = 546 OR < 817 // 2 = 408

evCA = np.array(loadmat('evCA.mat')['evCA']).ravel()
def create_y_rows(training_size):
    y = np.empty((167 * training_size))
    for i in range(0, 3 * training_size):
        y[i] = 0
    for i in range(3 * training_size, 3 * training_size * 2):
        y[i] = 1
    c = 2
    for i in range(3 * training_size * 2, training_size * 161):
        y[i] = c
        if (i - (training_size - 1)) % (training_size * 5) == 0:
            c += 1
    for i in range(training_size * 161, training_size * 164):
        y[i] = 33
    for i in range(training_size * 164, training_size * 167):
        y[i] = 34
    return y
def create_y(training_size, num_outputs):
    if num_outputs == 1:
        y_1 = np.empty((167, training_size, 1))
        for i in range(167):
            y_1[i] = np.ones((training_size, 1)) * i
        y_1 = y_1.ravel()
        return y_1
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
        d = 0
        for i in range(167 * training_size):
            y_2[i] = y_temp[d]
            if (i + 1) % training_size == 0:
                d += 1
        return y_2
def create_X(training_size):
    testing_size = 200 - training_size
    X_train = np.empty((167 * training_size, 817))
    X_test = np.empty((167 * testing_size, 817))
    c = 0
    d = 0
    for i in range(len(evCA)):
        if len(evCA[i]) > 1:
            temp1 = evCA[i][:,0:training_size].transpose()
            temp2 = evCA[i][:,training_size:200].transpose()
            for j in range(training_size):
                X_train[c] =temp1[j]
                c += 1
            for j in range(testing_size):
                X_test[d] = temp2[j]
                d += 1
    return X_train, X_test

# Train with 5 hidden layers, full dataset (100 training points), different activation functions
# training_size = 100
# X_train_100 = np.empty((167 * training_size, 817))
# c = 0
# for i in range(len(evCA)):
#     if len(evCA[i]) > 1:
#         temp = evCA[i][:,0:training_size].transpose()
#         for j in range(training_size):
#             X_train_100[c] =temp[j]
#             c += 1
# reg_sig = NN.Vanilla_Network(817, 'sigmoid', num_hidden_layers = 5)
# reg_sig.train(X_train_100, y_1)
# pickle.dump(reg_sig, open('sig_817_5_1_100samples.sav', 'wb'))
# reg_lin = NN.Vanilla_Network(817, 'linear', num_hidden_layers = 5)
# reg_lin.train(X_train_100, y_1)
# pickle.dump(reg_lin, open('lin_817_5_1_100samples.sav', 'wb'))
# reg_relu = NN.Vanilla_Network(817, 'ReLU', num_hidden_layers = 5)
# reg_relu.train(X_train_100, y_1)
# pickle.dump(reg_relu, open('relu_817_5_1_100samples.sav', 'wb'))

# Train with 2 hidden layers, full dataset (50 training points), different activation functions
# training_size = 50
# X_train_50 = np.empty((167 * training_size, 817))
# c = 0
# for i in range(len(evCA)):
#     if len(evCA[i]) > 1:
#         temp = evCA[i][:,0:training_size].transpose()
#         for j in range(training_size):
#             X_train_50[c] =temp[j]
#             c += 1
# reg_sig = NN.Vanilla_Network(817, 'sigmoid', num_hidden_layers = 2)
# reg_sig.train(X_train_50, y_1)
# pickle.dump(reg_sig, open('sig_817_2_1.sav', 'wb'))
# reg_lin = NN.Vanilla_Network(817, 'linear', num_hidden_layers = 2)
# reg_lin.train(X_train_50, y_1)
# pickle.dump(reg_lin, open('lin_817_2_1.sav', 'wb'))
# reg_relu = NN.Vanilla_Network(817, 'ReLU', num_hidden_layers = 2)
# reg_relu.train(X_train_50, y_1)
# pickle.dump(reg_relu, open('relu_817_2_1.sav', 'wb'))

# Train with 5 hidden layers, full dataset (50 training points), different activation functions, different gammas
# y_1 = create_y(training_size, 1)
# data = ['gamma','num_hidden_layers', 'tps', 'num_inputs', 'num_hidden_layer_nodes', 'predictions_during_training', 'training_correl',
#         'training_predictions', 'testing_predictions', 'loss1', 'loss2', 'clf', 'num_labels']
# df = pd.DataFrame(columns = data)
# # Different gammas
# for i in [0.9]:
#     # Different number of hidden layers
#     for j in range(1,3):
#         # Different size of training data
#         for m in [2,10]:
#             temp = create_X(m)
#             X = temp[0]
#             testing = temp[1]
#             y_rows = create_y_rows(m)
#             y_1 = create_y(m, 1)
#             y_test = create_y(200 - m, 1)
#             # Different number of input features
#             for k in range(2, 4):
#                 X_train_reduced_dim = X[:,0::k]
#                 X_test_reduced_dim = testing[:,0::k]
#                 num_features = X_train_reduced_dim.shape[1]
#                 # Different number of hidden layer neurons
#                 for l in [num_features * 2 // 3 + 35, num_features // 2, num_features // 4]:
#                     print(i, j, m, k, l, 35)
#                     reg_sig_rows = NN.Vanilla_Network(num_features, 'sigmoid', gamma = i, num_hidden_layers = j, num_hidden_layer_nodes = l, num_output_nodes = 35)
#                     prelim = reg_sig_rows.predict(X_train_reduced_dim, y_rows)
#                     loss1 = reg_sig_rows.acc(np.array([prelim[i].argmax() for i in range(m*167)]), y_rows).sum()
#                     reg_sig_rows.store(m * 167)
#                     correl = reg_sig_rows.train(X_train_reduced_dim, y_rows)
#                     train_preds = reg_sig_rows.predict(X_train_reduced_dim, y_rows)
#                     loss2 = reg_sig_rows.acc(np.array([train_preds[i].argmax() for i in range(len(train_preds))]), y_rows).sum()
#                     # if loss2 - loss1 < 0:
#                     test_preds = reg_sig_rows.predict(X_test_reduced_dim, y_test)
#                     df.loc[len(df.index)] = [i, j, m, k, l, reg_sig_rows.predictions, correl, train_preds, test_preds, loss1, loss2, reg_sig_rows, 35]
                    # print(i, j, m, k, l, 167)
                    # reg_sig = NN.Vanilla_Network(num_features, 'sigmoid', gamma = i, num_hidden_layers = j, num_hidden_layer_nodes = l, num_output_nodes = 167)
                    # prelim = reg_sig.predict(X_train_reduced_dim, y_1)
                    # loss1 = reg_sig.acc(np.array([prelim[i].argmax() for i in range(m*167)]), y_1).sum()
                    # reg_sig.store(m * 167)
                    # correl = reg_sig.train(X_train_reduced_dim, y_1)
                    # train_preds = reg_sig.predict(X_train_reduced_dim, y_1)
                    # loss2 = reg_sig.acc(np.array([train_preds[i].argmax() for i in range(len(train_preds))]), y_1).sum()
                    # if loss2 - loss1 < 0:
                    #     print('neg loss')
                    #     test_preds = reg_sig.predict(X_test_reduced_dim, y_test)
                    #     df.loc[len(df.index)] = [i, j, m, k, l, reg_sig.predictions, correl, train_preds, test_preds, loss1, loss2, reg_sig, 167]

# pickle.dump(df, open('sigmoid_NNs.sav','wb'))
# # Different gammas
# for i in [0.003, 0.005, 0.01, 0.05, 0.1, 0.5, 0.9]:
#     # Different number of hidden layers
#     for j in range(1, 6):
#         # Different size of training data
#         for m in [2,5,7,10,20,50,75,100]:
#             X = create_X(m)
#             y_1 = create_y(m, 1)
#             print(X.shape)
#             # Different number of input features
#             for k in range(1, 4):
#                 X_train_reduced_dim = X[:,0::k]
#                 print(X_train_reduced_dim.shape)
#                 num_features = X_train_reduced_dim.shape[1]
#                 # Different number of hidden layer neurons
#                 for l in [None, num_features * 2 // 3 + 1, num_features // 2, num_features // 4, 2]:
#                     reg_lin = NN.Vanilla_Network(num_features, 'linear', gamma = i, num_hidden_layers = j, num_hidden_layer_nodes = l)
#                     reg_lin.train(X_train_reduced_dim, y_1)
# # Different gammas
# for i in [0.003, 0.005, 0.01, 0.05, 0.1, 0.5, 0.9]:
#     # Different number of hidden layers
#     for j in range(1, 6):
#         # Different size of training data
#         for m in [2,5,7,10,20,50,75,100]:
#             X = create_X(m)
#             y_1 = create_y(m, 1)
#             print(X.shape)
#             # Different number of input features
#             for k in range(1, 4):
#                 X_train_reduced_dim = X[:,0::k]
#                 print(X_train_reduced_dim.shape)
#                 num_features = X_train_reduced_dim.shape[1]
#                 # Different number of hidden layer neurons
#                 for l in [None, num_features * 2 // 3 + 1, num_features // 2, num_features // 4, 2]:
#                     reg_relu = NN.Vanilla_Network(num_features, 'ReLU', gamma = i, num_hidden_layers = j, num_hidden_layer_nodes = l)
#                     reg_relu.train(X_train_reduced_dim, y_1)