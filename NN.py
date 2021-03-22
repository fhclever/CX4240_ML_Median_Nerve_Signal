import numpy as np
from scipy.io import loadmat
import sys
from mpl_toolkits import mplot3d
from random import uniform
import math
import pickle

def linear_combo(weights, inputs):
    return weights * inputs

def sig_act_func(bias, inputs):
    return (1 + math.e ** (-bias - inputs)) ** (-1)

def relu(bias, inputs):
    return max(0, bias + inputs)

def sig_derived(sig):
    return sig * (1 - sig)

class Neuron():
    def __init__(self, activation_func = 'sigmoid', bias = 1):
        self.b = bias
        self.weights = []
        self.act = activation_func
        self.value = 0 # activation function result specifically;
                       # self.output is what is going to the next layer (already multiplied by weights)

    def initialize_weights(self, size):
        self.weights = np.array([uniform(-2,2) for i in range(size)])
        # self.weights = np.ones((size))

    # Each neuron should have an output which is the activation function of the input value
    # and the weights. Each weight should have a corresponding output value that will go to its
    # corresponding next neuron.
    # AKA len(self.weights) == len(self.output)
    def activation(self, train):
        self.inputs = train
        if self.act == 'sigmoid':
            if len(self.inputs) == 1:
                self.value = self.inputs[0]
                self.output = np.array([linear_combo(self.weights[i], self.inputs[0]) 
                    for i in range(len(self.weights))])
            else:
                self.value = sig_act_func(self.b, self.inputs.sum())
                self.output = np.array([linear_combo(self.weights[i], self.value) 
                    for i in range(len(self.weights))])
            return self.output
        elif self.act == 'ReLU':
            if len(self.inputs) == 1:
                self.value = self.inputs[0]
                self.output = np.array([linear_combo(self.weights[i], self.inputs[0]) 
                    for i in range(len(self.weights))])
            else:
                self.value = sig_act_func(self.b, self.inputs.sum())
                self.output = np.array([linear_combo(self.weights[i], self.value) 
                    for i in range(len(self.weights))])
            return self.output
        elif self.act == 'linear':
            if len(self.inputs) == 1:
                self.value = self.inputs[0]
                self.output = np.array([linear_combo(self.weights[i], self.inputs[0]) 
                    for i in range(len(self.weights))])
            else:
                self.value = self.b + self.inputs.sum()
                self.output = np.array([linear_combo(self.weights[i], self.value) 
                    for i in range(len(self.weights))])
            return self.output

    def calc_error_output(self, true):
        self.derivative = (self.value - true)
        return self.derivative

    def summary(self, msg):
        try:
            print('\tNeuron' + str(msg) + ', Inputs ' + str(self.inputs))
            print('\t\tWeights ' + str(self.weights) +  ', Value ' + str(self.value))
            print('\t\tOutput ' + str(self.output))
        except:
            print('\tNeuron' + str(msg))
            print('\t\tWeights ' + str(self.weights))

class Layer():
    def __init__(self, num_neurons, num_weights, activation_func = 'sigmoid'):
        print('Layer')
        self.size = num_neurons
        self.neurons = np.array([Neuron() for i in range(num_neurons)])
        for i in range(num_neurons):
            self.neurons[i].initialize_weights(num_weights)

    def train(self, X):
        self.input = X
        if self.input.shape == self.neurons.shape:
            self.outputs = np.array([self.neurons[i].activation(np.array([X[i]])) for i in range(self.size)])
        else:
            self.outputs = np.array([self.neurons[i].activation(X[:,i].transpose()) for i in range(self.size)])
        return self.outputs

    def summary(self, msg):
        print(msg)
        c = 0
        for neuron in self.neurons:
            neuron.summary(c)
            c += 1
        print('\n')

class Vanilla_Network():
    def __init__(self, num_input_nodes, activation_func, gamma = 0.003, num_output_nodes = 1, num_hidden_layers = 30):
        print('Vanilla_Network')
        if num_output_nodes < 1 or num_hidden_layers < 1 or num_input_nodes < 1:
            print('Error: number of neurons and layers must be >= 1')
            return 0
        if activation_func not in ['sigmoid', 'ReLU', 'linear']:
            print('Error: activation function must be sigmoid, ReLU, or linear')
            return 0
        self.act = activation_func
        self.num_nodes = num_input_nodes
        self.num_hidden_layers = num_hidden_layers
        self.input = Layer(num_input_nodes, num_input_nodes, activation_func)
        self.hidden = np.array([Layer(num_input_nodes, num_input_nodes) for i in range(num_hidden_layers - 1)])
        self.hidden = np.append(self.hidden, Layer(num_input_nodes, num_output_nodes))
        self.num_outputs = num_output_nodes
        self.g = gamma
        self.predictions = []

    # Train the model's neurons on each training dataset point
    def train(self, X, y):
        self.predictions = np.zeros((len(y), self.num_outputs))
        c = 0
        for i in range(len(y)):
            primary_outputs = self.input.train(X[i]) # inital outputs of each neuron
                                # for the training point based on the activation function
            outputs = np.empty((self.num_nodes, self.num_nodes))
            for j in range(len(self.hidden)):
                outputs = self.hidden[j].train(primary_outputs)
                primary_outputs = outputs
            if self.num_outputs == 1:
                self.predictions[i] = np.array([primary_outputs.sum()])
            else:
                self.predictions[i] = np.array([primary_outputs[:,i].sum() for i in range(self.num_outputs)])
            self.backpropagate(self.predictions[i], y[i])
            c += 1
            print('Processed ' + str(c) + ' training points.')
        print(self.predictions)
        # return np.corrcoef(self.predictions, y)[0,1]

    def backpropagate(self, pred, true):
        errors = np.ones((len(self.hidden) + 1, self.num_nodes, self.num_nodes))
        # print(errors)
        for i in range(len(self.hidden[-1].neurons)):
            current_layer = self.hidden[-1]
            for j in range(self.num_outputs):
                errors[errors.shape[0] - 1][i][j] = (pred - true).sum() * current_layer.neurons[i].value
                self.update_weight(current_layer, i, j, errors[errors.shape[0] - 1][i][j])
        # print(errors)
        for i in range(len(self.hidden) - 1, 0, - 1):
            current_layer = self.hidden[i - 1]
            for j in range(len(current_layer.neurons)):
                for k in range(len(current_layer.neurons[j].weights)):
                    errors[i][j][k] = sig_derived(current_layer.neurons[j].value) * errors[i][j][k]
                    if  self.hidden[i].neurons[j].value != 0:
                        errors[i][j][k] /= self.hidden[i].neurons[j].value
                    self.update_weight(current_layer, j, k, errors[i][j][k])
        # print(errors)
        for i in range(self.num_nodes):
            current_layer = self.input
            for j in range(len(self.input.neurons[i].weights)):
                errors[0][i][j] = errors[1][i][j] * sig_derived(self.input.neurons[i].value) * self.input.neurons[i].inputs[0]
                if self.hidden[0].neurons[j].value != 0:
                    errors[0][i][j] /= self.hidden[0].neurons[j].value
                self.update_weight(current_layer, i, j, errors[0][i][j])
        # print(errors)
        self.errors = errors

    def update_weight(self, layer, src, dest, val):
        layer.neurons[src].weights[dest] -= self.g * val

    def predict(self, x):
        if len(x[0]) != self.num_nodes:
            print('Input for predict must be an array of arrays of the same size as inputs: [[],[]]')
            return 0
        output = np.empty((len(x), self.num_outputs))
        for i in range(len(x)):
            primary_outputs = self.input.train(x[i]) # inital outputs of each neuron
                                # for the training point based on the activation function
            outputs = np.empty((self.num_nodes, self.num_nodes))
            for j in range(len(self.hidden)):
                outputs = self.hidden[j].train(primary_outputs)
                primary_outputs = outputs
            if self.num_outputs == 1:
                output[i] = np.array([linear_combo(1, primary_outputs.sum()) for i in range(self.num_outputs)])
            else:
                output[i] = np.array([linear_combo(1, primary_outputs[:,i].sum()) for i in range(self.num_outputs)])
        return output

    def summary(self):
        print('\n\nSUMMARY')
        print('Number of Inputs: ' + str(self.num_nodes) + '\nActivation Function of Hidden Layers: ' + str(self.act))
        print('Learning Rate: ' + str(self.g) + '\nNumber of Output Nodes per Sample: ' + str(self.num_outputs))
        print('Number of Hidden Layers: ' + str(self.num_hidden_layers))
        print('[' + str(self.num_nodes) + ']' + '--> [' + str(self.num_nodes) + ' x ' + str(self.num_hidden_layers) + '--> [' + str(self.num_outputs) + ']')

# X = np.array([[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[9,3],[9,3],[8,5],[9,3],[9,3],[8,5]])
# y = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.4,0.4,0.4,0.4,0.4,0.4])
# y_2 = np.array([[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.85,0.4],[0.85,0.4],[0.85,0.4],[0.85,0.4],[0.85,0.4],[0.85,0.4]])
# model = Vanilla_Network(2, 'linear', num_hidden_layers = 3)
# model = Vanilla_Network(2, 'sigmoid', num_hidden_layers = 3)
# model.train(X[5:8], y[5:8])
# model.train(X, y)
# print(model.predict(np.array([[1,2],[9,3]])))
# print(model.predict(np.array([[100,3]])))
# print(model.predict(np.array([[9,4]])))
# print(model.predict(np.array([[50,50]])))
# print(model.predict(np.array([[2,1]])))
# print(model.predict(np.array([[1,3]])))
# model = Vanilla_Network(2, 'sigmoid', num_hidden_layers = 1, num_output_nodes = 2)
# model.train(X, y_2)
# model.summary()
# sys.exit()

# Create data matrix: will have 175 positions. Positions 0,4,5,9,165,169,170,174 are empty (corners).
# All other positions will have 817 rows and 500, 1000, or 1500 columns. Each position represents
# an electrode on the patch.
evCA = np.array(loadmat('evCA.mat')['evCA']).ravel()
training_size = 100
num_outputs = 1
y = np.empty((167, training_size, num_outputs))
for i in range(167):
    y[i] = np.ones((training_size, num_outputs)) * i
y = y.ravel()
print(y)
time_features = 200
# reg = Vanilla_Network(817, 'sigmoid', num_hidden_layers = 2)
# pickle.dump(reg, open('sig_817_2_1.sav', 'wb'))
# reg = Vanilla_Network(817, 'linear', num_hidden_layers = 2)
# pickle.dump(reg, open('lin_817_2_1.sav', 'wb'))
# reg_sig = Vanilla_Network(817, 'sigmoid', num_hidden_layers = 5, num_output_nodes = 2)
# pickle.dump(reg_sig, open('sig_817_5_2.sav', 'wb'))
# reg_lin = Vanilla_Network(817, 'linear', num_hidden_layers = 5, num_output_nodes = 2)
# pickle.dump(reg_lin, open('lin_817_5_2.sav', 'wb'))
reg_sig = Vanilla_Network(817, 'sigmoid', gamma = 0.05, num_hidden_layers = 2)
pickle.dump(reg_sig, open('sig_817_2_1.sav', 'wb'))
reg_lin = Vanilla_Network(817, 'linear', gamma = 0.05, num_hidden_layers = 2)
pickle.dump(reg_lin, open('lin_817_2_1.sav', 'wb'))
# reg_sig = Vanilla_Network(817, 'sigmoid', num_hidden_layers = 5)
# pickle.dump(reg_sig, open('sig_817_5_1.sav', 'wb'))
# reg_lin = Vanilla_Network(817, 'linear', num_hidden_layers = 5)
# pickle.dump(reg_lin, open('lin_817_5_1.sav', 'wb'))
# reg_relu = Vanilla_Network(817, 'ReLU', num_hidden_layers = 5)
# pickle.dump(reg_relu, open('relu_817_5_1.sav', 'wb'))
X_train = np.empty((167 * training_size, 817))
c = 0
for i in range(len(evCA)):
    if len(evCA[i]) > 1:
        # 200 training points
        temp = evCA[i][:,0:training_size].transpose()
        for j in range(training_size):
            X_train[c] =temp[j]
            c += 1
accuracy = reg_sig.train(X_train, y)
accuracy = reg_lin.train(X_train, y)
# reg_relu.train(X_train, y)

X_train_reduced_dim = X_train[:,0::3]
if len(X_train_reduced_dim[0]) == 817 // 3:
    reg_relu = Vanilla_Network(817 // 3)
    reg_relu.train(X_train_reduced_dim, y)
    pickle.dump(reg_relu, 'relu_272_5_1')

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class Net(nn.Module):

#     def __init__(self):
#         super(Net, self).__init__()
#         # 1 input image channel, 6 output channels, 3x3 square convolution
#         # kernel
#         self.conv1 = nn.Conv2d(1, 6, 3)
#         self.conv2 = nn.Conv2d(6, 16, 3)
#         # an affine operation: y = Wx + b
#         self.fc1 = nn.Linear(16 * 5 * 35, 120)  # 6*6 from image dimension
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         # Max pooling over a (2, 2) window
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         # If the size is a square you can only specify a single number
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         x = x.view(-1, self.num_flat_features(x))
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

#     def num_flat_features(self, x):
#         size = x.size()[1:]  # all dimensions except the batch dimension
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features


# net = Net()
# print(net)

# X = np.array([[1,2,3,4],[2,3,5,1],[3,4,5,4],[9,3,2,1],[8,5,3,0]])
# y = np.array([1,1,2,4,4])

# out = net()
