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

def relu_derived(relu):
    return 

class Neuron():
    def __init__(self, activation_func = 'sigmoid', bias = 1):
        self.b = bias
        # self.weights will have one weight for each neuron in the next layer.
        self.weights = []
        self.act = activation_func
        self.value = 0 # activation function result specifically (sigmoid, linear, or relu of the linear combo of the input)
                       # self.output is what is going to the next layer (already multiplied by weights)

    # Weights are initialized and randomized over a uniform distribution
    def initialize_weights(self, size):
        self.weights = np.array([uniform(-2,2) for i in range(size)])

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
                self.value = relu(self.b, self.inputs.sum())
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

    # Used for summarizing a neuron
    def summary(self, msg):
        try:
            print('\tNeuron' + str(msg) + ', Inputs ' + str(self.inputs))
            print('\t\tWeights ' + str(self.weights) +  ', Value ' + str(self.value))
            print('\t\tOutput ' + str(self.output))
        except:
            print('\tNeuron' + str(msg))
            print('\t\tWeights ' + str(self.weights))

class Layer():
    def __init__(self, num_neurons, num_weights, activation_func):
        # num_neurons is number of neurons in the current layer
        self.size = num_neurons
        # Initialize each neuron with its weights, where the number of weights should be the number of neurons
        # in the next layer
        self.neurons = np.array([Neuron(activation_func) for i in range(num_neurons)])
        for i in range(num_neurons):
            self.neurons[i].initialize_weights(num_weights)

    # This will be called when the entire neural network is being trained. Each layer will store its input training
    # points. If the number of inputs is the same as the number of neurons, then you are in the input layer. Otherwise,
    # you are in one of the hidden layers and slightly different outputs will be produced.
    # The output produced by training the neurons will be an m x n array where m = the number of neurons in the layer
    # and n = the number of weights each neuron has. This means that self.outputs[i][j] will be the jth weight of the
    # ith neuron in the given layer; self.outputs[i][j] will go from neuron i in layer (z-1) to neuron j in layer z.
    def train(self, X):
        self.input = X
        if self.input.shape == self.neurons.shape:
            # This is the input layer. The shape of X should be (1,).
            self.outputs = np.array([self.neurons[i].activation(np.array([X[i]])) for i in range(self.size)])
        else:
            # This is a hidden or output layer. The shape of X should be (num_prev_layer_neurons,num_weights_of_prev_layer_neurons).
            # If m = number of neurons in previous layer, then X[:,i] is a vector of the weights from all m of those
            # neurons that are going to the ith neuron in the current layer. This is transposed to keep things consistent.
            self.outputs = np.array([self.neurons[i].activation(X[:,i].transpose()) for i in range(self.size)])
        return self.outputs

    # Used for summarizing a layer
    def summary(self, msg):
        print(msg)
        c = 0
        for neuron in self.neurons:
            neuron.summary(c)
            c += 1
        print('\n')

class Vanilla_Network():
    def __init__(self, num_input_nodes, activation_func, gamma = 0.003, num_output_nodes = 1, num_hidden_layers = 30, num_hidden_layer_nodes = None):
        print('Vanilla_Network')
        if num_hidden_layer_nodes != None:
            self.num_hidden_nodes = num_hidden_layer_nodes
        else:
            self.num_hidden_nodes = num_input_nodes * 2 // 3
        if num_output_nodes < 1 or num_hidden_layers < 1 or num_input_nodes < 1:
            print('Error: number of neurons and layers must be >= 1')
            return 0
        if activation_func not in ['sigmoid', 'ReLU', 'linear']:
            print('Error: activation function must be sigmoid, ReLU, or linear')
            return 0
        self.act = activation_func
        self.num_nodes = num_input_nodes
        self.num_hidden_layers = num_hidden_layers
        self.num_outputs = num_output_nodes
        self.input = Layer(num_input_nodes, self.num_hidden_nodes, self.act)
        self.hidden = np.array([Layer(self.num_hidden_nodes, self.num_hidden_nodes, self.act) for i in range(num_hidden_layers - 1)])
        self.hidden = np.append(self.hidden, Layer(self.num_hidden_nodes, self.num_outputs, 'linear'))
        self.num_outputs = num_output_nodes
        self.g = gamma
        self.predictions = []

    # Train the model's neurons on each training dataset point
    def train(self, X, y):
        # Initialize the predictions that will be made on the training set during training. Just for storage.
        self.predictions = np.zeros((len(y), self.num_outputs))
        self.summary()
        # Iterate through the training points.
        for i in range(len(y)):
            # Inital outputs of each neuron for the training point based on the activation function. X should be
            # of shape (self.num_nodes,). primary_outputs will be of shape (self.num_nodes,self.num_hidden_nodes).
            # So primary_outputs[i][j] will be the jth weight of the ith neuron in the input layer.
            primary_outputs = self.input.train(X[i])
            # This is a temporary variable for storage. It will have a slightly different shape than primary_outputs
            # because after the input layer, the number of neurons and the number of weights will be different.
            outputs = np.empty((self.num_hidden_nodes, self.num_hidden_nodes))
            # Once you have gotten the outputs of the input layer (primary_outputs variable), iterate through each
            # hidden layer and determine the outputs of those. Now, primary_outputs will be updated with the new
            # weights for the following hidden layers.
            for j in range(len(self.hidden)):
                outputs = self.hidden[j].train(primary_outputs)
                primary_outputs = outputs
            # The final prediction will be the linear combination (aka sum) of the last neurons' weights. The way
            # this is done will be different based on the shape of the data.
            if self.num_outputs == 1:
                self.predictions[i] = np.array([primary_outputs.sum()])
            else:
                self.predictions[i] = np.array([primary_outputs[:,i].sum() for i in range(self.num_outputs)])
            # After doing each training point, backpropagate and update weights.
            self.backpropagate(self.predictions[i], y[i])
            print('Processed ' + str(i + 1) + ' training points.')
        self.summary()
        # print(self.predictions)
        return np.corrcoef(self.predictions.ravel(), y.ravel())[0,1]

    def backpropagate(self, pred, true):
        # pred and true should both be np arrays to account for instances where the number of outputs maybe not =1.
        # Errors array: each layer will have an error for each of the weights for each neuron.
        # This was helpful: http://neuralnetworksanddeeplearning.com/chap2.html
        # This is what the code is based on: https://blog.yani.ai/backpropagation/
        errors_input = np.ones((1, self.num_nodes, self.num_hidden_nodes))
        errors_hidden = np.ones((self.num_hidden_layers, self.num_hidden_nodes, self.num_hidden_nodes))
        errors = np.ones((self.num_hidden_layers + 1, self.num_nodes, self.num_nodes))
        # Calculate the error for the output layer. Each neuron in this layer has self.num_outputs number of weights.
        # For the output layer, the error (derivative) is always (y_pred - y_true) * neuron_value. See slides from lecture.
        for i in range(len(self.hidden[-1].neurons)):
            current_layer = self.hidden[-1]
            for j in range(self.num_outputs):
                # Have .sum() here because pred and true are arrays
                errors_hidden[self.num_hidden_layers - 1][i][j] = (pred - true).sum()
                # Update the output layer's weights
                self.update_weight(current_layer, i, j, errors_hidden[self.num_hidden_layers - 1][i][j])
        # For sigmoid activation function
        if self.act == 'sigmoid':
            # Now we can calculate the error for all of the other hidden layers. Iterate through each of the hidden layers,
            # starting with the last one before the output hidden layer. For each neuron, update each of the weights. Now,
            # Each error will be the error of the neuron after it (errors[l][i][j]) multiplied by the derivative of its own
            # value, which is the sigmoid function (see the sig_derived() function). Based on my hand calculations of this
            # derivative, the value of the next hidden layer's neuron should be removed from the error (this is what the)
            # division part is doing.
            for l in range(len(self.hidden) - 1, 0, - 1):
                current_layer = self.hidden[l - 1]
                if l == len(self.hidden) - 1:
                    for i in range(len(current_layer.neurons)):
                        for j in range(len(current_layer.neurons[i].weights)):
                            errors_hidden[l - 1][i][j] = sig_derived(self.hidden[l].neurons[j].value) * errors_hidden[l][i][j]
                            # Update the output layer's weights
                            self.update_weight(current_layer, i, j, errors_hidden[l - 1][i][j])
                else:
                    for i in range(len(current_layer.neurons)):
                        for j in range(len(current_layer.neurons[i].weights)):
                            errors_hidden[l - 1][i][j] = sig_derived(self.hidden[l].neurons[j].value)
                            errors_hidden[l - 1][i][j] *= (errors_hidden[l][j] * current_layer.neurons[i].weights).sum()
                            # Update the output layer's weights
                            self.update_weight(current_layer, i, j, errors_hidden[l - 1][i][j])
            # The same is done for the input layer.
            current_layer = self.input
            if len(self.hidden) == 1:
                for i in range(len(current_layer.neurons)):
                    for j in range(len(current_layer.neurons[i].weights)):
                        print(current_layer.neurons[i].weights)
                        print(errors_hidden.shape)
                        errors_input[0][i][j] = sig_derived(self.hidden[0].neurons[j].value) * errors_hidden[0][j][0]
                        # Update the output layer's weights
                        self.update_weight(current_layer, i, j, errors_input[0][i][j])
            else:
                for i in range(self.num_nodes):
                    for j in range(len(self.input.neurons[i].weights)):
                        errors_input[0][i][j] = sig_derived(self.hidden[0].neurons[j].value)
                        errors_input[0][i][j] *= (errors_hidden[0][j] * current_layer.neurons[i].weights).sum()
                        # Update the output layer's weights
                        self.update_weight(current_layer, i, j, errors_input[0][i][j])
        elif self.act == 'linear':
            # Now we can calculate the error for all of the other hidden layers. Iterate through each of the hidden layers,
            # starting with the last one before the output hidden layer. For each neuron, update each of the weights. Now,
            # Each error will be the error of the neuron after it (errors[l][i][j]) multiplied by the derivative of its own
            # value, which will just be its value.
            for l in range(len(self.hidden) - 1, 0, - 1):
                current_layer = self.hidden[l - 1]
                for i in range(len(current_layer.neurons)):
                    for j in range(len(current_layer.neurons[i].weights)):
                        errors[l][i][j] = current_layer.neurons[i].value * errors[l][i][j]
                        # Update the output layer's weights
                        self.update_weight(current_layer, i, j, errors[l][i][j])
            # The same is done for the input layer.
            for i in range(self.num_nodes):
                current_layer = self.input
                for j in range(len(self.input.neurons[i].weights)):
                    errors[0][i][j] = errors[1][i][j] * self.input.neurons[i].value * self.input.neurons[i].inputs[0]
                    self.update_weight(current_layer, i, j, errors[0][i][j])
        elif self.act == 'ReLU':
            pass

    # Helper function to update the jth weight of the ith neuron in the given layer with the value determined by
    # the backpropagation function. Learning rate g is established upon the creation of the neural network.
    def update_weight(self, layer, src, dest, val):
        layer.neurons[src].weights[dest] -= self.g * val * layer.neurons[src].value

    # Function to predict the classification of a given x. x must be a numpy array, even if you only want
    # to predict the class of a single sample. If you are doing predictions to determine how well your
    # accuracy is and you have some ground truth data for x, then you should set y_true equal to that
    # ground truth data, which will result in your neural network having an "accuracy" attribute which is
    # the correlation of your predictions with the truth. If you are predicting new samples, with no
    # ground truth, then just exlude y_true entiring from the function call.
    def predict(self, x, y_true = np.array([])):
        if len(x[0]) != self.num_nodes:
            print('Input for predict must be an array of arrays of the same size as inputs: [[],[]]')
            return 0
        # Initialize the output to be the same size as the number of samples you want to predict.
        output = np.empty((len(x), self.num_outputs))
        # Iterate through samples
        for i in range(len(x)):
            # Do the same steps as you would for training the neural network, just don't include backpropagation at
            # the end.
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
        # Get the accuracy if you have ground truth data y_true.
        if len(y_true) > 0 and self.num_outputs == 1:
            self.accuracy = np.corrcoef(output.ravel(), y_true.ravel())
        # elif len(y_true) > 0:
        #     self.accuracy = np.corrcoef()
        return output

    def acc(self, y_pred, y_test):
        return (y_pred - y_test) ** 2

    # Summarize entire neural network
    def summary(self):
        print('\n\nSUMMARY')
        print('Number of Inputs: ' + str(self.num_nodes) + '\nActivation Function of Hidden Layers: ' + str(self.act))
        print('Learning Rate: ' + str(self.g) + '\nNumber of Output Nodes per Sample: ' + str(self.num_outputs))
        print('Number of Hidden Layers: ' + str(self.num_hidden_layers))
        print('[' + str(self.num_nodes) + ']' + '--> [' + str(self.num_nodes) + '] x ' + str(self.num_hidden_layers) + '--> [' + str(self.num_outputs) + ']')
        print(self.input.summary('INPUT'))
        for i in range(self.num_hidden_layers):
            print(self.hidden[i].summary('HIDDEN'))

X = np.array([[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[9,3],[9,3],[8,5],[9,3],[9,3],[8,5]])
y = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.4,0.4,0.4,0.4,0.4,0.4])
X_4 = np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[9,3,2,1],[9,3,2,1],[8,5,2,1],[9,3,8,7],[9,3,8,7],[8,5,8,7]])
y_2 = np.array([[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.85,0.4],[0.85,0.4],[0.85,0.4],[0.85,0.4],[0.85,0.4],[0.85,0.4]])
# model_lin = Vanilla_Network(4, 'linear', num_hidden_layers = 3, num_hidden_layer_nodes = 3)
model_sig = Vanilla_Network(4, 'sigmoid', num_hidden_layers = 2, num_hidden_layer_nodes = 3)
# model_lin.train(X_4, y)
model_sig.train(X_4, y)
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
sys.exit()

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
    pickle.dump(reg_relu, open('relu_272_5_1.sav', 'wb'))

