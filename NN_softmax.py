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
    # https://stackoverflow.com/questions/23128401/overflow-error-in-neural-networks-implementation
    # np.clip(inputs, -500, 500)
    return (1 + math.e ** (-bias - inputs)) ** (-1)

def relu(bias, inputs):
    return max(0, bias + inputs)

def sig_derived(sig):
    return sig * (1 - sig)

def relu_derived(relu):
    if relu > 0:
        return 1
    elif relu < 0:
        return 0
    else:
        return None 

class Neuron():
    def __init__(self, activation_func = 'sigmoid', bias = None):
        self.b = bias if bias else (np.random.rand(1) * np.sqrt(2))[0]
        # self.weights will have one weight for each neuron in the next layer.
        self.weights = []
        self.act = activation_func
        self.value = 0 # activation function result specifically (sigmoid, linear, or relu of the linear combo of the input)
                       # self.output is what is going to the next layer (already multiplied by weights)

    # Weights are initialized and randomized using Xavier initialization
    def initialize_weights(self, size):
        self.weights = np.random.rand(size) * np.sqrt(2/size)

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

    def derivative(self):
        if self.act == 'linear':
            return self.value
        elif self.act == 'sigmoid':
            return sig_derived(self.value)
        elif self.act == 'ReLU':
            return relu_derived(self.value)

    # Used for summarizing a neuron
    def summary(self, msg):
        try:
            print('\tNeuron' + str(msg) + ', Inputs ' + str(self.inputs) + ', Bias ' + str(self.b))
            print('\t\tWeights ' + str(self.weights) +  ', Value ' + str(self.value))
            print('\t\tOutput ' + str(self.output))
        except:
            print('\tNeuron' + str(msg) + ', Bias ' + str(self.b))
            print('\t\tWeights ' + str(self.weights))

class Layer():
    def __init__(self, num_neurons, num_weights, activation_func):
        # num_neurons is number of neurons in the current layer
        self.size = num_neurons
        # Initialize each neuron with its weights, where the number of weights should be the number of neurons
        # in the next layer
        self.neurons = np.array([Neuron(activation_func) for i in range(self.size)])
        for i in range(self.size):
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
    def __init__(self, num_input_nodes, activation_func, gamma = 0.01, num_output_nodes = 1, num_hidden_layers = 1, num_hidden_layer_nodes = None):
        print('Vanilla_Network')
        if num_output_nodes < 1 or num_hidden_layers < 1 or num_input_nodes < 1:
            print('Error: number of neurons and layers must be >= 1')
            return 0
        if activation_func not in ['sigmoid', 'ReLU', 'linear']:
            print('Error: activation function must be sigmoid, ReLU, or linear')
            return 0
        if num_hidden_layer_nodes != None:
            self.num_hidden_nodes = num_hidden_layer_nodes
        else:
            self.num_hidden_nodes = num_input_nodes * 2 // 3 + num_output_nodes
        self.act = activation_func
        self.num_nodes = num_input_nodes
        self.num_hidden_layers = num_hidden_layers
        self.num_outputs = num_output_nodes
        self.input = Layer(num_input_nodes, self.num_hidden_nodes, self.act)
        self.hidden = np.array([Layer(self.num_hidden_nodes, self.num_hidden_nodes, self.act) for i in range(num_hidden_layers - 1)])
        self.hidden = np.append(self.hidden, Layer(self.num_hidden_nodes, self.num_outputs, self.act))
        self.g = gamma
        self.out_bias = np.random.rand(self.num_outputs) * np.sqrt(2)
        self.predictions = []
        self.trained = False

    def soft_max(self, values, bias):
        # https://victorzhou.com/blog/softmax/
        # print(values)
        scores = values.ravel() + bias
        # print('soft')
        # print(scores)
        out = (np.exp(scores) / np.sum(np.exp(scores)))
        # out = (np.exp(values) / np.sum(np.exp(values), axis = 0))
        # print(out)
        return out.ravel()

    def soft_max_derivative(self, s):
        # Take the derivative of softmax element w.r.t the each logit which is usually Wi * X
        # input s is softmax value of the original input x. 
        # s.shape = (1, n) 
        # i.e. s = np.array([0.3, 0.7]), x = np.array([0, 1])
        # initialize the 2-D jacobian matrix.
        jacobian_m = np.diag(s)
        for i in range(len(jacobian_m)):
            for j in range(len(jacobian_m)):
                if i == j:
                    jacobian_m[i][j] = s[i] * (1-s[i])
                else: 
                    jacobian_m[i][j] = -s[i]*s[j]
        return jacobian_m

    # Train the model's neurons on each training dataset point
    def train(self, X, y):
        self.trained = True
        self.training_data = X
        self.gtruth = y
        # Initialize the predictions that will be made on the training set during training. Just for storage.
        self.predictions = np.zeros((len(y), self.num_outputs))
        # self.summary()
        # Iterate through the training points.
        for i in range(len(y)):
            yi = np.zeros((self.num_outputs))
            yi[int(y[i])] = 1
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
            for j in range(len(self.hidden) - 1):
                outputs = self.hidden[j].train(primary_outputs)
                primary_outputs = outputs
            final_outputs = self.hidden[-1].train(primary_outputs)
            # The final prediction will be the linear combination (aka sum) of the last neurons' weights. The way
            # this is done will be different based on the shape of the data.
            if self.num_outputs == 1:
                self.predictions[i] = self.soft_max(final_outputs.sum(), self.out_bias)
            else:
                self.predictions[i] = self.soft_max(np.array([final_outputs[:,i].sum() for i in range(self.num_outputs)]), self.out_bias)
            # After doing each training point, backpropagate and update weights.
            self.backpropagate(self.predictions[i], yi)
            if (i - 4) % 50 == 0:
                print('Processed ' + str(i + 1) + ' training points.')
        # self.summary()
        # print(self.predictions)
        print('Finished')
        # return np.corrcoef(self.predictions.ravel(), y.ravel())[0,1]

    def store(self, tps):
        filename = self.act + '_' + str(self.num_nodes) + 'inode_' + str(self.num_hidden_nodes) + 'hnode_' + str(self.num_outputs) + 'onot_'
        filename += str(self.num_hidden_layers) + 'hlayers_' + str(self.g) + 'g_' + str(tps) + 'tps' + '.sav'
        pickle.dump(self, open(filename, 'wb'))

    def backpropagate(self, pred, true):
        # First, easy, update the final bias term(s).
        self.out_bias -= self.g * (pred - true)
        # pred and true should both be np arrays to account for instances where the number of outputs maybe not =1.
        # Errors array: each layer will have an error for each of the weights for each neuron.
        # This was helpful: http://neuralnetworksanddeeplearning.com/chap2.html
        # This is what the code is based on: https://blog.yani.ai/backpropagation/
        errors_input = np.ones((1, self.num_nodes, self.num_hidden_nodes))
        errors_hidden = np.ones((self.num_hidden_layers, self.num_hidden_nodes, self.num_hidden_nodes))
        # Calculate the error for the output layer. Each neuron in this layer has self.num_outputs number of weights.
        # For the output layer, the error (derivative) is always (y_pred - y_true) * neuron_value. See slides from lecture.
        for i in range(len(self.hidden[-1].neurons)):
            current_layer = self.hidden[-1]
            self.update_bias(current_layer, i, (pred - true).sum())
            for j in range(self.num_outputs):
                # Have .sum() here because pred and true are arrays
                errors_hidden[self.num_hidden_layers - 1][i][j] = (pred[j] - true[j]) * self.soft_max_derivative(pred)[j][j]
                # Update the output layer's weights
                self.update_weight(current_layer, i, j, errors_hidden[self.num_hidden_layers - 1][i][j])
        # Now we can calculate the error for all of the other hidden layers. Iterate through each of the hidden layers,
        # starting with the last one before the output hidden layer. For each neuron, update each of the weights. Now,
        # Each error will be the error of the neuron after it (errors[l][i][j]) multiplied by the derivative of its own
        # value, which is the sigmoid function (see the sig_derived() function). Based on my hand calculations of this
        # derivative, the value of the next hidden layer's neuron should be removed from the error (this is what the)
        # division part is doing.
        for l in range(len(self.hidden) - 1, 0, - 1):
            current_layer = self.hidden[l - 1]
            if l == len(self.hidden) - 1 and self.num_outputs == 1:
                for i in range(len(current_layer.neurons)):
                    for j in range(len(current_layer.neurons[i].weights)):
                        errors_hidden[l - 1][i][j] = self.hidden[l].neurons[j].derivative() * errors_hidden[l][i][j] * self.hidden[l].neurons[j].weights[0]
                        # Update the output layer's weights
                        self.update_weight(current_layer, i, j, errors_hidden[l - 1][i][j])
                    self.update_bias(current_layer, i, (pred - true).sum() * (errors_hidden[l][i] * current_layer.neurons[i].weights).sum())
            else:
                for i in range(len(current_layer.neurons)):
                    for j in range(len(current_layer.neurons[i].weights)):
                        errors_hidden[l - 1][i][j] = self.hidden[l].neurons[j].derivative()
                        errors_hidden[l - 1][i][j] *= (errors_hidden[l][j] * current_layer.neurons[i].weights).sum()
                        # Update the output layer's weights
                        self.update_weight(current_layer, i, j, errors_hidden[l - 1][i][j])
                    self.update_bias(current_layer, i, (pred - true).sum() * (errors_hidden[l][i] * current_layer.neurons[i].weights).sum())
        # The same is done for the input layer.
        current_layer = self.input
        if len(self.hidden) == 1 and self.num_outputs == 1:
            for i in range(len(current_layer.neurons)):
                for j in range(len(current_layer.neurons[i].weights)):
                    errors_input[0][i][j] = self.hidden[0].neurons[j].derivative() * errors_hidden[0][j][0] * self.hidden[0].neurons[j].weights[0]
                    # Update the output layer's weights
                    self.update_weight(current_layer, i, j, errors_input[0][i][j])
        else:
            for i in range(self.num_nodes):
                for j in range(len(self.input.neurons[i].weights)):
                    errors_input[0][i][j] = self.hidden[0].neurons[j].derivative()
                    errors_input[0][i][j] *= (errors_hidden[0][j] * current_layer.neurons[i].weights).sum()
                    # Update the output layer's weights
                    self.update_weight(current_layer, i, j, errors_input[0][i][j])
        self.errors = [errors_hidden, errors_input]

    # Helper function to update the jth weight of the ith neuron in the given layer with the value determined by
    # the backpropagation function. Learning rate g is established upon the creation of the neural network.
    def update_weight(self, layer, src, dest, val):
        layer.neurons[src].weights[dest] -= self.g * val * layer.neurons[src].value

    def update_bias(self, layer, src, val):
        layer.neurons[src].b -= self.g * val * layer.neurons[src].derivative()

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
        self.output = np.empty((len(x), self.num_outputs))
        # Iterate through samples
        for i in range(len(x)):
            # Do the same steps as you would for training the neural network, just don't include backpropagation at
            # the end.
            primary_outputs = self.input.train(x[i]) # inital outputs of each neuron
                                # for the training point based on the activation function
            outputs = np.empty((self.num_hidden_nodes, self.num_hidden_nodes))
            for j in range(len(self.hidden) - 1):
                outputs = self.hidden[j].train(primary_outputs)
                primary_outputs = outputs
            final_outputs = self.hidden[-1].train(primary_outputs)
            if self.num_outputs == 1:
                self.output[i] = self.soft_max(final_outputs.sum(), self.out_bias)
            else:
                self.output[i] = self.soft_max(np.array([final_outputs[:,i].sum() for i in range(self.num_outputs)]), self.out_bias)
            if i % 50 == 0:
                print('Processed ' + str(i + 1) + ' testing points.')
        # if len(y_true) > 0:
        #     self.accuracy = 1-(np.array([self.output[i].argmax() for i in range(len(x))])-y_true).sum()/len(x)
        #     print(self.accuracy)
        return self.output

    def acc(self, y_pred, y_true):
        return (y_pred - y_true) ** 2

    # Summarize entire neural network
    def summary(self):
        print('\nSUMMARY')
        print('Number of Inputs: ' + str(self.num_nodes) + '\nActivation Function of Hidden Layers: ' + str(self.act))
        try:
            print('Number of Samples: ' + str(len(self.gtruth)))
        except:
            print('Not yet trained')
        print('Learning Rate: ' + str(self.g) + '\nNumber of Output Nodes per Sample: ' + str(self.num_outputs))
        print('Number of Hidden Layers: ' + str(self.num_hidden_layers))
        print('[' + str(self.num_nodes) + ']' + '--> [' + str(self.num_hidden_nodes) + '] x ' + str(self.num_hidden_layers) + '--> [' + str(self.num_outputs) + ']')
    
    def print_layers(self):
        print(self.input.summary('INPUT'))
        for i in range(self.num_hidden_layers):
            print(self.hidden[i].summary('HIDDEN'))

y=np.array([[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[1, 0],[1, 0],[1, 0],[1, 0],[1, 0],[1, 0]])
X_4 = np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[9,3,2,1],[9,3,2,1],[8,5,2,1],[9,3,8,7],[9,3,8,7],[8,5,8,7]])
X = np.array([[1,2,1,1],[1,2,1,1],[1,1,3,1],[2,2,1,1],[1,1,1,1],[1,2,1,1],[-10,-10,-10,-10],[-9,-9,-9,-9],[-9,-9,-7,-7],[-9,-9,-9,-7],[-10,-10,-9,-9],[-10,-10,-9,-9]])
clf = Vanilla_Network(4, 'sigmoid', num_output_nodes=2, gamma = 0.5)
# sys.exit()
# X = np.array([[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[9,3],[9,3],[8,5],[9,3],[9,3],[8,5]])
# y = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.4,0.4,0.4,0.4,0.4,0.4])
# X_4 = np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[9,3,2,1],[9,3,2,1],[8,5,2,1],[9,3,8,7],[9,3,8,7],[8,5,8,7]])
# # y_2 = np.array([[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.85,0.4],[0.85,0.4],[0.85,0.4],[0.85,0.4],[0.85,0.4],[0.85,0.4]])
# # model_lin = Vanilla_Network(4, 'linear', num_hidden_layers = 3, num_hidden_layer_nodes = 3)
# model_sig = Vanilla_Network(4, 'sigmoid', num_hidden_layers = 2, num_hidden_layer_nodes = 3)
# print(model_sig.out_bias)
# model_sig.print_layers()
# # model_lin.train(X_4, y)
# print(model_sig.train(X_4, y))
# # model.train(X[5:8], y[5:8])
# # model.train(X, y)
# # print(model.predict(np.array([[1,2],[9,3]])))
# # print(model.predict(np.array([[100,3]])))
# # print(model.predict(np.array([[9,4]])))
# # print(model.predict(np.array([[50,50]])))
# # print(model.predict(np.array([[2,1]])))
# # print(model.predict(np.array([[1,3]])))
# # model = Vanilla_Network(2, 'sigmoid', num_hidden_layers = 1, num_output_nodes = 2)
# # model.train(X, y_2)
# # model.summary()

