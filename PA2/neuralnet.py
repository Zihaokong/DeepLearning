################################################################################
# CSE 151B: Programming Assignment 2
# Code snippet by Ajit Kumar, Savyasachi
# Edited by Zihao Kong, Baichuan Wu
# Fall 2020
################################################################################

import numpy as np
import math

class Activation:
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        //>>> sigmoid_layer = Activation("sigmoid")
        //>>> z = sigmoid_layer(a)
        //>>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type="sigmoid"):
        """
        Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU"]:
            raise NotImplementedError("%s is not implemented." % (activation_type))

        # Type of non-linear activation.
        self.activation_type = activation_type
        # Placeholder for input. This will be used for computing gradients.
        self.x = None

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

    def backward(self, delta):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        # Hadamard product
        return grad * delta

    def sigmoid(self, x):
        """
        Sigmoid activation function.
        """
        self.x = x
        if x >= 0:
            return 1 / (1 + exp(-x))
        else:
            return exp(x) / (1 + exp(x))

    def tanh(self, x):
        """
        Tanh activation function.
        """
        self.x = x
        return np.tanh(x)

    def ReLU(self, x):
        """
        ReLU activation function.
        """
        self.x = x
        res = np.array(self.x)
        res[res < 0] = 0
        return res

    def grad_sigmoid(self):
        """
        Calculates Gradient of sigmoid activation function.
        """
        return self.sigmoid(self.x) * (1 - self.sigmoid(self.x))

    def grad_tanh(self):
        """
        Calculates Gradient of tanh activation function.
        """
        return 1 - self.tanh(self.x) ** 2

    def grad_ReLU(self):
        """
        Calculates Gradient of ReLU activation function.
        """
        res = np.array(self.x)
        res[res <= 0] = 0
        res[res > 0] = 1
        return res


class Layer:
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        //>>> fully_connected_layer = Layer(1024, 100)
        //>>> output = fully_connected_layer(input)
        //>>> gradient = fully_connected_layer.backward(delta=1.0)
    """

    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        self.w = math.sqrt(2 / in_units) * np.random.randn(in_units, out_units) # Kaiming initialization
        self.b = np.zeros((1, out_units))  # Create a placeholder for Bias
        self.x = None  # Save the input to forward in this
        self.a = None  # Save the output of forward pass in this (without activation)

        self.d_x = None  # Save the gradient w.r.t x in this w_{jk}
        self.d_w = None  # Save the gradient w.r.t w in this x_j
        self.d_b = None  # Save the gradient w.r.t b in this

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        Compute the forward pass through the layer here.
        Do not apply activation here.
        Return self.a
        """
        self.x = x
        self.a = self.x @ self.w + self.b
        return self.a

    def backward(self, delta):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        # \frac{\partial a_j}{\partial w_{ij}} = x, 
        # where the gradient is - \frac{\partial E}{\partial a_j} \frac{a_j}{\partial w_{ij}}
        self.d_w = self.x

        # derivative of bias is 1
        self.d_b = np.ones((1, len(self.b)))

        # derivative of input is the weighted sum of input of delta j and w_j
        # delta is row major, change to column major first
        self.d_x = (self.w @ delta.T).T

        # propogate partial X to calculate last layer's delta
        return self.d_x


class NeuralNetwork:
    """
    Create a Neural Network specified by the input configuration.

    Example:
        //>>> net = NeuralNetwork(config)
        //>>> output = net(input)
        //>>> net.backward()
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []  # Store all layers in this list.
        self.x = None  # Save the input to forward in this
        self.y = None  # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable
        self.alpha = config['learning_rate'] # Save the learning rate from config
        self.batch_size = config['batch_size'] #

        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        self.x = x
        self.targets = targets
        
        temp = self.x
        
        for i in range(len(self.layers)):
            # Calculate weighted sum of inputs / pass weighted sum through activation
            temp = self.layers[i].forward(temp)

        # activate ak to yk
        self.y = self.softmax(temp)

        # calculate loss if target is passed into the function
        if targets is not None:
            batch_loss = self.loss(self.y, self.targets)
            return self.y, batch_loss
        else:
            return self.y


    def softmax(self, x):
        """
        Numerically stable softmax function
        """
        row_max = np.amax(x, axis=1)
        x = x - row_max.reshape(len(x), 1)

        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)

    def backward(self):
        """
        Implement backpropagation here.
        Call backward methods of individual layer's.
        """
        delta = [None] * len(self.layers)
        delta[len(delta) - 1] = self.targets - self.y

        # Backprop deltas
        for i in reversed(range(len(self.layers) - 1)):
            # Evaluate the delta term
            delta[i] = self.layers[i + 1].backward(delta[i + 1])
        
        # Update weights
        for i in range(0, len(self.layers), 2):
            # w = [fan_in, fan_out] => x = [n, fan_in], delta = [n, fan_out], alpha = [1] => x.T @ delta * alpha
            self.layers[i].w = self.layers[i].w + self.alpha * (self.layers[i].x.T @ delta[i]) / self.batch_size
            # a = w_0 * b + w_1 * x1 ...
            # b = [n, fan_out] => delta = [n, fan_out]
            self.layers[i].b = self.layers[i].b + self.alpha * np.mean(delta[i], axis=0)

    def loss(self, logits, targets):
        """
        compute the categorical cross-entropy loss and return it.
        """
        y_ylog = targets * np.log(logits + 1e-8)
        return -1 * np.sum(y_ylog) / len(targets)

