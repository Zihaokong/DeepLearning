################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Ajit Kumar, Savyasachi
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

        #对位相乘
        return grad * delta

    def sigmoid(self, x):
        self.x = x
        return 1/(1+np.exp(-1*x))

    def tanh(self, x):
        """
        Implement tanh here.
        """
        self.x = x
        return 1.7159*np.tanh((2/3)*x)

    def ReLU(self, x):
        """
        Implement ReLU here.
        """
        self.x = x
        result = x
        result[x<0] = 0
        return result

    def grad_sigmoid(self):
        """
        Compute the gradient for sigmoid here.
        """
        return self.sigmoid(self.x)*(1-self.sigmoid(self.x))

    def grad_tanh(self):
        """
        Compute the gradient for tanh here.
        """
        return 1-self.tanh(self.x)*self.tanh(self.x)

    def grad_ReLU(self):
        """
        Compute the gradient for ReLU here.
        """
        result = self.x
        result[self.x<=0] = 0
        result[self.x>0] = 1
        return result


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
        self.w = math.sqrt(2 / in_units) * np.random.randn(in_units,
                                                           out_units)  # You can experiment with initialization.
        self.b = np.zeros((1, out_units))  # Create a placeholder for Bias
        self.x = None  # Save the input to forward in this
        self.a = None  # Save the output of forward pass in this (without activation)

        self.d_x = None  # Save the gradient w.r.t x in this // wjk
        self.d_w = None  # Save the gradient w.r.t w in this // xj
        self.d_b = None  # Save the gradient w.r.t b in this //

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
        self.d_w = self.x
        self.d_b = np.ones((1,self.b.shape[1]))
        self.d_x = self.w @ delta.T

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

        layer1 = self.layers[0]
        act1 = self.layers[1]
        layer2 = self.layers[2]

        aj = layer1.forward(self.x)
        zj = act1.forward(aj)
        ak = layer2.forward(zj)
        self.y = self.softmax(ak)

        if targets is not None:
            batch_loss = self.loss(self.y,self.targets)
            return self.y,batch_loss
        else:
            return self.y


    def softmax(self, x):
        """
        Implement the softmax function here.
        Remember to take care of the overflow condition.
        """
        row_max = np.amax(x, axis=1)

        # prevent from value getting too big, substract every row by max.
        x = x - row_max.reshape(x.shape[0], 1)
        ex = np.exp(x)
        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)

    def backward(self):
        """
        Implement backpropagation here.
        Call backward methods of individual layer's.
        """
        delta = self.targets - self.y
        self.layers[2].backward(delta)
        raise NotImplementedError("Backprop not implemented for NeuralNetwork")


    def loss(self, logits, targets):
        """
        compute the categorical cross-entropy loss and return it.
        """
        y_ylog = targets * np.log(logits + 0.000000000001)
        return -1 * np.sum(y_ylog)

