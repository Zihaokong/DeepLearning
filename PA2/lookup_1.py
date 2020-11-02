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