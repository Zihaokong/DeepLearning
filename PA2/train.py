################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

from neuralnet import *

def train(x_train, y_train, x_valid, y_valid, config):
    """
    Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    return five things -
        training and validation loss and accuracies - 1D arrays of loss and accuracy values per epoch.
        best model - an instance of class NeuralNetwork. You can use copy.deepcopy(model) to save the best model.
    """
    train_acc = []
    valid_acc = []
    train_loss = []
    valid_loss = []
    best_model = None

    model = NeuralNetwork(config=config)

    # return train_acc, valid_acc, train_loss, valid_loss, best_model
    return train_acc, valid_Acc, train_loss, valid_loss, best_model


def test(model, x_test, y_test):
    """
    Does a forward pass on the model and return loss and accuracy on the test set.
    """
    y, loss = model.forward(x_test, y_test)
    accuracy = 0
    # return loss, accuracy
    return loss, accuracy
