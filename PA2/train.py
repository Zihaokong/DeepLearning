################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

from neuralnet import *
import copy


def accuracy(y, t):
    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)
    res = [y_hat == t_hat for y_hat, t_hat in zip(y, t)]
    return np.sum(res) / len(res)


def train(x_train, y_train, x_val, y_val, config):
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
    best_loss = float('inf')
    curr_loss = float('inf')
    prev_loss = float("inf")
    early_stop_mark = 0

    batch_index = 0

    model = NeuralNetwork(config=config)

    for i in range(config['epochs']):
        print("start epoch", i)
        batch_index = 0
        # Randomize the order of the indices into the training set
        shuffled_indices = np.random.permutation(len(x_train))
        x_train = x_train[shuffled_indices]
        y_train = y_train[shuffled_indices]
        for j in range(0, len(x_train), config['batch_size']):
            batch_index += 1

            if (j + config['batch_size'] < len(x_train)):
                batch_x = x_train[j:j + config['batch_size'], :]
                batch_y = y_train[j:j + config['batch_size'], :]
            else:
                batch_x = x_train[[j, len(x_train) - 1]]
                batch_y = y_train[[j, len(x_train) - 1]]

            model.forward(x=batch_x, targets=batch_y)
            model.backward()


        y, tr_loss = model.forward(x=x_train, targets=y_train)
        train_loss.append(tr_loss)
        train_acc.append(accuracy(y,y_train))

        y, curr_loss = model.forward(x=x_val, targets=y_val)
        valid_loss.append(curr_loss)
        valid_acc.append(accuracy(y,y_val))
        if curr_loss <= best_loss:
            print("best loss detected:", best_loss)
            print("accuracy: ", accuracy(y, y_val) * 100, "%")

            best_loss = curr_loss
            best_model = copy.deepcopy(model)

        # if early stop is true
        if curr_loss >= prev_loss:
            early_stop_mark += 1

        if early_stop_mark == config['early_stop_epoch']:
            early_stop_mark = 0
            break
        prev_loss = curr_loss
    #print(train_acc,valid_acc)
    # return train_acc, valid_acc, train_loss, valid_loss, best_model
    return train_acc, valid_acc, train_loss, valid_loss, best_model


def test(model, x_test, y_test):
    """
    Does a forward pass on the model and return loss and accuracy on the test set.
    """
    y, loss = model.forward(x_test, y_test)
    accu = accuracy(y, y_test)
    # return loss, accuracy
    return loss, accu
