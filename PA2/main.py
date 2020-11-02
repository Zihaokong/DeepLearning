################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################
# We've provided you with the dataset in PA2.zip
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

from utils import load_data, load_config, write_to_file, one_hot_encoding
from train import *
from neuralnet import *
import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./config.yaml")

    # Load the data and reshape from (32 x 32) to (1024 x 1)
    x_train, y_train, x_test, y_test = load_data()

    # One-hot encoding
    y_train = np.eye(len(y_train), 10)[y_train]
    y_test = np.eye(len(y_test), 10)[y_test]

    x_train = np.array([image.reshape((1024)) for image in x_train], dtype='float')
    x_test = np.array([image.reshape((1024)) for image in x_test], dtype='float')

    # Create validation set out of training data.
    num = int(len(x_train) * 0.8)
    [x_train, x_val] = np.split(x_train, [num])
    [y_train, y_val] = np.split(y_train, [num])

    # Any pre-processing on the datasets goes here.


    # Calculate feature mean and standard deviation for x_train, and use them to
    # Z score x_train, X_val and X_test
    # Calculate feature mean and standard deviation for x_train, and use them to
    # Z score x_train, X_val and X_test
    def z_score_train_test(train, val, test):
        train_T = train.T
        val_T = val.T
        test_T = test.T
        for i in range(len(train_T)):
            mean = np.mean(train_T[i])
            SD = np.std(train_T[i])
            train_T[i] = (train_T[i] - mean) / SD
            val_T[i] = (val_T[i] - mean) / SD
            test_T[i] = (test_T[i] - mean) / SD
        return train_T.T, val_T.T, test_T.T


    # Z-scoring
    x_train, x_val, x_test = z_score_train_test(x_train, x_val, x_test)
    #train the model
    train_acc, valid_acc, train_loss, valid_loss, best_model = \
    train(x_train, y_train, x_val, y_val, config)

    test_loss, test_acc = test(best_model, x_test, y_test)

    print("Config: %r" % config)
    print("Test Loss", test_loss)
    print("Test Accuracy", test_acc)

    # DO NOT modify the code below.
    data = {'train_loss': train_loss, 'val_loss': valid_loss, 'train_acc': train_acc, 'val_acc': valid_acc,
            'best_model': best_model, 'test_loss': test_loss, 'test_acc': test_acc}

    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.show()
    write_to_file('./results.pkl', data)
