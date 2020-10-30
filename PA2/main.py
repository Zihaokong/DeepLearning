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

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./config.yaml")

    # Load the data and reshape the data
    x_train, y_train, x_test, y_test = load_data()

    x_train = np.array([image.reshape((1024)) for image in x_train],dtype='float')
    x_test = np.array([image.reshape((1024)) for image in x_test], dtype='float')

    # Create validation set out of training data.
    num = int(x_train.shape[0]*0.8)
    [x_train, x_val]= np.split(x_train,[num])
    [y_train, y_val] = np.split(y_train, [num])
    y_train = y_train.reshape(-1,1)
    y_val = y_val.reshape(-1, 1)

    # Any pre-processing on the datasets goes here.


    # Calculate feature mean and standard deviation for x_train, and use them to
    # Z score x_train, X_val and X_test
    def z_score_train_test(train,val,test):
        train_T = train.T
        val_T = val.T
        test_T = test.T
        for i in range(train_T.shape[0]):
            mean = np.mean(train_T[i])
            SD = np.std(train_T[i])
            train_T[i] = (train_T[i] - mean)/SD
            val_T[i] = (val_T[i] - mean) / SD
            test_T[i] = (test_T[i] - mean) / SD
        return train_T.T, val_T.T,test_T.T

    # Z-scoring
    x_train, x_val,x_test = z_score_train_test(x_train,x_val,x_test)

    # train the model
    # train_acc, valid_acc, train_loss, valid_loss, best_model = \
    #     train(x_train, y_train, x_val, y_val, config)
    #
    # test_loss, test_acc = test(best_model, x_test, y_test)
    #
    # print("Config: %r" % config)
    # print("Test Loss", test_loss)
    # print("Test Accuracy", test_acc)
    #
    # # DO NOT modify the code below.
    # data = {'train_loss': train_loss, 'val_loss': valid_loss, 'train_acc': train_acc, 'val_acc': valid_acc,
    #         'best_model': best_model, 'test_loss': test_loss, 'test_acc': test_acc}
    #
    # write_to_file('./results.pkl', data)
