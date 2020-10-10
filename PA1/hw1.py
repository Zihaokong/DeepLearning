import numpy as np
from fashion_mnist_dataset.utils import mnist_reader
from PIL import Image


X_train, y_train = mnist_reader.load_mnist('fashion_mnist_dataset/data/fashion',kind='train')
X_test, y_test = mnist_reader.load_mnist('fashion_mnist_dataset/data/fashion',kind='t10k')

def normalize_axis(vect):
    ma = max(vect)
    mi = min(vect)
    vect = (vect-mi)/(ma-mi)
    return vect




onehot = np.zeros((y_test.size,max(y_test)+1))
#Array of 10000*10
onehot[np.arange(y_test.size),y_test] = 1
#set corresponding position to 1


X_train = np.apply_along_axis(normalize_axis, 0, X_train)
X_test = np.apply_along_axis(normalize_axis, 0, X_test)


