import numpy as np
def my_func(a):
    ma = max(a)
    mi = min(a)
    a=(a-mi)/(ma-mi)
    return a
b = np.array([[1,2,3], [4,5,6], [7,8,9]])

print(np.apply_along_axis(my_func, 0, b))