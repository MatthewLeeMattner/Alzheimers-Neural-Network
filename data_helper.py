import numpy as np


def get_dummy_patches():
    x_train = np.random.randn(10000, 5, 5, 5)
    x_test = np.random.randn(10000, 5, 5, 5)
    x_val = np.random.randn(10000, 5, 5, 5)
    return x_train, x_test, x_val
