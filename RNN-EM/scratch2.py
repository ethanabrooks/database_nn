import numpy as np
import theano
import theano.tensor as T
from collections import namedtuple


def sliding_window(matrix, window_radius):
    window_diameter = 1 + window_radius * 2
    window_idxs = np.fromfunction(lambda i, j: i + j,
                                  (matrix.shape[1], window_diameter),
                                  dtype='int32')
    padded = np.pad(matrix,
                    pad_width=[(0, 0), (window_radius, window_radius)],
                    mode='constant')
    return np.swapaxes(padded.T[window_idxs], 1, 2)

print(sliding_window(np.arange(20).reshape((5,4)), 2))

# x = np.arange(9).reshape((3,3))
# y = np.array([[0, 1], [1, 2]])
# print(x[y])