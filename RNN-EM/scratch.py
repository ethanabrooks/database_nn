import numpy as np
import theano
import theano.tensor as T
from collections import namedtuple


def norm(x):
    return T.sqrt(T.sum(T.sqr(x), axis=1))
a = np.array([[0,0,0], [1,1,1]])
# w = T.constant(np.arange(12).reshape(2, 2, 3))
w = T.constant(np.ones((2, 4, 1)))
h = T.constant(np.ones((2, 1, 5)))
z = T.batched_dot(w, h)
for whatever in theano.function([], [w, h, z])():
    print('-'*20)
    print whatever

"""
3D x 1D: innermost dimmension
3D x 2D: innermost dimmension
"""