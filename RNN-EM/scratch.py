import numpy as np
import theano
import theano.tensor as T
from collections import namedtuple

from theano.ifelse import ifelse

a = T.constant(np.array([[0, 2, 4, 0],
                         [0, 0, 0, np.nan]], dtype='float32'))
b = T.switch(T.isnan(a), 1.0, 0.0)
c = a + np.inf
padright = T.shape_padright(b.sum(axis=1))
b /= padright
for whatever in theano.function([], [a / c])():
    print('-' * 20)
    print whatever

"""
3D x 1D: innermost dimmension
3D x 2D: innermost dimmension
"""
