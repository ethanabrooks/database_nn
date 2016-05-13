import numpy as np
import theano
import theano.tensor as T
from collections import namedtuple

def f(a, b):
    return a+b

add1 = T.partial(f, a=1)
add2 = T.partial(f, a=2)
print(add1(3))
print(add2(4))

#
#
# def norm(x):
#     return T.sqrt(T.sum(T.sqr(x), axis=1))
#
#
# a = np.array([[0, 0, 0], [1, 1, 1]])
# b = np.array([[1, 1, 1], [1, 1, 1]])
# counts = T.extra_ops.bincount(a.ravel(), assert_nonneg=True)
# # w = T.constant(np.arange(12).reshape(2, 2, 3))
# z = T.nnet.binary_crossentropy(a, b)
# for whatever in theano.function([], [z, counts])():
#     print('-' * 20)
#     print whatever

"""
3D x 1D: innermost dimmension
3D x 2D: innermost dimmension
"""
