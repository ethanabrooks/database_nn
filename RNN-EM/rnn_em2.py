from __future__ import print_function

from functools import partial

import lasagne
import numpy
import os
import theano
from theano.ifelse import ifelse
from theano.printing import Print
from theano import tensor as T

# i = Print('i')(i)
# x_t = Print('x_t', ["shape"])(x_t)

matrix = T.imatrix()
offset = 1


res, _ = theano.scan(fn=lambda i: [(matrix[:, i:i + offset]), i + 1],
                     outputs_info=[None, T.constant(0)],
                     n_steps=matrix.shape[1] - offset)

test = theano.function([matrix], outputs=res)
test(numpy.ones((1, 128), dtype='int32'))  # passes
test(numpy.ones((1, 129), dtype='int32'))  # throws error

# for result in test(matrix):
#     print('-' * 10)
#     print(result.shape)
