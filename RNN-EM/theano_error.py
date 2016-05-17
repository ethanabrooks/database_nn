from __future__ import print_function

import numpy
import theano
from theano import tensor as T
from theano.printing import Print

# i = Print('i')(i)
# x_t = Print('x_t', ["shape"])(x_t)

vector = T.ivector()
offset = 1


result, _ = theano.scan(fn=lambda i: [vector[i:i + offset], i + 1],
                        outputs_info=[None, T.constant(0)],
                        n_steps=vector.size - offset)

test = theano.function([vector], outputs=result)
test(numpy.ones(128, dtype='int32'))  # passes
test(numpy.ones(129, dtype='int32'))  # throws error

# for result in test(matrix):
#     print('-' * 10)
#     print(result.shape)
