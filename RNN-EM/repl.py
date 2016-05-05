import numpy as np
import theano

x = np.array([1, 1, 1, 0], dtype='int32')
y = theano.tensor.ivector("y")
count = theano.tensor.extra_ops.bincount(y, weights=None, minlength=None, assert_nonneg=True)

res = 1.0 / (count[y] + 1) * theano.tensor.neq(y, 0)

function = theano.function([y], res)
res2 = theano.tensor.neq(y, 1)
f = theano.function([y], res2)

print(function(x))
print(f(x))
