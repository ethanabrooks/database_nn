import numpy as np
import theano
import theano.tensor as T

x = T.constant([1, 2])
y = T.constant([2, 3, 4])[:2]

dot = T.dot(x, y)
grad = T.grad(dot, x)
f = theano.function([], grad)


print(f())
