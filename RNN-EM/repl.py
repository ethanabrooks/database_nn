import numpy as np
import theano
import theano.tensor as T

x = T.constant([1, 2])
y = T.constant([[2, 3, 4],
                [3, 4, 5]])

dot = T.dot(x, y)
# grad = T.grad(dot, x)
argmax = T.argmax(y, axis=1)
f = theano.function([], argmax)

print(f())
