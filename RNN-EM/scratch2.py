import numpy as np
import theano
import theano.tensor as T
from collections import namedtuple

word_per_instance = 2
window_size = 4
nclasses = 3

y_pred = T.constant(np.random.randint(0, 3, (word_per_instance,
                                             window_size, nclasses)))
y_true = T.constant(np.random.randint(0, 3, (window_size, word_per_instance)))

y_pred_flatten = y_pred.dimshuffle(2, 1, 0).flatten(ndim=2).T

f = theano.function([], [y_pred, y_pred_flatten, y_true, y_true.ravel()])
for whatever in f():
    print('-' * 10)
    print(whatever)

losses = T.nnet.binary_crossentropy(y_pred, y_true)
