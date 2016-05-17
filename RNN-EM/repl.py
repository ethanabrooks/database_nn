import numpy as np
import theano
import theano.tensor as T
from collections import namedtuple

x = T.constant(np.arange(12).reshape(3, 4), dtype='int32')

i = T.iscalar()

win_radius = 1
win_diameter = win_radius * 2 + 1

pad = T.zeros((x.shape[0], win_radius))
x_extend = T.concatenate([pad, x, pad], axis=1)

print(theano.function([], outputs=x_extend)())
w = T.constant(np.ones((win_diameter, 5)))


def f(i):
    window = x_extend[:, i:i + win_diameter]
    return window, T.dot(window, w), i + 1


[windows, outs, i_], _ = theano.scan(fn=f,
                                     outputs_info=[None, None, T.constant(0)],
                                     n_steps=x.shape[1])

y_pred = T.argmax(outs, axis=2).T

def get_loss():
    pass

losses = T.nnet.binary_crossentropy(y_pred, x)

for u in theano.function([], outputs=[windows, outs, y_pred, losses])():
    print('-------')
    print u
