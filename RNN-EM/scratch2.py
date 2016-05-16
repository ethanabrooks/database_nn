import numpy as np
import theano
import theano.tensor as T
from collections import namedtuple


x = np.arange(3)
print(x==2).sum()
# for whatever in f():
#     print('-' * 10)
#     print(whatever)
#
# losses = T.nnet.binary_crossentropy(y_pred, y_true)
