from __future__ import print_function
import os
import sys

import numpy as np
import theano
import theano.tensor as T
from collections import namedtuple
from theano.ifelse import ifelse

y = 999999
x = T.iscalar()
f = theano.function([x], [T.exp(x)])
print(f(y))


