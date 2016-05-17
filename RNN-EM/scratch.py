from __future__ import print_function
import os
import sys

import numpy as np
import theano
import theano.tensor as T
from collections import namedtuple
from theano.ifelse import ifelse

print(np.random.randint(0, high=3, size=(2,3), dtype='int32'))

