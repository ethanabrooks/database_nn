from __future__ import print_function
import os
import sys

import numpy as np
import theano
import theano.tensor as T
from collections import namedtuple, defaultdict
from theano.ifelse import ifelse

x = "a b c d e f".split()
x.sort()
print(x)
