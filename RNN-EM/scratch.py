from __future__ import print_function

import pickle
import random

import subprocess

import os
import sys

import numpy as np
import theano
import theano.tensor as T
from collections import namedtuple, defaultdict
from theano.ifelse import ifelse
from bokeh.io import output_file, show, vplot
from bokeh.plotting import figure

y = list(range(11))
y0 = y
y1 = [10 - i for i in y]
y2 = [abs(i - 5) for i in y]

# create a new plot
s1 = figure(width=250, plot_height=250, title=None)
s1.line(y, y0, color="navy")
s1.circle(y, y0, size=10, color="navy", alpha=0.5)

# create another one
s2 = figure(width=250, height=250, title=None)
s2.triangle(y, y1, size=10, color="firebrick", alpha=0.5)

# create and another
s3 = figure(width=250, height=250, title=None)
s3.square(y, y2, size=10, color="olive", alpha=0.5)

output_file("layout.html")

Bucket = namedtuple("bucket", "questions documents targets")
Datasets = namedtuple("data_sets", "train test valid")
ConfusionMatrix = namedtuple("confusion_matrix", "f1 precision recall")


def metrics():
    return {metric: [random.random() for _ in range(10)]
            for metric in ConfusionMatrix._fields}


scores = {dataset_name: metrics() for dataset_name in Datasets._fields}

properties_per_dataset = {
    'train': {'line_color': 'firebrick'},
    'test': {'line_color': 'orange'},
    'valid': {'line_color': 'olive'}
}

plots = []
for metric in ConfusionMatrix._fields:
    plot = figure(width=500, plot_height=500, title=metric)
    for dataset_name in scores:
        metric_scores = scores[dataset_name][metric]
        plot.line(range(len(metric_scores)),
                  metric_scores,
                  legend=dataset_name,
                  **properties_per_dataset[dataset_name])
    plots.append(plot)

p = vplot(*plots)

show(p)
