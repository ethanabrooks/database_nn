from __future__ import print_function

import pickle

import os
import sys

import numpy as np
import theano
import theano.tensor as T
from collections import namedtuple, defaultdict
from theano.ifelse import ifelse

ANSWER_VALUE = 2
ConfusionMatrix = namedtuple("confusion_matrix", "f1 precision recall")


def evaluate(predictions, targets):
    """
    @:param predictions: list of predictions
    @:param targets: list of targets
    @:return dictionary with entries 'f1'
    """

    def to_vector(list_of_arrays):
        return np.hstack(array.ravel() for array in list_of_arrays)

    predictions, targets = map(to_vector, (predictions, targets))

    metrics = np.zeros(3)

    def confusion((pred_is_pos, tgt_is_pos)):
        logical_and = np.logical_and(
            (predictions == ANSWER_VALUE) == pred_is_pos,
            (targets == ANSWER_VALUE) == tgt_is_pos
        )
        return logical_and.sum()

    tp, fp, fn = map(confusion, ((True, True), (True, False), (False, True)))
    metrics += np.array((tp, fp, fn))
    tp, fp, fn = metrics
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    return ConfusionMatrix(f1, precision, recall)


with open("predictions.pkl") as handle:
    predictions = pickle.load(handle)
with open("targets.pkl") as handle:
    targets = pickle.load(handle)
evaluate(targets, predictions)
