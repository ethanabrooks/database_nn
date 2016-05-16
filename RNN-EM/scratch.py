from __future__ import print_function
import os
import sys

import numpy as np
import theano
import theano.tensor as T
from collections import namedtuple
from theano.ifelse import ifelse

folder = 'debug'

predictions = map(np.array, [[0, 0], [0, 1], [1, 1]])
targets = [np.ones(2) for _ in range(3)]

ConfusionMatrix = namedtuple("confusion_matrix", "f1 precision recall")
ANSWER_VALUE = 1


def evaluate(predictions, targets):
    """
    @:param predictions: list of predictions
    @:param targets: list of targets
    @:return dictionary with entries 'f1'
    """

    predictions, targets = (np.array(list_of_arrays).ravel()
                            for list_of_arrays in (predictions, targets))
    metrics = np.zeros(3)

    def confusion((pred_is_pos, tgt_is_pos)):
        return np.logical_and((predictions == ANSWER_VALUE) == pred_is_pos,
                              (targets == ANSWER_VALUE) == tgt_is_pos).sum()

    tp, fp, fn = map(confusion, ((True, True), (True, False), (False, True)))
    metrics += np.array((tp, fp, fn))
    tp, fp, fn = metrics
    print('tp', tp, 'fp', fp, 'fn', fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    return ConfusionMatrix(f1, precision, recall)


def track_best(best_scores, results, epoch):
    Score = namedtuple("score", "value epoch")
    for key in results._fields:
        result = results.__getattribute__(key)
        if key not in best_scores or result > best_scores[key].value:
            best_scores[key] = Score(result, epoch)

        else:
            print(results)
            sys.stdout.flush()

    print("\n{:15}{:10}{:10}".format('BEST RESULT', 'score', 'epoch'))
    for key in best_scores:
        best_score = best_scores[key]
        print("{:16}{:<10.2f}{:<10.2f}".format(key + ':', best_score.value, best_score.epoch))


best_scores = {name: dict() for name in 'train test valid'.split()}
confusion_matrix = evaluate(predictions, targets)
track_best(best_scores['train'], confusion_matrix, 0)

"""
3D x 1D: innermost dimmension
3D x 2D: innermost dimmension
"""
