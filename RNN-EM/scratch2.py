import pickle
import subprocess
from collections import namedtuple

import numpy as np
from tabulate import tabulate

NON_ANSWER_VALUE = 1
ANSWER_VALUE = 2

root_dir = "../data/"
folder = ''

ConfusionMatrix = namedtuple("confusion_matrix", "f1 precision recall")
Datasets = namedtuple("data_sets", "train test valid")

metrics = {metric: [] for metric in ConfusionMatrix._fields}
scores = {dataset_name: metrics for dataset_name in Datasets._fields}

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


def print_random_scores(targets, predictions):
    predictions = [np.random.randint(low=NON_ANSWER_VALUE,
                                     high=ANSWER_VALUE + 1,
                                     size=array.shape) for array in predictions]
    confusion_matrix = evaluate(predictions, targets)
    print(tabulate(confusion_matrix.__dict__.iteritems(),
                   headers=["RANDOM", "score"]))


def track_scores(scores, confusion_matrix, epoch, dataset_name):
    Score = namedtuple("score", "value epoch")
    scores = scores[dataset_name]
    table = []
    for key in confusion_matrix._fields:
        result = confusion_matrix.__getattribute__(key)
        scores[key].append(Score(result, epoch))
        best_score = max(scores[key], key=lambda score: score.value)
        table.append([key, result, best_score.value, best_score.epoch])
        if result > best_score.value:
            command = 'mv {0}/current.{1}.txt {0}/best.{1}.txt'.format(folder, dataset_name)
            subprocess.call(command.split())
    print(tabulate(table, headers=[dataset_name, "score", "best score", "best score epoch"]))


with open('predictions.pkl') as handle:
    predictions = pickle.load(handle)
with open('targets.pkl') as handle:
    targets = pickle.load(handle)
confusion_matrix = evaluate(predictions, targets)
print_random_scores(predictions, targets)
