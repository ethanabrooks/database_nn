import os
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


def write_predictions_to_file(dataset_name, targets, predictions):
    filename = 'current.{0}.txt'.format(dataset_name)
    filepath = os.path.join(folder, filename)
    with open(filepath, 'w') as handle:
        for prediction_array, target_array in zip(predictions, targets):
            for prediction, target in zip(prediction_array, target_array):
                for label, arr in (('p: ', prediction), ('t: ', target)):
                    handle.write(label)
                    np.savetxt(handle, arr.reshape(1, -1), delimiter=' ', fmt='%i')
                    handle.write('\n')


with open('predictions.pkl') as handle:
    predictions = pickle.load(handle)
with open('targets.pkl') as handle:
    targets = pickle.load(handle)
write_predictions_to_file('', targets, predictions)
