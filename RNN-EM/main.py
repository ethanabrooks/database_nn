from __future__ import print_function
from __future__ import print_function
import argparse
import copy
from functools import partial

import re

import numpy as np
import time
import sys
import subprocess
import os
import random
from collections import namedtuple
from collections import defaultdict
import rnn_em
from rnn_em import Model
from is13.data import load
from is13.metrics.accuracy import conlleval
from is13.utils.tools import shuffle, minibatch, contextwin

# from spacy import English

parser = argparse.ArgumentParser()
parser.add_argument('--debug', help='Set test = train = valid',
                    action='store_true')
parser.add_argument('--hidden_size', type=int, default=100, help='Hidden size')
parser.add_argument('--memory_size', type=int, default=40, help='Memory size')
parser.add_argument('--embedding_dim', type=int, default=100, help='Embedding size')
parser.add_argument('--n_memory_slots', type=int, default=8, help='Memory slots')
parser.add_argument('--n_epochs', type=int, default=50, help='Num epochs')
parser.add_argument('--seed', type=int, default=345, help='Seed')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Number of backprop through time steps')
parser.add_argument('--window_size', type=int, default=7,
                    help='Number of words in context window')
parser.add_argument('--fold', type=int, default=4, help='Fold number, 0-4')
parser.add_argument('--learn_rate', type=float, default=0.0627142536696559,
                    help='Learning rate')
parser.add_argument('--verbose', type=int, default=1, help='Verbose or not')
parser.add_argument('--decay', type=int, default=0, help='Decay learn_rate or not')
parser.add_argument('--dataset', type=str, default='jeopardy',
                    help='select dataset [atis|Jeopardy]')
parser.add_argument('--num_questions', type=int, default=1000,
                    help='number of questions to use in Jeopardy dataset')
parser.add_argument('--bucket_factor', type=int, default=4,
                    help='number of questions to use in Jeopardy dataset')
s = parser.parse_args()

print('-' * 80)
print(s)

""" Globals """
folder = os.path.basename(__file__).split('.')[0]
if not os.path.exists(folder): os.mkdir(folder)

np.random.seed(s.seed)
random.seed(s.seed)

PAD_VALUE = 0
NON_ANSWER_VALUE = 1
ANSWER_VALUE = 2

root_dir = "../data/"
s.best_f1 = -np.inf
assert s.window_size % 2 == 1, "`window_size` must be an odd number."


def get_bucket_idx(length, base):
    return np.math.ceil(np.math.log(length, base))


""" namedtuples """

Bucket = namedtuple("bucket", "questions documents targets")
Datasets = namedtuple("data_sets", "train test valid")
ConfusionMatrix = namedtuple("confusion_matrix", "f1 precision recall")

""" classes """


class Dataset:
    def __init__(self, percent=None):
        self.buckets = defaultdict(lambda: Bucket([], [], []))
        self.percent = percent

    def append(self, question, document, target):
        assert document.size == target.size
        key = tuple(get_bucket_idx(array.size, s.bucket_factor)
                    for array in (question, document))

        for i, array in enumerate((question, document, target)):
            self.buckets[key][i].append(array)


class Data:
    """
    contains global data parameters.
    Collects data and assigns to different datasets.
    """

    def __init__(self):
        # load the dataset
        # tokenizer = English(parser=False)

        datasets = Datasets(*[Dataset(percent=p) for p in (.7,    # train
                                                           .2,    # test
                                                           .1)])  # valid
        dic = {'*': 0}

        def choose_dataset():
            random_num = random.random()
            # sort by percent
            for dataset in sorted(datasets, key=lambda ds: ds.percent):
                if random_num < dataset.percent:
                    return dataset
                random_num -= dataset.percent

        def to_int(word):
            word = word.lower()
            if word not in dic:
                w = len(dic)
                dic[word] = w
            return dic[word]

        def to_array(string, bucket=True):
            tokens = re.findall(r'\w+|[:;,-=\n\.\?\(\)\-\+\{\}]', string)
            shape = len(tokens)
            if bucket:
                shape = s.bucket_factor ** get_bucket_idx(shape, s.bucket_factor)
            sentence_vector = np.zeros(shape, dtype='int32') + PAD_VALUE
            for i, word in enumerate(tokens):
                sentence_vector[i] = to_int(word)
            return sentence_vector

        def get_target(document, answer):
            targets = (document != PAD_VALUE).astype('int32')
            answer_array = to_array(answer, bucket=False)

            # set parts of inputs corresponding to complete answer to ANSWER_VALUE
            for i in range(document.size - answer_array.size):
                window = slice(i, i + answer_array.size)

                if np.all(document[window] == answer_array):
                    targets[window] = ANSWER_VALUE
                    break

            return targets

        new_question = True
        self.num_questions = 0
        self.num_train = 0
        with open(root_dir + "wiki.dat") as data:
            for line in data:
                if new_question:
                    self.num_questions += 1
                    if s.debug:
                        dataset = datasets.train
                    else:
                        dataset = choose_dataset()
                    if dataset == datasets.train:
                        self.num_train += 1

                    question = to_array(line)
                    answer = next(data).rstrip()
                    document = to_array(next(data))
                    remaining_sentences = int(next(data))
                    instances = []
                    new_question = False
                    target = get_target(document, answer)
                    instances.append((question, document, target))
                else:
                    remaining_sentences -= 1
                    # instance = to_instance(line)
                    # input_target_tuples.append(instance)
                if not remaining_sentences:
                    new_question = True
                    # set target of eos for answer sentence to answer
                    random.shuffle(instances)
                    for instance in instances:
                        dataset.append(*instance)
                    if self.num_questions >= s.num_questions:
                        break

        if s.debug:
            datasets = Datasets(*[datasets.train] * 3)

        print('Bucket allocation:')
        for dataset in datasets:
            delete = []
            print('\nNumber of buckets: ', len(dataset.buckets))
            for key in dataset.buckets:
                num_instances = len(dataset.buckets[key].questions)
                if num_instances < 10:
                    delete.append(key)
                else:
                    print(key, num_instances)

            for key in delete:
                del(dataset.buckets[key])

            dataset.buckets = [Bucket(*map(np.array, [questions, docs, labels]))
                               for questions, docs, labels in dataset.buckets.values()]

        self.vocsize = len(dic)
        self.nclasses = 3
        self.sets = datasets

    def print_data_stats(self):
        print("\nsize of dictionary:", self.vocsize)
        print("number of questions:", self.num_questions)
        print("size of training set:", self.num_train)


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
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    return ConfusionMatrix(f1, precision, recall)


def print_progress(epoch, questions_processed, num_questions, loss, start_time):
    progress = float(questions_processed) / num_questions
    print('\r###\t{:<10d}{:<10.2%}{:<10.5f}{:<10.2f}###'
          .format(epoch, progress, float(loss), time.time() - start_time), end='')
    sys.stdout.flush()


def write_predictions_to_file(dataset_name, targets, predictions):
    filename = 'current.{0}.txt'.format(dataset_name)
    filepath = os.path.join(folder, filename)
    with open(filepath, 'w') as handle:
        for prediction, target in zip(predictions, targets):
            if target is not None:
                for label, arr in (('p: ', prediction), ('t: ', target)):
                    handle.write(label)
                    np.savetxt(handle, arr.reshape(1, -1), delimiter=' ', fmt='%i')
                    handle.write('\n')


def track_best(best_scores, confusion_matrix, epoch):
    Score = namedtuple("score", "value epoch")
    for key in confusion_matrix._fields:
        result = confusion_matrix.__getattribute__(key)
        if key not in best_scores or result > best_scores[key].value:
            best_scores[key] = Score(result, epoch)

            # rnn.save(folder)
            if key is 'F1':
                if s.verbose:
                    print('\nNEW BEST F1: {0}\nEpoch\n'.format(*best_scores[key]))
                    sys.stdout.flush()

                for dataset in ('test', 'valid'):
                    command = 'mv {0}/current.{1}.txt {0}/best.{1}.txt'.format(folder, dataset)
                    subprocess.call(command.split())
        else:
            if s.verbose:
                print(confusion_matrix)
                sys.stdout.flush()

    print("{:15}{:10}{:10}".format('best result', 'score', 'epoch'))
    for key in best_scores:
        best_score = best_scores[key]
        print("{:16}{:<10.2f}{:<10.2f}".format(key + ':',
                                               best_score.value,
                                               best_score.epoch))


if __name__ == '__main__':
    data = Data()
    data.print_data_stats()

    rnn = Model(s.hidden_size,
                data.nclasses,
                data.vocsize,  # num_embeddings
                s.embedding_dim,  # embedding_dim
                s.window_size,
                s.memory_size,
                s.n_memory_slots)

    for epoch in range(s.n_epochs):

        print('###\t{:10}{:10}{:10}{:10}###'
              .format('epoch', 'progress', 'loss', 'runtime'))
        start_time = time.time()
        names = data.sets._fields
        best_scores = {name: dict() for name in names}

        for name in names:
            predictions, targets = [], []
            instances_processed = 0
            for bucket in data.sets.__getattribute__(name).buckets:
                if name == 'train':
                    # np.savetxt('questions.npy', bucket.questions)
                    # np.savetxt('documents.npy', bucket.documents)
                    # np.savetxt('targets.npy', bucket.targets)
                    bucket_predictions, loss = rnn.train(bucket.questions,
                                                         bucket.documents,
                                                         bucket.targets)
                    print(bucket_predictions)
                    print('-----------')
                    print(loss)
                    exit(0)
                    rnn.normalize()
                    instances_processed += bucket.questions.shape[0]
                    print_progress(epoch,
                                   instances_processed,
                                   data.num_train,
                                   loss,
                                   start_time)
                else:
                    bucket_predictions = rnn.predict(bucket.questions, bucket.documents)
                predictions.append(bucket_predictions)
                targets.append(bucket.targets)
            write_predictions_to_file(name, predictions, targets)
            confusion_matrix = evaluate(predictions, targets)
            print("\n" + name.upper())
            track_best(best_scores[name], confusion_matrix, epoch)
