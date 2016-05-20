from __future__ import print_function
from __future__ import print_function

import argparse
import random
import subprocess
import sys
import time
from collections import defaultdict
from collections import namedtuple
from functools import partial

import numpy as np
import os
import re
from bokeh.io import output_file, vplot, save
from bokeh.plotting import figure
from rnn_em import Model
from tabulate import tabulate

# from spacy import English

parser = argparse.ArgumentParser()
parser.add_argument('--num_questions', type=int, default=100000,
                    help='number of questions to use in Jeopardy dataset')
parser.add_argument('--hidden_size', type=int, default=90, help='Hidden size')
parser.add_argument('--memory_size', type=int, default=40, help='Memory size')
parser.add_argument('--embedding_dim', type=int, default=100, help='Embedding size')
parser.add_argument('--n_memory_slots', type=int, default=8, help='Memory slots')
parser.add_argument('--n_epochs', type=int, default=1000, help='Num epochs')
parser.add_argument('--seed', type=int, default=345, help='Seed')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Number of backprop through time steps')
parser.add_argument('--window_size', type=int, default=7,
                    help='Number of words in context window')
parser.add_argument('--fold', type=int, default=4, help='Fold number, 0-4')
parser.add_argument('--learn_rate', type=float, default=0.0627142536696559,
                    help='Learning rate')
parser.add_argument('--verbose', help='Verbose or not', action='store_true')
parser.add_argument('--decay', type=int, default=0, help='Decay learn_rate or not')
parser.add_argument('--dataset', type=str, default='jeopardy',
                    help='select dataset [atis|Jeopardy]')
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
    return int(np.math.ceil(np.math.log(length, base)))


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

        datasets = Datasets(*[Dataset(percent=p) for p in (.7,  # train
                                                           .2,  # test
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
                del (dataset.buckets[key])

            dataset.buckets = [Bucket(*map(np.array, [questions, docs, labels]))
                               for questions, docs, labels in dataset.buckets.values()]

        self.vocsize = len(dic)
        self.nclasses = 3
        self.sets = datasets

    def print_data_stats(self):
        print("\nsize of dictionary:", self.vocsize)
        print("number of questions:", self.num_questions)
        print("size of training set:", self.num_train)


def get_batches(bucket):
    num_batches = bucket.questions.shape[0] // s.batch_size + 1
    split = partial(np.array_split, indices_or_sections=num_batches)
    return zip(*map(split, (bucket.questions,
                            bucket.documents,
                            bucket.targets)))


def running_average(loss, new_loss, instances_processed, num_instances):
    if loss is None:
        return new_loss / instances_processed
    else:
        return (loss * (instances_processed - num_instances) + new_loss) / instances_processed


def print_progress(epoch, questions_processed, num_questions, loss, start_time):
    progress = round(float(questions_processed) / num_questions, ndigits=3)
    print('\r###\t{:<10d}{:<10.1%}{:<10.5f}{:<10.2f}###'
          .format(epoch, progress, float(loss), time.time() - start_time), end='')
    sys.stdout.flush()


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
    print('\n' + tabulate(confusion_matrix.__dict__.iteritems(),
                          headers=["RANDOM", "score"]))


def track_scores(all_scores, confusion_matrix, epoch, dataset_name):
    Score = namedtuple("score", "value epoch")
    scores = all_scores[dataset_name]
    table = []
    for key in confusion_matrix._fields:
        result = confusion_matrix.__getattribute__(key)
        scores[key].append(Score(result, epoch))
        best_score = max(scores[key], key=lambda score: score.value)
        table.append([key, result, best_score.value, best_score.epoch])
        if result > best_score.value:
            command = 'mv {0}/current.{1}.txt {0}/best.{1}.txt'.format(folder, dataset_name)
            subprocess.call(command.split())
    headers = [dataset_name.upper(), "score", "best score", "best score epoch"]
    print('\n\n' + tabulate(table, headers=headers))


def print_graphs(scores):
    output_file("plots.html")
    properties_per_dataset = {
        'train': {'line_color': 'firebrick'},
        'test': {'line_color': 'orange'},
        'valid': {'line_color': 'olive'}
    }

    plots = []
    for metric in ConfusionMatrix._fields:
        plot = figure(width=500, plot_height=500, title=metric)
        for dataset_name in scores:
            metric_scores = [score.value for score in scores[dataset_name][metric]]
            plot.line(range(len(metric_scores)),
                      metric_scores,
                      legend=dataset_name,
                      **properties_per_dataset[dataset_name])
        plots.append(plot)
    p = vplot(*plots)
    save(p)


if __name__ == '__main__':

    ConfusionMatrix = namedtuple("confusion_matrix", "f1 precision recall")

    data = Data()
    data.print_data_stats()

    rnn = Model(s.hidden_size,
                data.nclasses,
                data.vocsize,  # num_embeddings
                s.embedding_dim,  # embedding_dim
                s.window_size,
                s.memory_size,
                s.n_memory_slots)

    scores = {dataset_name: defaultdict(list)
              for dataset_name in Datasets._fields}
    for epoch in range(s.n_epochs):
        print('\n###\t{:10}{:10}{:10}{:10}###'
              .format('epoch', 'progress', 'loss', 'runtime'))
        start_time = time.time()
        for name in list(Datasets._fields):
            random_predictions, predictions, targets = [], [], []
            instances_processed = 0
            loss = None
            for bucket in data.sets.__getattribute__(name).buckets:
                for questions, documents, labels in get_batches(bucket):
                    if name == 'train':
                        bucket_predictions, new_loss = rnn.train(questions,
                                                                 documents,
                                                                 labels)
                        rnn.normalize()
                        num_instances = questions.shape[0]
                        instances_processed += num_instances
                        loss = running_average(loss,
                                               new_loss,
                                               instances_processed,
                                               num_instances)
                        print_progress(epoch,
                                       instances_processed,
                                       data.num_train,
                                       loss,
                                       start_time)
                    else:
                        bucket_predictions = rnn.predict(questions, documents)

                    predictions.append(bucket_predictions.reshape(labels.shape))
                    targets.append(labels)
            write_predictions_to_file(name, predictions, targets)
            confusion_matrix = evaluate(predictions, targets)
            track_scores(scores, confusion_matrix, epoch, name)
            if name == 'test':
                print_random_scores(predictions, targets)
        print_graphs(scores)
