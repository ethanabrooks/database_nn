from __future__ import print_function
from __future__ import print_function
import argparse
import copy

import re

import numpy as np
import time
import sys
import subprocess
import os
import random
from rnn_em import model
from is13.data import load
from is13.metrics.accuracy import conlleval
from is13.utils.tools import shuffle, minibatch, contextwin

# from spacy import English

parser = argparse.ArgumentParser()
parser.add_argument('--debug', help='Set test = train = valid',
                    action='store_true')
parser.add_argument('--hidden_size', type=int, default=100, help='Hidden size')
parser.add_argument('--memory_size', type=int, default=40, help='Memory size')
parser.add_argument('--emb_size', type=int, default=100, help='Embedding size')
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
parser.add_argument('--bucket_factor', type=int, default=2,
                    help='number of questions to use in Jeopardy dataset')
s = parser.parse_args()

print('*' * 80)
print(s)
folder = os.path.basename(__file__).split('.')[0]
if not os.path.exists(folder): os.mkdir(folder)

# instantiate the RNN-EM
np.random.seed(s.seed)
random.seed(s.seed)

PAD_VALUE = 0
NON_ANSWER_VALUE = 1
ANSWER_VALUE = 2

bucket_list = {}


def evaluate(predictions, targets):
    measures = np.zeros(3)

    for prediction, target in zip(predictions, targets):
        prediction, target = (t == np.zeros_like(t) + 2
                              for t in (prediction, target))

        def confusion((pred_is_pos, tgt_is_pos)):
            return np.logical_and(prediction == pred_is_pos,
                                  target == tgt_is_pos).sum()

        tp, fp, fn = map(confusion, ((True, True), (True, False), (False, True)))
        measures += np.array((tp, fp, fn))

    tp, fp, fn = measures
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    confusion_dic = dict()
    for var in ('f1', 'precision', 'recall'):
        confusion_dic[var] = eval(var)
    return confusion_dic


class Dataset:
    def __init__(self, percent=None):
        self.questions, self.documents, self.targets = [], [], []
        self.percent = percent

    def append(self, question, document, target):
        assert document.size == target.size
        self.questions.append(question)
        self.documents.append(document)
        self.targets.append(target)

    def predict(self):
        predictions = []
        # previous_is_question = False
        for instance in zip(self.questions, self.documents):
            # reset memory?
            # is_question = train.is_questions[i]
            # reset = is_question and not previous_is_question
            # previous_is_question = is_question

            # if is_question:
            #     question = words
            # rnn.ask_question(words)
            question, document = (np.asarray(contextwin(sentence, s.window_size), dtype='int32')
                                  for sentence in instance)
            predictions.append(rnn.classify(question, document))
        return predictions
        # [rnn.classify(
        #     np.asarray(contextwin(sentence, s.window_size), dtype='int32'))
        #         for sentence in self.inputs]


# load the dataset
root_dir = "../data/"
new_question = True
dic = {'*': 0}
# tokenizer = English(parser=False)

datasets = [Dataset(percent=p) for p in (.7,  # train
                                         .2,  # test
                                         .1)]  # valid
train, test, valid = datasets


def choose_set():
    random_num = random.random()
    datasets.sort(key=lambda ds: ds.percent)  # sort by percent
    for dataset in datasets:
        if random_num < dataset.percent:
            return dataset
        random_num -= dataset.percent


def to_int(word):
    word = word.lower()
    if word not in dic:
        w = len(dic)
        dic[word] = w
    return dic[word]


def get_bucket_size(length):
    bucket_size = 1
    while length > bucket_size:
        bucket_size *= s.bucket_factor
    if bucket_size not in bucket_list:
        bucket_list[bucket_size] = 0
    bucket_list[bucket_size] += 1
    return bucket_size


def to_array(string, bucket=True):
    tokens = re.findall(r'\w+|[:;,-=\n\.\?\(\)\-\+\{\}]', string)
    # tokens = [token.lower_.encode('unicode-escape')
    #           for token in tokenizer(unicode(string, 'utf-8'))]
    shape = len(tokens)
    if bucket:
        shape = get_bucket_size(shape)
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


num_questions = 0
with open(root_dir + "wiki.dat") as data:
    for line in data:
        if new_question:
            num_questions += 1
            if s.debug:
                dataset = train
            else:
                # determine train, valid, or test
                dataset = choose_set()

            question = to_array(line)
            # inputs, targets = to_instance(line, is_question=True, answer=None)
            # dataset.append(inputs, targets, True)  # question

            answer = next(data).rstrip()
            document = to_array(next(data))

            target = get_target(document, answer)
            remaining_sentences = int(next(data))  # num sentences remaining
            instances = []
            new_question = False
            instances.append((question, document, target))
        else:
            remaining_sentences -= 1
        # TODO instance = to_instance(line)
        # TODO input_target_tuples.append(instance)
        if not remaining_sentences:
            new_question = True
            # set target of eos for answer sentence to answer
            random.shuffle(instances)
            for instance in instances:
                dataset.append(*instance)
            if num_questions >= s.num_questions:
                break

print(bucket_list)
vocsize = len(dic)
nclasses = 3
if s.debug:
    test = valid = train

print("number of questions:", num_questions)

for ds in datasets:
    lengths = map(len, [ds.questions, ds.documents, ds.targets])
    assert len(set(lengths)) == 1
nsentences = len(train.questions)
print("size of dictionary:", vocsize)
print("number of training sentences:", nsentences)

rnn = model(s.hidden_size,
            nclasses,
            vocsize,  # num_embeddings
            s.emb_size,  # embedding_dim
            s.window_size,
            s.memory_size,
            s.n_memory_slots)

# train with early stopping on validation set
best_f1 = -np.inf
s.learn_rate = s.learn_rate
for epoch in range(s.n_epochs):
    # TODO: shuffle inputs without breaking connection between question and answer
    # shuffle([train.inputs, train.targets], s.seed)
    s.current_epoch = epoch
    tic = time.time()
    print('###\t{:10}{:10}{:10}{:10}###'
          .format('epoch', 'progress', 'loss', 'runtime'))
    for i in range(nsentences):  # for each sentence

        # context_words = contextwin(train.inputs[i], s.window_size)
        question, document = ([np.asarray(window, dtype='int32')
                               for window in contextwin(words, s.window_size)]
                              for words in (train.questions[i], train.documents[i]))

        # minibatch(context_words, s.batch_size)]
        # for word_batch, label in zip(words, train.targets[i]):

        # reset memory?
        labels = train.targets[i]
        # words = np.array(context_words, dtype='int32')
        loss = rnn.train(question, document, labels, s.learn_rate)
        # rnn.normalize() ???????
        if s.verbose:
            progress = float(i + 1) / nsentences
            print('\r###\t{:<10d}{:<10.2%}{:<10.5f}{:<10.2f}###'
                  .format(epoch, progress, float(loss), time.time() - tic), end='')
            sys.stdout.flush()


        def save_predictions(filename, targets, predictions):
            filename = 'current.{0}.txt'.format(filename)
            filepath = os.path.join(folder, filename)
            with open(filepath, 'w') as handle:
                for prediction, target in zip(predictions, targets):
                    if target is not None:
                        for label, arr in (('p: ', prediction), ('t: ', target)):
                            handle.write(label)
                            np.savetxt(handle, arr.reshape(1, -1), delimiter=' ', fmt='%i')
                            handle.write('\n')


        results = []
        for dataset, set_name in ((test, 'test'), (valid, 'valid')):
            predictions = dataset.predict()
            save_predictions(set_name, dataset.targets, predictions)
            results.append(evaluate(predictions, dataset.targets))
        res_test, res_valid = results

    if res_valid['f1'] > best_f1:
        rnn.save(folder)
        best_f1 = res_valid['f1']
        if s.verbose:
            print('\nNEW BEST: '
                  'valid F1:', res_valid['f1'],
                  'test F1:', res_test['f1'], '\n')
            sys.stdout.flush()

        for key in res_test:
            exec "s.test_{0} = res_test['{0}']".format(key)
        for key in res_valid:
            exec "s.valid_{0} = res_valid['{0}']".format(key)
        s.best_epoch = epoch
        for dset in ('test', 'valid'):
            command = 'mv {0}/current.{1}.txt {0}/best.{1}.txt'.format(folder, dset)
            subprocess.call(command.split())
    else:
        if s.verbose:
            print('\nvalid F1:', res_valid['f1'],
                  'test F1:', res_test['f1'], '\n')
            sys.stdout.flush()

    # learning rate decay if no improvement in 10 epochs
    if s.decay and abs(s.best_epoch - s.current_epoch) >= 10:
        s.learn_rate *= 0.5
    if s.learn_rate < 1e-5:
        break

print('BEST RESULT: epoch', s.best_epoch,
      'valid F1', s.valid_f1,
      'best test F1', s.test_f1,
      'with the RNN-EM', folder)
