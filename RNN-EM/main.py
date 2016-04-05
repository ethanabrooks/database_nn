from __future__ import print_function
from __future__ import print_function
import argparse
import copy

import tabulate as table

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
from spacy import English

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
parser.add_argument('--num_questions', type=int, default=100000,
                    help='number of questions to use in Jeopardy dataset')
s = parser.parse_args()

print('*' * 80)
print(s)
folder = os.path.basename(__file__).split('.')[0]
if not os.path.exists(folder): os.mkdir(folder)


def evaluate(predictions, targets):
    measures = np.zeros(3)
    for pred, target in zip(predictions, targets):
        def confusion((predicted, actual)):
            return pred[np.logical_and(pred == predicted, target == actual)].sum()

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
    def __init__(self,
                 percent=None,
                 inputs=None,
                 targets=None):
        if inputs is None:
            inputs = []
        if targets is None:
            targets = []
        self.is_questions = []
        self.__dict__.update(locals())
        # self.percent = percent;
        # self. ...

    def size(self):
        len_inputs, len_targets = map(len, (self.inputs, self.targets))
        assert len_inputs == len_targets
        return len_inputs

    def append(self, inputs, targets, is_question):
        self.inputs.append(inputs)
        self.targets.append(targets)
        self.is_questions.append(is_question)

    def predict(self):
        predictions = []
        for sentence, is_question in zip(self.inputs, self.is_questions):
            predictions.append(rnn.classify(
                np.asarray(contextwin(sentence, s.window_size), dtype='int32'),
                is_question))
        return predictions
        # [rnn.classify(
        #     np.asarray(contextwin(sentence, s.window_size), dtype='int32'))
        #         for sentence in self.inputs]


# load the dataset
if s.dataset == 'jeopardy':
    root_dir = "../data/"
    new_question = True
    dic = {'*': 0}
    tokenizer = English(parser=False)

    datasets = [Dataset(percent=p) for p in (.7,  # train
                                             .2,  # test
                                             .1)]  # valid
    train, test, valid = datasets


    def choose_set():
        random_num = random.random()
        datasets.sort(key=lambda ds: ds.percent)
        for dataset in datasets:  # sort by percent
            if random_num < dataset.percent:
                return dataset
            random_num -= dataset.percent


    def to_int(word):
        word = word.lower()
        if word not in dic:
            w = len(dic)
            dic[word] = w
        return dic[word]


    def to_array(string):
        tokens = [token.lower_.encode('unicode-escape')
                  for token in tokenizer(unicode(string, 'utf-8'))]
        sentence_vector = np.empty(len(tokens), dtype=int)
        for i, word in enumerate(tokens):
            sentence_vector[i] = to_int(word)
        return sentence_vector


    def to_instance(line, answer=None):
        inputs = to_array(line)
        targets = np.zeros_like(inputs)
        if answer is not None:
            answer_array = to_array(answer)
            answer_size = answer_array.size

            # set parts of inputs corresponding to complete answer to 1
            for i in range(inputs.size - answer_size):
                window = inputs[i:i + answer_size]
                targets[i] = np.all(window == answer_array)
        return inputs, targets


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

                inputs, targets = to_instance(line)
                dataset.append(inputs, targets, True)  # question

                answer = next(data).rstrip()  # answer
                line = next(data)  # answer sentence

                instance = to_instance(line, answer)
                remaining_sentences = int(next(data))  # num sentences remainind
                instances = []
                new_question = False
                instances.append(instance)
            else:
                remaining_sentences -= 1
            # TODO instance = to_instance(line)
            # TODO input_target_tuples.append(instance)
            if not remaining_sentences:
                new_question = True
                # set target of eos for answer sentence to answer
                random.shuffle(instances)
                for inputs, targets in instances:
                    dataset.append(inputs, targets, False)
                if num_questions >= s.num_questions:
                    break

    vocsize = len(dic)
    nclasses = 2
    idx2word = {k: v for v, k in dic.iteritems()}  # {numeric code: label}
    idx2label = {0: '0', 1: '1'}
    if s.debug:
        test = valid = train

    print("number of questions:", num_questions)
else:
    train_set, valid_set, test_set, dic = load.atisfold(s.fold)
    atis = load.atisfold(s.fold)
    idx2label, idx2word = ({dic[version][k]: k
                            for k in dic[version]}
                           for version in ('labels2idx', 'words2idx'))

    train, valid, test = (Dataset(inputs=lex, targets=y)
                          for (lex, _, y) in (train_set, valid_set, test_set))

    # number of distinct words
    vocsize = len(dic['words2idx'])

    # number of distinct classes
    nclasses = len(dic['labels2idx'])

nsentences = len(train.inputs)  # perhaps train_lex is a list of sentences
print("size of dictionary:", vocsize)
print("number of sentences:", nsentences)

# instantiate the RNN-EM
np.random.seed(s.seed)
random.seed(s.seed)

rnn = model(s.hidden_size,
            nclasses,
            vocsize,
            s.emb_size,
            s.window_size,
            s.memory_size,
            s.n_memory_slots)

# train with early stopping on validation set
best_f1 = -np.inf
s.learn_rate = s.learn_rate
for epoch in range(s.n_epochs):
    shuffle([train.inputs, train.targets], s.seed)
    s.current_epoch = epoch
    tic = time.time()
    for i in range(nsentences):  # for each sentence
        context_words = contextwin(train.inputs[i], s.window_size)
        words = [np.asarray(instance, dtype='int32') for instance in
                 minibatch(context_words, s.batch_size)]
        for word_batch, label in zip(words, train.targets[i]):
            rnn.train(word_batch, label, s.learn_rate, train.is_questions[i])
            rnn.train.profile.print_summary()
            rnn.normalize()
        if s.verbose:
            progress = (i + 1) * 100. / nsentences
            print('[learning] epoch {0:d} >> {1:2.2f}%'.format(epoch, progress),
                  'completed in {0:.2f} (sec) <<'.format(time.time() - tic),
                  end='\r')  # write in-place
            sys.stdout.flush()

    if s.dataset == 'atis':

        def translate(subset, dic=idx2label):
            return [map(dic.__getitem__, w) for w in subset]


        for set_name in ('test', 'valid'):
            dataset = eval(set_name)
            predictions = dataset.predict()

            predicted_labels, actual_labels = map(translate,
                                                  (predictions, dataset.targets))
            input_words = translate(dataset.inputs, idx2word)
            res = conlleval(predicted_labels, actual_labels, input_words,
                            '{0}/current.{1}.txt'.format(folder, set_name))
            exec 'res_{} = res'.format(set_name)
    else:
        def save_predictions(filename, targets, predictions):
            filename = 'current.{0}.txt'.format(filename)
            filepath = os.path.join(folder, filename)
            with open(filepath, 'w') as handle:
                for target, prediction in zip(targets, predictions):
                    for label, arr in (('t: ', target), ('p: ', prediction)):
                        handle.write(label)
                        np.savetxt(handle, arr.reshape(1, -1), delimiter=' ', fmt='%i')
                        handle.write('\n')


        for set_name in ('test', 'valid'):
            dataset = eval(set_name)
            predictions = dataset.predict()
            save_predictions(set_name, dataset.targets, predictions)
            res = evaluate(predictions, dataset.targets)
            exec 'res_{0} = res'.format(set_name)

    if res_valid['f1'] > best_f1:
        rnn.save(folder)
        best_f1 = res_valid['f1']
        if s.verbose:
            print('NEW BEST: epoch', epoch,
                  'valid F1', res_valid['f1'],
                  'best test F1', res_test['f1'],
                  ' ' * 20)
            sys.stdout.flush()

        for key in res_test:
            exec "s.test_{0} = res_test['{0}']".format(key)
        for key in res_valid:
            exec "s.valid_{0} = res_valid['{0}']".format(key)
        s.best_epoch = epoch
        for dset in ('test', 'valid'):
            command = 'mv {0}/current.{1}.txt {0}/best.{1}.txt'.format(folder, dset)
            subprocess.call(command.split())

    # learning rate decay if no improvement in 10 epochs
    if s.decay and abs(s.best_epoch - s.current_epoch) >= 10:
        s.learn_rate *= 0.5
    if s.learn_rate < 1e-5:
        break

print('BEST RESULT: epoch', epoch,
      'valid F1', s.valid_f1,
      'best test F1', s.test_f1,
      'with the RNN-EM', folder)
