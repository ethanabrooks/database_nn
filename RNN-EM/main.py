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


def jeopardy_eval(raw, y):
    def confusion(predicted=True, actual=True):
        return raw[np.logical_and(raw == predicted, y == actual)].sum()

    tp, fp, fn = map(confusion, ((True, True), (True, False), (False, True)))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    confusion_dic = dict()
    for var in ('f1', 'precision', 'recall'):
        confusion_dic[var] = eval(var)
    return confusion_dic


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', help='Set test = train = valid',
                        action='store_true')
    parser.add_argument('--hidden_size', type=int, default=100, help='Hidden size')
    parser.add_argument('--memory_size', type=int, default=40, help='Memory size')
    parser.add_argument('--emb_size', type=int, default=100, help='Embedding size')
    parser.add_argument('--n_memory_slots', type=int, default=1, help='Memory slots')
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
    parser.add_argument('--num_questions', type=int, default=100,
                        help='number of questions to use in Jeopardy dataset')
    s = parser.parse_args()

    print('*' * 80)
    print(s)
    folder = os.path.basename(__file__).split('.')[0]
    if not os.path.exists(folder): os.mkdir(folder)


    class Dataset:
        def __init__(self, dic, percent=None, inputs=None, targets=None):
            if inputs is None:
                inputs = []
            if targets is None:
                targets = []
            self.__dict__.update(locals())
            # self.percent = percent;
            # self. ...

        def size(self):
            len_inputs, len_targets = map(len, (self.inputs, self.targets))
            assert len_inputs == len_targets
            return len_inputs

        def append(self, inputs, targets):
            self.inputs.append(inputs)
            self.targets.append(targets)

            # def words(self, idxs):
            #     return [self.dic[i] for i in idxs]
            #
            # def labels(self, idxs):


    # load the dataset
    if s.dataset == 'jeopardy':
        root_dir = "../data/"
        new_question = True
        dic = {'*': 0}
        tokenizer = English(parser=False)
        datasets = (Dataset(dic, percent=p) for p in (.7,  # train
                                                      .2,  # test
                                                      .1))  # valid
        train, test, valid = datasets


        def choose_set():
            random_num = random.random()
            for dataset in sorted(datasets, key=lambda ds: ds.percent):  # sort by percent
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

                if answer_size > 1:
                    x = 0
                    pass

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
                        input_words, targets = choose_set()

                    inputs, targets = to_instance(line)
                    dataset.append(inputs, targets)  # question
                    answer = next(data).rstrip()  # answer
                    line = next(data)  # answer sentence

                    instance = to_instance(line, answer)
                    remaining_sentences = int(next(data))  # num sentences remainind
                    instances = []
                    new_question = False
                    instances.append(instance)
                else:
                    remaining_sentences -= 1
                # instance = to_instance(line)
                # input_target_tuples.append(instance)
                if not remaining_sentences:
                    new_question = True
                    # set target of eos for answer sentence to answer
                    random.shuffle(instances)
                    for inputs, targets in instances:
                        dataset.append(inputs, targets)
                    if num_questions >= s.num_questions:
                        break

        vocsize = len(dic)
        nclasses = 2
        idx2word = {k: v for v, k in dic.iteritems()}  # {numeric code: label}
        idx2label = {0: '0', 1: '1'}
        if s.debug:
            test.inputs = valid.inputs = train.inputs
            test.targets = valid.targets = train.targets

        print("number of questions:", num_questions)
    else:
        train_set, valid_set, test_set, dic = load.atisfold(s.fold)
        atis = load.atisfold(s.fold)
        # idx2label = dict((k, v) for v, k in dic['labels2idx'].iteritems())  # {numeric code: label}
        # idx2word = dict((k, v) for v, k in dic['words2idx'].iteritems())  # {numeric code: word}
        idx2label, idx2word = ({k: dic[k] for k in dic[version]}
                               for version in ('labels2idx', 'words2idx'))

        # train_lex, train_ne, train_y = train_set  # number of sentences = len(train_lex)
        # valid_lex, valid_ne, valid_y = valid_set
        # test_lex, test_ne, test_y = test_set
        train, valid, test = (Dataset(inputs=lex, targets=y)
                              for (lex, _, y) in (train_set, valid_set, test_set))

        # number of distinct words
        vocsize = len(dic['words2idx'])
        # vocsize = len(set(reduce( \
        #     lambda x, y: list(x) + list(y), \
        #     train_lex + valid_lex + test_lex)))

        # number of distinct classes
        nclasses = len(dic['labels2idx'])
        # nclasses = len(set(reduce( \
        #     lambda x, y: list(x) + list(y), \
        #     train_y + test_y + valid_y)))

    nsentences = len(train.inputs)  # perhaps train_lex is a list of sentences
    print("size of dictionary:", vocsize)
    print("number of sentences:", nsentences)

    # instantiate the RNN-EM
    np.random.seed(s.seed)
    random.seed(s.seed)
    rnn = model(hidden_size=s.hidden_size,
                nclasses=nclasses,
                num_embeddings=vocsize,
                embedding_dim=s.emb_size,
                window_size=s.window_size,
                memory_size=s.memory_size,
                n_memory_slots=s.n_memory_slots)

    # train with early stopping on validation set
    best_f1 = -np.inf
    s.learn_rate = s.learn_rate
    for epoch in range(s.n_epochs):
        # shuffle
        # shuffle([train_lex, train_ne, train_y], s.seed) I CHANGED THIS
        shuffle([train.inputs, train.targets], s.seed)
        s.current_epoch = epoch
        tic = time.time()
        for i in range(nsentences):  # for each sentence
            context_words = contextwin(train.inputs[i], s.window_size)
            words = np.array(minibatch(context_words, s.batch_size), dtype='int32')
            for word_batch, label in zip(words, train.targets[i]):
                rnn.train(word_batch, label, s.learn_rate)
                rnn.normalize()
            if s.verbose:
                print('[learning] epoch %i >> %2.2f%%' % (
                    epoch, (i + 1) * 100. / nsentences), 'completed in %.2f (sec) <<\r' % (time.time() - tic))
                sys.stdout.flush()

        # evaluation // back into the real world : idx -> words

        # test_predicted_labels = []
        # test_predictions = []
        # for x in test.inputs:
        #     prediction = rnn.classify(np.asarray(contextwin(x, s.window_size)).astype('int32'))
        #     test_predictions.append(prediction)
        #     test_predicted_labels.append(map(lambda x: idx2label[x], prediction))
        #
        # test_labels = [map(lambda x: idx2label[x], targets) for targets in test.targets]
        # test_input_words = [map(lambda x: idx2word[x], w) for w in test.inputs]
        #
        # valid_predicted_labels = []
        # valid_predictions = []
        # for x in valid.inputs:
        #     prediction = rnn.classify(np.asarray(contextwin(x, s.window_size)).astype('int32'))
        #     valid_predictions.append(prediction)
        #     valid_predicted_labels.append(map(lambda x: idx2label[x], prediction))
        #
        # valid_labels = [map(lambda x: idx2label[x], targets) for targets in valid.targets]
        # valid_input_words = [map(lambda x: idx2word[x], w) for w in valid.inputs]

        # evaluation // compute the accuracy using conlleval.pl
        if s.dataset == 'atis':
            # res_test = conlleval(test_predicted_labels, test_labels,
            #                      test_input_words, folder + '/current.test.txt')
            # res_valid = conlleval(valid_predicted_labels, valid_labels,
            #                       valid_input_words, folder + '/current.valid.txt')
            res_test = conlleval(test.predicted_labels(), test.actual_labels(),
                                 test.input_words(), folder + '/current.test.txt')
            res_valid = conlleval(valid.predicted_labels(), valid.actual_labels(),
                                  valid.input_words(), folder + '/current.test.txt')
        else:
            def save_predictions(dataset, filename):
                with (open(os.path.join(folder, 'current.test.txt'))) as save:
                    y_vs_predicted = zip(dataset['y'], dataset['predicted'])
                    save.write(table.tabulate(y_vs_predicted))


            res_test = jeopardy_eval(test.predicted(), test.targets)
            res_valid = jeopardy_eval(valid.predicted(), valid.targets)

        if res_valid['f1'] > best_f1:
            rnn.save(folder)
            best_f1 = res_valid['f1']
            if s.verbose:
                print('NEW BEST: epoch', epoch,
                      'valid F1', res_valid['f1'],
                      'best test F1', res_test['f1'],
                      ' ' * 20)

            for key in res_test:
                exec 's.test_{0} = res_test[{0}]'.format(key)
            for key in res_valid:
                exec 's.valid_{0} = res_valid[{0}]'.format(key)
            # s.valid_f1, s.valid_precision, s.valid_recall = (
            #     res_valid[key] for key in ('f1', 'precision', 'recall'))
            # s.test_f1, s.test_precision, s.test_recall = (
            #     res_test[key] for key in ('f1', 'precision', 'recall'))
            s.best_epoch = epoch
            for dset in ('test', 'valid'):
                command = 'mv {0}/current.{1}.txt {0}/best.{1}.txt'.format(folder, dset)
                subprocess.call(command.split())

        # learning rate decay if no improvement in 10 epochs
        if s.decay and abs(s.best_epoch - s.current_epoch) >= 10:
            s.learn_rate *= 0.5
        if s.learn_rate < 1e-5: break

    print('BEST RESULT: epoch', epoch,
          'valid F1', s.valid_f1,
          'best test F1', s.test_f1,
          'with the RNN-EM', folder)
