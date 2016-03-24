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
from spacy import English

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
    parser.add_argument('--bs', type=int, default=64, help='Number of backprop through time steps')
    parser.add_argument('--win', type=int, default=7, help='Number of words in context window')
    parser.add_argument('--fold', type=int, default=4, help='Fold number, 0-4')
    parser.add_argument('--lr', type=float, default=0.0627142536696559, help='Learning rate')
    parser.add_argument('--verbose', type=int, default=1, help='Verbose or not')
    parser.add_argument('--decay', type=int, default=0, help='Decay lr or not')
    parser.add_argument('--dataset', type=str, default='jeopardy', help='select dataset [atis|Jeopardy]')
    parser.add_argument('--num_questions', type=int, default=100,
                        help='number of questions to use in Jeopardy dataset')
    s = parser.parse_args()

    print '*' * 80
    print s
    folder = os.path.basename(__file__).split('.')[0]
    if not os.path.exists(folder): os.mkdir(folder)

    # load the dataset
    if s.dataset == 'jeopardy':
        root_dir = "../data/"
        new_question = True
        train, test, valid = (([], []) for _ in range(3))
        train_lex, train_y = train
        valid_lex, valid_y = valid
        test_lex, test_y = test
        lex, y, input_target_tuples = ([] for _ in range(3))
        dic = {'*': 0}
        tokenizer = English(parser=False)


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
            inpt = to_array(line)
            target = np.zeros_like(inpt)
            if answer is not None:
                answer_array = to_array(answer)
                answer_length = answer_array.size
                for i in range(inpt.size - answer_length):
                    idxs = np.arange(i, i + answer_length)
                    if np.array_equal(inpt[idxs], answer_array):
                        target[idxs] = 1
            return inpt, target


        def append_to_set(pair):
            lex.append(pair[0])
            y.append(pair[1])


        num_questions = 0
        with open(root_dir + "wiki.dat") as inputs:
            for line in inputs:
                if new_question:
                    num_questions += 1
                    if s.debug:
                        set = train
                    else:
                        # determine train, valid, or test
                        random_num = random.random()
                        if random_num < .7 or not train_y:  # 70% percent of the time
                            set = train
                        if .7 <= random_num < .8 or not valid_y:  # 10% percent of the time
                            set = valid
                        if .8 <= random_num or not test_y:  # 20% percent of the time
                            set = test
                    lex, y = set

                    append_to_set(to_instance(line))  # question
                    answer = next(inputs).rstrip()  # answer
                    line = next(inputs)  # answer sentence

                    instance = to_instance(line, answer=answer)
                    remaining_sentences = int(next(inputs))
                    input_target_tuples = []
                    new_question = False
                    input_target_tuples.append(instance)
                else:
                    remaining_sentences -= 1
                #     instance = to_instance(line)
                # input_target_tuples.append(instance)
                if not remaining_sentences:
                    new_question = True
                    # set target of eos for answer sentence to answer
                    random.shuffle(input_target_tuples)
                    for instance in input_target_tuples:
                        append_to_set(instance)
                    if num_questions >= s.num_questions:
                        break
        vocsize = len(dic)
        nclasses = 2
        idx2word = {k: v for v, k in dic.iteritems()}  # {numeric code: label}
        idx2label = {0: '0', 1: '1'}
        if s.debug:
            test_lex = valid_lex = train_lex
            test_y = valid_y = train_y

        print "number of questions:", num_questions
    else:
        train_set, valid_set, test_set, dic = load.atisfold(s.fold)
        idx2label = dict((k, v) for v, k in dic['labels2idx'].iteritems())  # {numeric code: label}
        idx2word = dict((k, v) for v, k in dic['words2idx'].iteritems())  # {numeric code: word}

        train_lex, train_ne, train_y = train_set  # number of sentences = len(train_lex)
        valid_lex, valid_ne, valid_y = valid_set
        test_lex, test_ne, test_y = test_set

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

    nsentences = len(train_lex)  # perhaps train_lex is a list of sentences
    print "size of dictionary:", vocsize
    print "number of sentences:", nsentences

    # instantiate the RNN-EM
    np.random.seed(s.seed)
    random.seed(s.seed)
    rnn = model(nh=s.hidden_size,
                nc=nclasses,
                ne=vocsize,
                de=s.emb_size,
                cs=s.win,
                memory_size=s.memory_size,
                n_memory_slots=s.n_memory_slots)

    # train with early stopping on validation set
    best_f1 = -np.inf
    s.clr = s.lr
    for e in xrange(s.n_epochs):
        # shuffle
        # shuffle([train_lex, train_ne, train_y], s.seed) I CHANGED THIS
        shuffle([train_lex, train_y], s.seed)
        s.ce = e
        tic = time.time()
        for i in xrange(nsentences):  # for each sentence
            cwords = contextwin(train_lex[i], s.win)
            words = map(lambda x: np.asarray(x).astype('int32'),
                        minibatch(cwords, s.bs))
            labels = train_y[i]
            for word_batch, label_last_word in zip(words, labels):
                rnn.train(word_batch, label_last_word, s.clr)
                rnn.normalize()
            if s.verbose:
                print '[learning] epoch %i >> %2.2f%%' % (
                    e, (i + 1) * 100. / nsentences), 'completed in %.2f (sec) <<\r' % (time.time() - tic),
                sys.stdout.flush()

        # evaluation // back into the real world : idx -> words

        predictions_test = []
        raw_predictions = []
        for x in test_lex:
            raw_prediction = rnn.classify(np.asarray(contextwin(x, s.win)).astype('int32'))
            raw_predictions.append(raw_prediction)
            predictions_test.append(map(lambda x: idx2label[x],
                                        raw_prediction))

        groundtruth_test = [map(lambda x: idx2label[x], y) for y in test_y]
        words_test = [map(lambda x: idx2word[x], w) for w in test_lex]

        predictions_valid = [map(lambda x: idx2label[x],
                                 rnn.classify(np.asarray(contextwin(x, s.win)).astype('int32')))
                             for x in valid_lex]
        groundtruth_valid = [map(lambda x: idx2label[x], y) for y in valid_y]
        words_valid = [map(lambda x: idx2word[x], w) for w in valid_lex]

        # evaluation // compute the accuracy using conlleval.pl
        res_test = conlleval(predictions_test, groundtruth_test, words_test, folder + '/current.test.txt')
        res_valid = conlleval(predictions_valid, groundtruth_valid, words_valid, folder + '/current.valid.txt')

        if res_valid['f1'] > best_f1:
            rnn.save(folder)
            best_f1 = res_valid['f1']
            if s.verbose:
                print 'NEW BEST: epoch', e, 'valid F1', res_valid['f1'], 'best test F1', res_test['f1'], ' ' * 20
            s.vf1, s.vp, s.vr = res_valid['f1'], res_valid['p'], res_valid['r']
            s.tf1, s.tp, s.tr = res_test['f1'], res_test['p'], res_test['r']
            s.be = e
            subprocess.call(['mv', folder + '/current.test.txt', folder + '/best.test.txt'])
            subprocess.call(['mv', folder + '/current.valid.txt', folder + '/best.valid.txt'])

        # print('Predictions: ')
        # for raw_prediction, prediction in zip(raw_predictions, predictions_test):
        #     print(raw_prediction)
        #     print(' '.join(prediction))

        else:
            print ''

        # learning rate decay if no improvement in 10 epochs
        if s.decay and abs(s.be - s.ce) >= 10: s.clr *= 0.5
        if s.clr < 1e-5: break

    print 'BEST RESULT: epoch', e, 'valid F1', s.vf1, 'best test F1', s.tf1, 'with the RNN-EM', folder
