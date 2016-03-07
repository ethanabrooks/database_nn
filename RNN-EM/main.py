import argparse

import re

import numpy
import time
import sys
import subprocess
import os
import random
from rnn_em import model
from is13.data import load
from is13.metrics.accuracy import conlleval
from is13.utils.tools import shuffle, minibatch, contextwin

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=100, help='Hidden size')
    parser.add_argument('--memory_size', type=int, default=40, help='Memory size')
    parser.add_argument('--emb_size', type=int, default=100, help='Embedding size')
    parser.add_argument('--n_memory_slots', type=int, default=1, help='Memory slots')
    parser.add_argument('--n_epochs', type=int, default=50, help='Num epochs')
    parser.add_argument('--seed', type=int, default=345, help='Seed')
    parser.add_argument('--bs', type=int, default=9, help='Number of backprop through time steps')
    parser.add_argument('--win', type=int, default=7, help='Number of words in context window')
    parser.add_argument('--fold', type=int, default=4, help='Fold number, 0-4')
    parser.add_argument('--lr', type=float, default=0.0627142536696559, help='Learning rate')
    parser.add_argument('--verbose', type=int, default=1, help='Verbose or not')
    parser.add_argument('--decay', type=int, default=0, help='Decay lr or not')
    parser.add_argument('--dataset', type=str, default='atis', help='select dataset [atis|Jeopardy]')
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
        lex, y, sentences = ([] for _ in range(3))
        dic = {'*': 0}


        def to_vector(string):
            tokens = re.findall(r'\w+|[:;,-=\.\?\(\)\-\+\{\}]', string)
            sentence_vector = numpy.empty(len(tokens), dtype=int)
            for i, word in enumerate(tokens):
                word = word.lower()
                if word not in dic:
                    w = len(dic)
                    dic[word] = w
                sentence_vector[i] = dic[word]
            return sentence_vector


        def to_vectors(string):
            sentence_vector = to_vector(string)
            return sentence_vector, numpy.zeros_like(sentence_vector)


        def append_to_set(to_lex, to_y):
            if to_lex is None and to_y is None:
                lex.append(numpy.empty(0))
                y.append(numpy.empty(0))
            elif to_lex is None:
                to_y, to_lex = to_vectors(to_y)
            elif to_y is None:
                to_lex, to_y = to_vectors(to_lex)
            else:
                to_lex, _ = to_vectors(to_lex)
                to_y, _ = to_vectors(to_y)
                assert to_lex.size == to_y.size
            assert len(to_lex.shape) == 1
            assert len(to_y.shape) == 1
            lex.append(to_lex)
            y.append(to_y)


        num_questions = 0
        with open(root_dir + "nn_output") as inputs:
            for line in inputs:
                if new_question:
                    num_questions += 1
                    # determine train, valid, or test
                    random_num = random.random()
                    if random_num < .7:  # 70% percent of the time
                        set = train
                    elif random_num < .8:  # 10% percent of the time
                        set = valid
                    else:  # 20% percent of the time
                        set = test
                    lex, y = set

                    append_to_set(line, None)  # question
                    next(inputs)  # skip one-word answer
                    answer = next(inputs)  # save for later
                    remaining_sentences = int(next(inputs))
                    sentences = []  # to be filled by 'context' sentences
                    new_question = False
                else:
                    sentences.append(line)
                    remaining_sentences -= 1
                if not remaining_sentences:
                    new_question = True
                    random.shuffle(sentences)
                    for sentence in sentences:
                        append_to_set(sentence, None)
                    append_to_set(None, answer)
                    # if len(train_lex) > 1 and len(valid_lex) > 1 and len(test_lex) > 1:
                    # break
                    if num_questions > 500:
                        break
        vocsize = nclasses = len(dic)
        idx2label = idx2word = {k: v for v, k in dic.iteritems()}  # {numeric code: label}

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
    print "number of questions:", num_questions

    # instanciate the RNN-EM
    numpy.random.seed(s.seed)
    random.seed(s.seed)
    rnn = model(nh=s.hidden_size,
                nc=nclasses,
                ne=vocsize,
                de=s.emb_size,
                cs=s.win,
                memory_size=s.memory_size,
                n_memory_slots=s.n_memory_slots)

    # train with early stopping on validation set
    best_f1 = -numpy.inf
    s.clr = s.lr
    for e in xrange(s.n_epochs):
        # shuffle
        # shuffle([train_lex, train_ne, train_y], s.seed) I CHANGED THIS
        shuffle([train_lex, train_y], s.seed)
        s.ce = e
        tic = time.time()
        for i in xrange(nsentences):  # for each sentence
            cwords = contextwin(train_lex[i], s.win)
            words = map(lambda x: numpy.asarray(x).astype('int32'),
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
        x_with_context = numpy.asarray(contextwin(x, s.win)).astype('int32')
        if x_with_context.ndim == 0:
            print "WTF!!!!"
            exit(1)
        if x_with_context.ndim == 1:
            x_with_context.shape = (1, -1)

        assert x_with_context.ndim == 2
        predictions_test = [map(lambda x: idx2label[x],
                                rnn.classify(x_with_context))
                            for x in test_lex]
        print('Predictions: ')
        for prediction in predictions_test:
            print(' '.join(prediction))
        groundtruth_test = [map(lambda x: idx2label[x], y) for y in test_y]
        words_test = [map(lambda x: idx2word[x], w) for w in test_lex]

        predictions_valid = [map(lambda x: idx2label[x],
                                 rnn.classify(numpy.asarray(contextwin(x, s.win)).astype('int32')))
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
        else:
            print ''

        # learning rate decay if no improvement in 10 epochs
        if s.decay and abs(s.be - s.ce) >= 10: s.clr *= 0.5
        if s.clr < 1e-5: break

    print 'BEST RESULT: epoch', e, 'valid F1', s.vf1, 'best test F1', s.tf1, 'with the RNN-EM', folder
