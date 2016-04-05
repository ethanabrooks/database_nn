from __future__ import print_function

import lasagne
import numpy
import os
import theano
from theano import tensor as T
from theano.ifelse import ifelse


def norm(x):
    axis = None if x.ndim == 1 else 1
    return T.sqrt(T.sum(T.sqr(x), axis=axis))


def cdist(matrix, vector):
    matrix = matrix.T
    dotted = T.dot(matrix, vector)
    matrix_norms = norm(matrix)
    vector_norms = norm(vector)
    matrix_vector_norms = matrix_norms * vector_norms
    neighbors = dotted / matrix_vector_norms
    return 1. - neighbors


class model(object):
    def __init__(self, hidden_size, nclasses, num_embeddings, embedding_dim, window_size, memory_size=40,
                 n_memory_slots=8):
        """
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size
        """
        # parameters of the RNN-EM

        self.num_slots_reserved_for_questions = int(n_memory_slots / 4)  # TODO derive this from an arg
        self.emb = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (num_embeddings + 1, embedding_dim)).astype(
            theano.config.floatX))  # add one for PADDING at the end
        self.Wx = theano.shared(
            0.2 * numpy.random.uniform(-1.0, 1.0, (hidden_size, embedding_dim * window_size)).astype(
                theano.config.floatX))
        self.Wh = theano.shared(
            0.2 * numpy.random.uniform(-1.0, 1.0, (hidden_size, memory_size)).astype(theano.config.floatX))
        self.W = theano.shared(
            0.2 * numpy.random.uniform(-1.0, 1.0, (nclasses, hidden_size)).astype(theano.config.floatX))
        self.bh = theano.shared(numpy.zeros(hidden_size, dtype=theano.config.floatX))
        self.b = theano.shared(numpy.zeros(nclasses, dtype=theano.config.floatX))
        self.h0 = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (hidden_size,)).astype(theano.config.floatX))

        self.M = theano.shared(
            0.2 * numpy.random.uniform(-1.0, 1.0, (memory_size, n_memory_slots)).astype(theano.config.floatX))
        self.w0 = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (n_memory_slots,)).astype(theano.config.floatX))

        self.Wk = theano.shared(
            0.2 * numpy.random.uniform(-1.0, 1.0, (memory_size, hidden_size)).astype(theano.config.floatX))
        self.Wg = theano.shared(
            0.2 * numpy.random.uniform(-1.0, 1.0, (n_memory_slots, embedding_dim * window_size)).astype(
                theano.config.floatX))
        self.Wb = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (1, hidden_size)).astype(theano.config.floatX))
        self.Wv = theano.shared(
            0.2 * numpy.random.uniform(-1.0, 1.0, (memory_size, hidden_size)).astype(theano.config.floatX))
        self.We = theano.shared(
            0.2 * numpy.random.uniform(-1.0, 1.0, (n_memory_slots, hidden_size)).astype(theano.config.floatX))

        self.bk = theano.shared(numpy.zeros(memory_size, dtype=theano.config.floatX))
        self.bg = theano.shared(numpy.zeros(n_memory_slots, dtype=theano.config.floatX))
        self.bb = theano.shared(numpy.zeros(1, dtype=theano.config.floatX))
        self.bv = theano.shared(numpy.zeros(memory_size, dtype=theano.config.floatX))
        self.be = theano.shared(numpy.zeros(n_memory_slots, dtype=theano.config.floatX))

        # bundle
        self.params = [self.emb, self.Wx, self.Wh, self.W, self.bh, self.b, self.h0, self.Wg, self.Wb, self.Wv, self.We,
                       self.Wk, self.bk, self.bg, self.bb, self.bv, self.be]
        self.names = ['embeddings', 'Wx', 'Wh', 'W', 'bh', 'b', 'h0', 'Wg', 'Wb', 'Wv', 'We', 'Wk', 'bk', 'bg', 'bb',
                      'bv', 'be']

        is_question = T.iscalar()
        idxs = T.imatrix()  # as many columns as context window size/lines as words in the sentence

        idxs_print = theano.printing.Print('idxs')(idxs)
        shape_print = theano.printing.Print('idxs.shape[0]')(idxs.shape[0])

        x = self.emb[idxs].reshape((idxs.shape[0], embedding_dim * window_size))  # QUESTION: what is idxs.shape[0]?
        y = T.iscalar('y')  # label

        def inspect_inputs(i, node, fn):
            print(i, node, "input(s) value(s):", [input[0].shape for input in fn.inputs], end='')

        def recurrence(x_t, h_tm1, w_previous, M_previous):
            # eqn not specified in paper
            x_t_print = theano.printing.Print('x_t')(x_t)
            bg_print = theano.printing.Print('self.bg')(self.bg)
            g_t = T.nnet.sigmoid(T.dot(self.Wg, x_t) + self.bg)
            g_t_print = theano.printing.Print('g_t')(g_t)

            ### EXTERNAL MEMORY READ
            # eqn 11
            k = T.dot(self.Wk, h_tm1) + self.bk

            # eqn 13
            beta_pre = T.dot(self.Wb, h_tm1) + self.bb
            beta = T.log(1 + T.exp(beta_pre))
            beta = T.addbroadcast(beta, 0)

            # eqn 12
            w_hat = cdist(M_previous, k)
            w_hat = T.exp(beta * w_hat)
            w_hat /= T.sum(w_hat)

            # eqn 14
            w_t = (1 - g_t) * w_previous + g_t * w_hat

            # eqn 15
            c = T.dot(M_previous, w_t)

            ### EXTERNAL MEMORY UPDATE
            # eqn 16
            v = T.dot(self.Wv, h_tm1) + self.bv

            # eqn 17
            e = T.nnet.sigmoid(T.dot(self.We, h_tm1) + self.be)
            f = 1. - w_t * e

            # eqn 19
            f_diag = T.diag(f)

            # select for slots reserved for questions
            n = self.num_slots_reserved_for_questions

            subtensor = f_diag[n:, n:]
            f_diag_q = T.set_subtensor(subtensor, T.identity_like(subtensor))

            subtensor = f_diag[:n, :n]
            f_diag_nq = T.set_subtensor(subtensor, T.identity_like(subtensor))

            f_diag = ifelse(is_question, f_diag_q, f_diag_nq)

            v_print = theano.printing.Print('v')(v)
            v_ravel = v.dimshuffle(0, 'x')

            subtensor = v_ravel[n:]
            v_ravel_q = T.set_subtensor(subtensor, T.zeros_like(subtensor))

            subtensor = v_ravel[:n]
            v_ravel_nq = T.set_subtensor(subtensor, T.zeros_like(subtensor))

            v_ravel = ifelse(is_question, v_ravel_q, v_ravel_nq)

            v_ravel_print = theano.printing.Print('v_shuffle')(v_ravel)
            M_t = T.dot(M_previous, f_diag) + T.dot(v_ravel, w_t.dimshuffle('x', 0))

            # eqn 9
            h_t = T.nnet.sigmoid(T.dot(self.Wx, x_t) + T.dot(self.Wh, c) + self.bh)

            # eqn 10?
            s_t = T.nnet.softmax(T.dot(self.W, h_t) + self.b)

            return [h_t, s_t, w_t, M_t]

        [_, s, _, M], _ = theano.scan(fn=recurrence,
                                      sequences=x,
                                      outputs_info=[self.h0, None, self.w0, self.M],
                                      n_steps=x.shape[0],
                                      name='SCAN_FUNCTION')

        self.M = M

        p_y_given_x_lastword = s[-1, 0, :]
        p_y_given_x_lastword_print = theano.printing.Print('p(y|x) last word')(p_y_given_x_lastword)
        p_y_given_x_sentence = s[:, 0, :]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)
        y_pred_print = theano.printing.Print('y_pred')(y_pred)

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        nll = -T.mean(T.log(p_y_given_x_lastword)[y])
        nll_print = theano.printing.Print('negative log likelihood: ')(nll)
        updates = lasagne.updates.adadelta(nll, self.params)

        # theano functions
        self.classify = theano.function(inputs=[idxs, is_question], outputs=y_pred)

        self.train = theano.function(inputs=[idxs, y, lr, is_question],
                                     outputs=nll,
                                     updates=updates,
                                     on_unused_input='ignore',
                                     profile=True)

        self.normalize = theano.function(inputs=[],
                                         updates={self.emb:
                                                      self.emb / T.sqrt((self.emb ** 2).sum(axis=1)).dimshuffle(0,
                                                                                                                'x')})

    def save(self, folder):
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())
