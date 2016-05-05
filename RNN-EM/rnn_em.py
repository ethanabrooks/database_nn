from __future__ import print_function

import lasagne
import numpy
import os
import theano
from theano.printing import Print
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


# noinspection PyPep8Naming
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

        num_slots_reserved_for_questions = int(n_memory_slots / 4)  # TODO derive this from an arg
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

        self.M0 = theano.shared(
            0.2 * numpy.random.uniform(-1.0, 1.0, (memory_size, n_memory_slots)).astype(theano.config.floatX))
        self.M = self.M0
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
        self.params = [self.emb, self.Wx, self.Wh, self.W, self.bh, self.b, self.h0, self.Wg, self.Wb,
                       # self.Wv,# self.We,
                       self.Wk, self.bk, self.bg, self.bb]  # , self.bv] #, self.be]
        self.names = ['embeddings', 'Wx', 'Wh', 'W', 'bh', 'b', 'h0', 'Wg', 'Wb', 'Wv', 'We', 'Wk', 'bk', 'bg', 'bb',
                      'bv', 'be']

        is_question = T.iscalar()
        idxs = T.imatrix()  # as many columns as context window size/lines as words in the sentence

        idxs_print = theano.printing.Print('idxs')(idxs)
        shape_print = theano.printing.Print('idxs.shape[0]')(idxs.shape[0])

        x = self.emb[idxs].reshape((idxs.shape[0], embedding_dim * window_size))  # QUESTION: what is idxs.shape[0]?
        y = T.ivector('y')  # label

        def recurrence(x_t, h_tm1, w_previous, M_previous, is_question):
            # eqn not specified in paper
            g_t = T.nnet.sigmoid(T.dot(self.Wg, x_t) + self.bg)  # [n_memory_slots]

            ### EXTERNAL MEMORY READ
            # eqn 11
            k = T.dot(self.Wk, h_tm1) + self.bk  # [memory_size]

            # eqn 13
            beta_pre = T.dot(self.Wb, h_tm1) + self.bb
            beta = T.log(1 + T.exp(beta_pre))
            beta = T.addbroadcast(beta, 0)  # [1]

            # eqn 12
            w_hat = cdist(M_previous, k)
            w_hat = T.exp(beta * w_hat)
            w_hat /= T.sum(w_hat)  # [n_memory_slots]

            # eqn 14
            w_t = (1 - g_t) * w_previous + g_t * w_hat  # [n_memory_slots]

            # eqn 15
            c = T.dot(M_previous, w_t)  # [memory_size]

            ### EXTERNAL MEMORY UPDATE
            # eqn 16
            v = T.dot(self.Wv, h_tm1) + self.bv  # [memory_size]

            # eqn 17
            e = T.nnet.sigmoid(T.dot(self.We, h_tm1) + self.be)  # [n_memory_slots]
            f = 1. - w_t * e  # [n_memory_slots]

            # select for slots reserved for questions
            n = num_slots_reserved_for_questions

            def set_subtensor(is_question):
                if is_question:
                    M_t = M_previous[:, :n]
                    f_t = f[:n]
                    u_t = w_t[:n]
                else:
                    M_t = M_previous[:, n:]
                    f_t = f[n:]
                    u_t = w_t[n:]

                u_t = u_t.dimshuffle('x', 0)
                v_t = v.dimshuffle(0, 'x')

                # eqn 19
                return T.set_subtensor(M_t, T.dot(M_t, T.diag(f_t)) + T.dot(v_t, u_t))

            # M_t = ifelse(is_question, set_subtensor(True), set_subtensor(False))
            M_t = set_subtensor(is_question)

            # f_diag = T.diag(f)
            # M_t = T.dot(M_previous, f_diag) + T.dot(v.dimshuffle(0, 'x'), w_t.dimshuffle('x', 0))
            # M_t = T.set_subtensor(M_t[:, :n], M_t[:, :n] + 1)

            # eqn 9
            h_t = T.nnet.sigmoid(T.dot(self.Wx, x_t) + T.dot(self.Wh, c) + self.bh)

            # eqn 10?
            s_t = T.nnet.softmax(T.dot(self.W, h_t) + self.b)

            M_t_print = Print('M_t')(M_t)
            return [h_t, s_t, w_t, M_t]

        ask_question, train = (lambda x, h, w, m: recurrence(x, h, w, m, is_question)
                               for is_question in (True, False))

        [_, _, _, M], _ = theano.scan(fn=ask_question,
                                      sequences=x,
                                      outputs_info=[self.h0, None, self.w0, self.M],
                                      name='ask_scan')

        [_, s, _, _], _ = theano.scan(fn=train,
                                      sequences=x,
                                      outputs_info=[self.h0, None, self.w0, self.M],
                                      name='train_scan')

        self.get_y = theano.function(inputs=[y], outputs=y)
        # self.get_m = theano.function(inputs=[idxs, is_question], outputs=M.shape)

        p_y_given_x_last_word = s[-1, 0, :]
        p_y_given_x_sentence = s[:, 0, :]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        # nll = -T.mean(T.log(p_y_given_x_lastword)[y])
        # CHANGED
        s = s.flatten(ndim=2)
        s_print = Print("s")(s)
        y_print = Print("y")(y)

        counts = T.extra_ops.bincount(y, assert_nonneg=True)
        weights = 1.0 / (counts[y] + 1) * T.neq(y, 0)
        print_weights = Print("weights")(weights)
        losses = T.nnet.categorical_crossentropy(s, y)
        loss = lasagne.objectives.aggregate(losses, weights, mode='normalized_sum')
        # self.get_nll = theano.function([idxs, y, is_question],
        #                                outputs=nll,
        #                                allow_input_downcast=True)
        # CHANGED
        updates = lasagne.updates.adadelta(loss, self.params, lr)

        # theano functions
        self.ask_question = theano.function(inputs=[idxs],
                                            givens=[(self.M, self.M0)],
                                            updates=[(self.M, M[-1, :, :])])

        self.classify = theano.function(inputs=[idxs],
                                        outputs=y_pred,
                                        on_unused_input='warn')

        self.train = theano.function(inputs=[idxs, y, lr],
                                     outputs=loss,
                                     updates=updates,
                                     on_unused_input='warn',
                                     allow_input_downcast=True)

        self.normalize = theano.function(
            inputs=[],
            updates={self.emb: self.emb / T.sqrt((self.emb ** 2).sum(axis=1)).dimshuffle(0,
                                                                                         'x')})

    def save(self, folder):
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())
