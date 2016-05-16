from __future__ import print_function

from functools import partial

import lasagne
import numpy
import os
import theano
from theano import tensor as T
from theano.printing import Print


def cosine_dist(tensor, matrix):
    """
    Along axis 1 for both inputs.
    Assumes dimensions 0 and 1 are equal
    """
    matrix_norm = T.shape_padright(matrix.norm(2, axis=1))
    tensor_norm = tensor.norm(2, axis=1)
    return T.batched_dot(matrix, tensor) / (matrix_norm * tensor_norm)

# noinspection PyPep8Naming
class Model(object):
    def __init__(self,
                 hidden_size=4,
                 nclasses=3,
                 num_embeddings=100,
                 embedding_dim=2,
                 window_size=7,
                 memory_size=6,
                 n_memory_slots=8):


        questions, docs = T.itensor3s('questions', 'docs')
        y_true_matrix = T.imatrix('y_true')

        n_question_slots = int(n_memory_slots / 4)  # TODO derive this from an arg
        n_doc_slots = n_memory_slots - n_question_slots
        n_instances = questions.shape[1]

        self.window_size = window_size

        randoms = {
            # attr: shape
            'emb': (num_embeddings + 1, embedding_dim),
            'Wg_q': (window_size * embedding_dim, n_question_slots),
            'Wg_d': (window_size * embedding_dim, n_doc_slots),
            'Wk': (hidden_size, memory_size),
            'Wb': (hidden_size, 1),
            'Wv': (hidden_size, memory_size),
            'We_q': (hidden_size, n_question_slots),
            'We_d': (hidden_size, n_doc_slots),
            'Wx': (window_size * embedding_dim, hidden_size),
            'Wh': (memory_size, hidden_size),
            'W': (hidden_size, nclasses),
            'h0': hidden_size,
            'w_q': (n_question_slots,),
            'w_d': (n_doc_slots,),
            'M_q': (memory_size, n_question_slots),
            # TODO can we set M0 to zeros without having issues with cosine_dist?
            'M_d': (memory_size, n_doc_slots)  # TODO can we set M0 to zeros without having issues with cosine_dist?
        }

        zeros = {
            # attr: shape
            'bg_q': n_question_slots,
            'bg_d': n_doc_slots,
            'bk': memory_size,
            'bb': 1,
            'bv': memory_size,
            'be_q': n_question_slots,
            'be_d': n_doc_slots,
            'bh': hidden_size,
            'b': nclasses
        }

        def random_shared(shape):
            return theano.shared(
                0.2 * numpy.random.uniform(-1.0, 1.0, shape).astype(theano.config.floatX))

        def zeros_shared(shape):
            return theano.shared(numpy.zeros(shape, dtype=theano.config.floatX))

        for key in randoms:
            # create an attribute with associated shape and random values
            setattr(self, key, random_shared(randoms[key]))

        for key in zeros:
            # create an attribute with associated shape and values = 0
            setattr(self, key, zeros_shared(zeros[key]))

        def repeat_for_each_instance(param):
            """ repeat param along new axis once for each instance """
            return T.repeat(T.shape_padleft(param), repeats=n_instances, axis=0)

        for key in 'h0 w_q M_q w_d M_d'.split():
            setattr(self, key, repeat_for_each_instance(self.__getattribute__(key)))

        self.names = randoms.keys() + zeros.keys()
        self.params = [eval('self.' + name) for name in 'bh'.split()]

        def recurrence(i, h_tm1, w_q, M_q, w_d=None, M_d=None, is_question=True):
            """
            notes
            Headers from paper in all caps
            mem = n_question slots if is_question else n_doc_slots

            :param i: center index of sliding window
            :param h_tm1: h_{t-1} (hidden state)
            :param w_q: attention weights for question memory
            :param M_q: question memory
            :param w_d: attention weights for docs memory
            :param M_d: docs memory
            :param is_question: we use different parts of memory when working with a question
            :return: [y_t = model outputs,
                      i + 1 = increment index,
                      h_t w_t, M_t (see above)]
            """
            if not is_question:
                assert w_d is not None and M_d is not None

            # get representation of word window
            idxs = questions if is_question else docs  # [instances, bucket_width]
            pad = T.zeros((idxs.shape[0], window_radius), dtype='int32')
            padded = T.concatenate([pad, idxs, pad], axis=1)
            window = padded[:, i:i + window_size]  # [instances, window_size]
            x_t = self.emb[window].flatten(ndim=2)  # [instances, window_size * embedding_dim]

            # EXTERNAL MEMORY READ
            # eqn 15

            if is_question:
                M_read = M_q  # [instances, memory_size, n_question_slots]
                w_read = w_q  # [instances, n_question_slots]
            else:
                M_read = T.concatenate([M_q, M_d], axis=2)  # [instances, memory_size, n_doc_slots]
                w_read = T.concatenate([w_q, w_d], axis=1)  # [instances, n_doc_slots]

            c = T.batched_dot(M_read, w_read)  # [instances, memory_size]

            def get_attention(Wg, bg, M, w):
                g_t = T.nnet.sigmoid(T.dot(x_t, Wg) + bg)  # [instances, mem]

                # eqn 11
                k = T.dot(h_tm1, self.Wk) + self.bk  # [instances, memory_size]

                # eqn 13
                beta_pre = T.dot(h_tm1, self.Wb) + self.bb
                beta = T.log(1 + T.exp(beta_pre))
                beta = T.addbroadcast(beta, 1)  # [instances, 1]

                # eqn 12
                w_hat = cosine_dist(M, k)
                w_hat = T.exp(beta * w_hat)
                w_hat /= T.shape_padright(T.sum(w_hat, axis=1))  # [instances, mem]
                w_hat = T.switch(T.isnan(w_hat), numpy.inf, w_hat)
                w_hat = Print('w_hat')(w_hat)

                # eqn 14
                return (1 - g_t) * w + g_t * w_hat  # [instances, mem]

            w_q = get_attention(self.Wg_q, self.bg_q, M_q, w_q)  # [instances, n_question_slots]
            if not is_question:
                w_d = get_attention(self.Wg_d, self.bg_d, M_d, w_d)  # [instances, n_doc_slots]

            # MODEL INPUT AND OUTPUT
            # eqn 9
            h_t = T.dot(x_t, self.Wx) + T.dot(c, self.Wh) + self.bh  # [instances, hidden_size]

            # eqn 10
            y_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b)  # [instances, nclasses]

            # EXTERNAL MEMORY UPDATE
            def update_memory(We, be, w_update, M_update):
                # eqn 17
                e = T.nnet.sigmoid(T.dot(h_tm1, We) + be)  # [instances, mem]
                f = 1. - w_update * e  # [instances, mem]

                # eqn 16
                v_t = T.dot(h_t, self.Wv) + self.bv  # [instances, memory_size]

                # need to add broadcast layers for memory update
                f_t = f.dimshuffle(0, 'x', 1)  # [instances, 1, mem]
                u_t = w_update.dimshuffle(0, 'x', 1)  # [instances, 1, mem]
                v_t = v_t.dimshuffle(0, 1, 'x')  # [instances, memory_size, 1]

                # eqn 19
                return M_update * f_t + T.batched_dot(v_t, u_t)  # [instances, memory_size, mem]

            M_q = update_memory(self.We_q, self.be_q, w_q, M_q)
            attention_and_memory = [w_q, M_q]
            if not is_question:
                M_d = update_memory(self.We_d, self.be_d, w_d, M_d)
                attention_and_memory += [w_d, M_d]
            return [y_t, h_t] + attention_and_memory

        outputs_info = [None, self.h0, self.w_q, self.M_q]
        ask_question = partial(recurrence, is_question=True)
        answer_question = partial(recurrence, is_question=False)

        [_, h, w, M], _ = theano.scan(fn=ask_question,
                                      outputs_info=outputs_info,
                                      n_steps=questions.shape[1],
                                      name='ask_scan')

        outputs_info[1:] = [param[-1, :, :] for param in (h, w, M)]

        output, _ = theano.scan(fn=answer_question,
                                outputs_info=outputs_info + [self.w_d, self.M_d],
                                n_steps=docs.shape[1],
                                name='train_scan')

        y_dist = output[0].dimshuffle(2, 1, 0).flatten(ndim=2).T
        y_true = y_true_matrix.ravel()
        counts = T.extra_ops.bincount(y_true, assert_nonneg=True)
        weights = 1.0 / (counts[y_true] + 1) * T.neq(y_true, 0)
        losses = T.nnet.categorical_crossentropy(y_dist, y_true)
        loss = lasagne.objectives.aggregate(losses, weights)
        updates = lasagne.updates.adadelta(loss, self.params)

        # theano functions
        self.predict = theano.function(inputs=[questions, docs],
                                       outputs=y_dist.argmax(axis=1))

        self.train = theano.function(inputs=[questions, docs, y_true_matrix],
                                     outputs=[y_dist, loss],
                                     updates=updates,
                                     allow_input_downcast=True)

        self.test = self.train

        normalized_embeddings = self.emb / T.sqrt((self.emb ** 2).sum(axis=1)).dimshuffle(0, 'x')
        self.normalize = theano.function(inputs=[],
                                         updates={self.emb: normalized_embeddings})

    def sliding_window(self, matrix):
        """
        :param matrix: 2D tensor. See example, below.
        :return 3D tensor:

        matrix       padded          returns
        [0 1 2]      [0 0 1 2 0]     [ [0 0 1]  [0 1 2]  [1 2 0]
        [3 4 5]  =>  [0 3 4 5 0]  =>   [0 3 4]  [3 4 5]  [4 5 0]
        [6 7 8]      [0 6 7 8 0]       [0 6 7]  [6 7 8]  [7 8 0] ]
        """
        window_idxs = numpy.fromfunction(lambda i, j: i + j,
                                         (matrix.shape[1], self.window_size),
                                         dtype='int32')
        radius = self.window_size // 2
        padded = numpy.pad(matrix,
                           pad_width=[(0, 0), (radius, radius)],
                           mode='constant')
        return numpy.swapaxes(padded.T[window_idxs], 1, 2)

    def save(self, folder):
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())


if __name__ == '__main__':
    rnn = Model()
    questions = numpy.ones((10, 64), dtype='int32')
    docs = numpy.ones((10, 256), dtype='int32')

    questions, docs_windows = map(rnn.sliding_window, (questions, docs))
    print("questions", questions.shape)  # [words/instance, n_instances, window_size]
    print("docs", docs_windows.shape)  # [words/instance, n_instances, window_size]
    print("targets", docs.shape)  # [n_instances, words/instance]
    for result in rnn.test(questions, docs_windows, docs):
        print('-' * 10)
        print(result)
    rnn.normalize()
