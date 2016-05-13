from __future__ import print_function

from functools import partial

import lasagne
import numpy
import os
import theano
from theano.printing import Print
from theano import tensor as T


def cosine_dist(tensor, matrix):
    """
    Along axis 1 for both inputs.
    Assumes first dimansion ~ different instances.
    """
    tensor_norms = tensor.norm(2, axis=1)
    matrix_norms = T.shape_padright(matrix.norm(2, axis=1))
    return 1. - T.batched_dot(matrix, tensor) / (matrix_norms * tensor_norms)


# noinspection PyPep8Naming
class Model(object):
    def __init__(self,
                 hidden_size=4,
                 nclasses=3,
                 num_embeddings=100,
                 embedding_dim=2,
                 window_radius=1,
                 memory_size=8,
                 n_memory_slots=6):

        questions, docs, y_true = T.imatrices(3)
        window_diameter = window_radius * 2 + 1
        num_slots_reserved_for_questions = int(n_memory_slots / 4)  # TODO derive this from an arg

        randoms = {
            # attr: shape
            'emb': (num_embeddings + 1, embedding_dim),
            'Wg': (window_diameter * embedding_dim, n_memory_slots),
            'Wk': (hidden_size, memory_size),
            'Wb': (hidden_size, 1),
            'Wv': (hidden_size, memory_size),
            'We': (hidden_size, n_memory_slots),
            'Wx': (window_diameter * embedding_dim, hidden_size),
            'Wh': (memory_size, hidden_size),
            'W': (hidden_size, nclasses),
            'h0': hidden_size,
            'w0': (n_memory_slots,),
            'M0': (memory_size, n_memory_slots)  # TODO can we set M0 to zeros without having issues with cosine_dist?
        }

        zeros = {
            # attr: shape
            'bg': n_memory_slots,
            'bk': memory_size,
            'bb': 1,
            'bv': memory_size,
            'be': n_memory_slots,
            'bh': hidden_size,
            'b': nclasses
        }

        def random_shared(shape):
            return theano.shared(
                0.2 * numpy.random.uniform(-1.0, 1.0, shape).astype(theano.config.floatX))

        def zeros_shared(shape):
            return theano.shared(numpy.zeros(shape, dtype=theano.config.floatX))

        for key in randoms:
            setattr(self, key, random_shared(randoms[key]))

        for key in zeros:
            setattr(self, key, zeros_shared(zeros[key]))

        self.names = randoms.keys() + zeros.keys()
        self.params = [eval('self.' + name) for name in 'bh'.split()]

        def recurrence(i, h_tm1, w_previous, M_previous, is_question):
            """
            :param is_question: we use different parts of memory when working with a question
            :param i: center index of sliding window
            :param h_tm1: h_{t-1} (hidden state)
            :param w_previous: attention weights. see paper
            :param M_previous: memory. see paper
            :return: [y_t = model outputs,
                      i + 1 = increment index,
                      h_t w_t, M_t (see above)]
            """

            # get representation of word window
            idxs = questions if is_question else docs  # [instances, bucket_width]
            pad = T.zeros((idxs.shape[0], window_radius), dtype='int32')
            padded = T.concatenate([pad, idxs, pad], axis=1)
            window = padded[:, i:i + window_diameter]  # [instances, window_diameter]
            x_t = self.emb[window].flatten(ndim=2)  # [instances, window_diameter * embedding_dim]

            # eqn not specified in paper (see eqn 14)
            # TODO: should this be conditioned on h not x?
            g_t = T.nnet.sigmoid(T.dot(x_t, self.Wg) + self.bg)  # [instances, n_memory_slots]

            ### EXTERNAL MEMORY READ
            # eqn 11
            k = T.dot(h_tm1, self.Wk) + self.bk  # [instances, memory_size]

            # eqn 13
            beta_pre = T.dot(h_tm1, self.Wb) + self.bb
            beta = T.log(1 + T.exp(beta_pre))
            beta = T.addbroadcast(beta, 1)  # [instances, 1]

            # eqn 12
            w_hat = cosine_dist(M_previous, k)
            w_hat = T.exp(beta * w_hat)
            w_hat /= T.shape_padright(T.sum(w_hat, axis=1))  # [n_memory_slots]

            # eqn 14
            w_t = (1 - g_t) * w_previous + g_t * w_hat  # [instances, n_memory_slots]

            n = num_slots_reserved_for_questions

            # if we are reading a question, we only read from and write to question memory
            if is_question:
                M_previous = M_previous[:, :, :n]  # [instances, memory_size, n]
                w_previous = w_previous[:, :n]  # [instances, n]

            # eqn 15
            c = T.batched_dot(M_previous, w_previous)  # [instances, memory_size]

            ### MODEL INPUT AND OUTPUT
            # eqn 9
            h_t = T.dot(x_t, self.Wx) + T.dot(c, self.Wh) + self.bh  # [instances, hidden_size]

            # eqn 10
            y_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b)  # [instances, nclasses]

            ### EXTERNAL MEMORY UPDATE

            # eqn 17
            e = T.nnet.sigmoid(T.dot(h_tm1, self.We) + self.be)  # [instances, n_memory_slots]
            f = 1. - w_t * e  # [instances, n_memory_slots]

            # if we are reading a question, we only read from and write to question memory
            if is_question:
                M_t = M_previous[:, :, :n]  # [instances, memory_size, n]
                f_t = f[:, :n]  # [instances, n]
                u_t = w_t[:, :n]  # [instances, n]

            # if not a question, we read from all memory and only write to non-question memory
            else:
                M_t = M_previous[:, :, n:]  # [instances, memory_size, n_memory_slots - n]
                f_t = f[:, n:]  # [instances, n_memory_slots - n]
                u_t = w_t[:, n:]  # [instances, n_memory_slots - n]

            # eqn 16
            v_t = T.dot(h_t, self.Wv) + self.bv  # [instances, memory_size]

            # need to add broadcast layers for memory update
            f_t = f_t.dimshuffle(0, 'x', 1)  # [instances, 1, ?]
            u_t = u_t.dimshuffle(0, 'x', 1)  # [instances, 1, ?]
            v_t = v_t.dimshuffle(0, 1, 'x')  # [instances, memory_size, 1]

            # M_t * f_t multiplies f_t[i] times M_t[:, :, i]
            # T.batched_dot(v_t, u_t) takes the outer product of v_t and u_t
            # for each instance.
            M_update = M_t * f_t + T.batched_dot(v_t, u_t)  # [instances, memory_size, ?]

            # eqn 19
            M_t = T.set_subtensor(M_t, M_update)  # [instances, memory_size, n_memory_slots]
            return [y_t, i + 1, h_t, w_t, M_t]

        def repeat_for_each_instance(param):
            """ repeat param along new axis once for each instance """
            return T.repeat(T.shape_padleft(param), repeats=questions.shape[0], axis=0)

        ask_question = partial(recurrence, is_question=True)
        answer_question = partial(recurrence, is_question=False)
        outputs_info = [None, T.constant(0)] + map(repeat_for_each_instance,
                                                   [self.h0, self.w0, self.M0])

        [_, _, h, w, M], _ = theano.scan(fn=ask_question,
                                         outputs_info=outputs_info,
                                         n_steps=questions.shape[1],
                                         name='ask_scan')

        outputs_info[2:] = [param[-1, :, :] for param in (h, w, M)]

        [y, _, _, _, _], _ = theano.scan(fn=answer_question,
                                         outputs_info=outputs_info,
                                         n_steps=docs.shape[1],
                                         name='train_scan')

        y_pred = T.argmax(y, axis=2).T
        counts = T.extra_ops.bincount(y_true.ravel(), assert_nonneg=True)
        weights = 1.0 / (counts[y_true] + 1) * T.neq(y_true, 0)
        losses = T.nnet.binary_crossentropy(y_pred, y_true)
        loss = lasagne.objectives.aggregate(losses, weights)
        updates = lasagne.updates.adadelta(loss, self.params)

        self.test = theano.function([questions, docs, y_true],
                                    outputs=[self.bh] + updates.values(),
                                    on_unused_input='ignore')

        # theano functions
        self.predict = theano.function(inputs=[questions, docs],
                                       outputs=y_pred,
                                       on_unused_input='warn')

        self.train = theano.function(inputs=[questions, docs, y_true],
                                     outputs=[y_pred, loss],
                                     updates=updates,
                                     on_unused_input='warn',
                                     allow_input_downcast=True)

        normalized_embeddings = self.emb / T.sqrt((self.emb ** 2).sum(axis=1)).dimshuffle(0, 'x')
        self.normalize = theano.function(inputs=[],
                                         updates={self.emb: normalized_embeddings})

    def save(self, folder):
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())


if __name__ == '__main__':
    rnn = Model()
    questions = numpy.ones((3, 2), dtype='int32')
    for result in rnn.test(questions, questions, questions):
        print('-' * 10)
        print(result)
        # rnn.normalize()
