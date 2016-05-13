import numpy
import os
import theano
from theano.printing import Print
from theano import tensor as T
import numpy as np

window_radius = 2
window_diameter = window_radius * 2 + 1
num_embeddings = 24
embedding_dim = 2
n_memory_slots = 6
n = 5
hidden_size = 4
memory_size = 8
nclasses = 3

weights = {
    'Wg': (window_diameter * embedding_dim, n_memory_slots),
    'Wk': (hidden_size, memory_size),
    'Wb': (hidden_size, 1),
    'Wv': (hidden_size, memory_size),
    'We': (hidden_size, n_memory_slots),
    'Wx': (window_diameter * embedding_dim, hidden_size),
    'Wh': (memory_size, hidden_size),
    'h0': hidden_size,
    'w0': (n_memory_slots,),
    'M0': (memory_size, n_memory_slots),
    'W':  (hidden_size, nclasses)
}

biases = {
    'bg': n_memory_slots,
    'bk': memory_size,
    'bb': 1,
    'bv': memory_size,
    'be': n_memory_slots,
    'bh': hidden_size,
    'b': nclasses
}


def cdist(tensor, matrix):
    tensor_norms = tensor.norm(2, axis=1)
    matrix_norms = T.shape_padright(matrix.norm(2, axis=1))
    return 1. - T.batched_dot(matrix, tensor) / (matrix_norms * tensor_norms)


def random_shared(shape):
    return theano.shared(numpy.ones(shape, dtype=theano.config.floatX))


def zeros_shared(shape):
    return theano.shared(numpy.zeros(shape, dtype=theano.config.floatX))


emb = T.constant(np.tile(np.arange(20).reshape(-1, 1), embedding_dim))

for (attributes, initializer) in ((weights, random_shared),
                                  (biases, zeros_shared)):
    for key in attributes:
        exec key + '= initializer(attributes[key])'

names = weights.keys() + biases.keys()
params = map(eval, names)

questions, docs = T.imatrices(2)  # as many columns as context window size/lines as words in the sentence
is_question = True
inputs = questions if is_question else docs


def repeat_for_each_instance(param):
    """ repeat param along new axis once for each instance """
    return T.repeat(T.shape_padleft(param), repeats=inputs.shape[0], axis=0)


h0, w0, M0 = map(repeat_for_each_instance, [h0, w0, M0])


def recurrence(i, h_tm1, w_previous, M_previous, is_question):
    # get representation of word window
    idxs = questions if is_question else docs  # [instances, bucket_width]
    pad = T.zeros((idxs.shape[0], window_radius), dtype='int32')
    padded = T.concatenate([pad, idxs, pad], axis=1)
    window = padded[:, i:i + window_diameter]  # [instances, window_diameter]
    x_t = emb[window].flatten(ndim=2)  # [instances, window_diameter * embedding_dim]

    # eqn not specified in paper (see eqn 14)
    # TODO: should this be conditioned on h not x?
    # g_t = T.nnet.sigmoid(T.dot(Wg, x_t) + bg)
    g_t = T.dot(x_t, Wg) + bg  # [instances, n_memory_slots]

    ### EXTERNAL MEMORY READ
    # eqn 11
    k = T.dot(h_tm1, Wk) + bk  # [instances, memory_size]

    # eqn 13
    beta_pre = T.dot(h_tm1, Wb) + bb
    beta = T.log(1 + T.exp(beta_pre))
    beta = T.addbroadcast(beta, 1)  # [instances, 1]
    #
    # # eqn 12
    w_hat = cdist(M_previous, k)  # [instances, n_memory_slots]
    w_hat = T.exp(beta * w_hat)
    w_hat /= T.shape_padright(T.sum(w_hat, axis=1))  # [n_memory_slots]

    # eqn 14
    w_t = (1 - g_t) * w_previous + g_t * w_hat  # [instances, n_memory_slots]

    # # if we are reading a question, we only read from and write to question memory
    if is_question:
        M_previous = M_previous[:, :, :n]  # [instances, memory_size, n]
        w_previous = w_previous[:, :n]  # [instances, n]

    # eqn 15
    c = T.batched_dot(M_previous, w_previous)  # [instances, memory_size]

    ### MODEL INPUT AND OUTPUT
    # eqn 9
    h_t = T.dot(x_t, Wx) + T.dot(c, Wh) + bh  # [instances, hidden_size]

    # eqn 10
    s_t = T.nnet.softmax(T.dot(h_t, W) + b)  # [instances, nclasses]

    ### EXTERNAL MEMORY UPDATE

    # eqn 17
    e = T.nnet.sigmoid(T.dot(h_tm1, We) + be)  # [instances, n_memory_slots]
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
    v_t = T.dot(h_t, Wv) + bv  # [instances, memory_size]

    u_t = u_t.dimshuffle(0, 'x', 1)  # [instances, 1, ?]
    v_t = v_t.dimshuffle(0, 1, 'x')  # [instances, memory_size, 1]
    f_t = f_t.dimshuffle(0, 'x', 1)  # [instances, 1, ?]

    # eqn 19
    M_update = M_t * f_t + T.batched_dot(v_t, u_t)  # [instances, memory_size, ?]
    M_t = T.set_subtensor(M_t, M_update)  # [instances, memory_size, n_memory_slots]

    return [k, M_previous]


f = theano.function([inputs], outputs=recurrence(0, h0, w0, M0, is_question),
                    on_unused_input='ignore')

res = f(numpy.arange(12, dtype='int32').reshape(2, -1))
for whtever in res:
    print('-' * 10)
    print whtever
