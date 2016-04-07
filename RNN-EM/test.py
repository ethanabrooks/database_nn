import numpy as np
from rnn_em import model

word_batch = np.array([[-1, -1, -1,  5, 17, 18, 19]], dtype='int32')
label = 1
learn_rate = 0.0627142536696559
is_question = True

rnn = model(hidden_size=100,
            nclasses=3,
            num_embeddings=30,
            embedding_dim=100,
            window_size=7,
            memory_size=40,
            n_memory_slots=8)

loss = rnn.train(word_batch, label, learn_rate, is_question)
print(loss)
