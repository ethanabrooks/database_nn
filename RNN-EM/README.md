### Recurrent Neural Networks with External Memory

This is a Theano implementation of the RNN-EM model as described in [this paper](http://research.microsoft.com/pubs/246720/rnn_em.pdf).

This repository uses code from [`mesnilgr/is13`](https://github.com/mesnilgr/is13) to load the ATIS dataset.

### Usage
To run with ATIS dataset:
```
python main.py
```
To run with Jeopardy dataset:

```
python main.py --dataset=jeopardy
```

By default, `s.hidden_size` and `s.emb_size` are 100. If the number of questions is capped at 1000, `vocsize` is 33811 and `nsentences` is 9881.
Consequently, the default dimensions of matrices within the model are:

```
self.emb = (vocsize + 1) x 100
x = 1 x (100 * 7)
self.Wg = 100 x (100 * 7)
self.bg = 100
x_t = 100 * 7 (??)
g_t = 100
self.Wk = 100 x 100
self.Wx = 100 x (100 * 7)
```