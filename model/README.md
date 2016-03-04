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
