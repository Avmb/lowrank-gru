# lowrank-gru
Gated Recurrent Unit with Low-rank matrix factorization

Paper: http://arxiv.org/abs/1603.03116

A reparametrization of the Gated Recurrent Unit (Cho et al. 2014, http://arxiv.org/abs/1406.1078) where the recurrent matrices are constrained to be low-rank. Reduces data complexity and memory usage.

This code is based on the code for the Unitary Evolution Recurrent Neural Networks (Arjovsky et al. 2015 http://arxiv.org/abs/1511.06464) in order to facilitate a direct comparison. Original repository is https://github.com/amarshah/complex_RNN .

Our model performs comparably or better than Unitary Evolution Recurrent Neural Networks on all the tasks we tested (memory, addition and randomly-permuted sequential MNIST) for similar number of parameters.

Notes:

Requires Theano: http://www.deeplearning.net/software/theano/

File fftconv.py is derived from Theano and is therefore under Theano licence. This file is needed only for the baseline uRNN model and not for out Low-rank GRU and Low-rank plus diagonal GRU models.

